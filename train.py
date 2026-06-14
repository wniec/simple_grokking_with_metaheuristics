import argparse
import torch
import torch.nn as nn
from log import Logger
import algorithms.gradient as gradient_alg
import algorithms.cmaes as cmaes_alg
import algorithms.cpso as cpso_alg
import algorithms.de as de_alg
import algorithms.g3pcx as g3pcx_alg
import algorithms.moea as moea_alg
import algorithms.lon as lon_alg

torch.manual_seed(2)

P = 3
train_frac = 0.6


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grokking experiments with modular addition"
    )
    parser.add_argument(
        "--algo",
        choices=["gradient", "cmaes", "cpso", "de", "g3pcx", "moea"],
        default="gradient",
        help="Optimization algorithm (default: gradient)",
    )
    parser.add_argument(
        "--grok",
        action="store_true",
        help="Grokking mode: low weight decay (0.03 vs 5)",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable metric logging and model checkpointing",
    )
    parser.add_argument(
        "--lon",
        action="store_true",
        help="Analyze the fitness landscape with a Local Optima Network (lonkit) "
        "instead of training. Best on a tiny model, e.g. --arch fft --hidden-dim 4.",
    )
    parser.add_argument(
        "--arch",
        choices=["mlp", "cnn", "fft"],
        default="mlp",
        help="Model architecture (default: mlp)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension of the model (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10_000,
        help="Training epochs / generations (default: 10000)",
    )

    g = parser.add_argument_group("gradient options")
    g.add_argument(
        "--lr", type=float, default=8e-2, help="Learning rate for AdamW (default: 3e-2)"
    )

    evo = parser.add_argument_group("evolutionary options (cmaes / cpso / de / g3pcx)")
    evo.add_argument(
        "--bound-norm",
        type=float,
        default=5.0,
        help="Search space boundary ±bound-norm (default: 5.0)",
    )
    evo.add_argument(
        "--n-individuals",
        type=int,
        default=None,
        help="Population size (default: auto from ndim)",
    )
    evo.add_argument(
        "--seed", type=int, default=2, help="RNG seed for the optimizer (default: 2)"
    )
    evo.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        help="Initial step size (cmaes) / initial population spread around model weights (cpso, de, g3pcx) (default: 0.5)",
    )

    cmaes = parser.add_argument_group("cmaes options")
    cmaes.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Low-rank approximation rank for CMA-ES (0 = disabled, >0 = reshape params to sqrt(ndim)×sqrt(ndim) and optimize U@V.T with given rank)",
    )

    pso = parser.add_argument_group("cpso options")
    pso.add_argument(
        "--cognition",
        type=float,
        default=1.49,
        help="Cognitive learning rate (default: 1.49)",
    )
    pso.add_argument(
        "--society",
        type=float,
        default=1.49,
        help="Social learning rate (default: 1.49)",
    )

    de = parser.add_argument_group("de options (TDE)")
    de.add_argument(
        "--f", type=float, default=0.99, help="Mutation factor (default: 0.99)"
    )
    de.add_argument(
        "--cr", type=float, default=0.85, help="Crossover rate (default: 0.85)"
    )
    de.add_argument(
        "--tm",
        type=float,
        default=0.05,
        help="Trigonometric mutation probability (default: 0.05)",
    )

    ga = parser.add_argument_group("g3pcx options")
    ga.add_argument(
        "--n-offsprings",
        type=int,
        default=2,
        help="Offspring per generation for G3PCX (default: 2)",
    )
    ga.add_argument(
        "--n-parents",
        type=int,
        default=3,
        help="Parents selected per generation for G3PCX (default: 3)",
    )

    lon = parser.add_argument_group(
        "fitness-landscape (lonkit) options, used with --lon"
    )
    lon.add_argument(
        "--lon-runs",
        type=int,
        default=100,
        help="Number of independent basin-hopping runs for the LON (default: 100)",
    )
    lon.add_argument(
        "--lon-no-change",
        type=int,
        default=250,
        help="Basin-hopping iterations without improvement before stopping (default: 250)",
    )
    lon.add_argument(
        "--lon-step-size",
        type=float,
        default=0.1,
        help="Basin-hopping perturbation step size (default: 0.1)",
    )

    return parser.parse_args()


class MLPModel(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(P, hidden_dim)
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, P)

    def forward(self, x):
        x = self.embedding(x).flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        return self.readout(x)


class CNNModel(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(P, hidden_dim)
        # kernel_size=2 covers both token positions at once
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2)
        self.readout = nn.Linear(hidden_dim, P)

    def forward(self, x):
        x = self.embedding(x)  # (batch, 2, hidden_dim)
        x = x.permute(0, 2, 1)  # (batch, hidden_dim, 2)
        x = torch.relu(self.conv(x))  # (batch, hidden_dim, 1)
        x = x.squeeze(-1)  # (batch, hidden_dim)
        return self.readout(x)


class FFTModel(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(P, hidden_dim)
        self.readout = nn.Linear(hidden_dim, P)

    def forward(self, x):
        e = self.embedding(x)  # (batch, 2, hidden_dim)
        Ea = torch.fft.rfft(e[:, 0, :])  # (batch, hidden_dim//2+1) complex
        Eb = torch.fft.rfft(e[:, 1, :])
        eab = torch.fft.irfft(
            Ea * Eb, n=e.shape[-1]
        )  # circular conv → (batch, hidden_dim)
        return self.readout(torch.relu(eab))


if __name__ == "__main__":
    args = parse_args()

    weight_decay = 6e-5 if args.grok else 1
    run_name = f"{args.algo}_{'grokking' if args.grok else 'comprehension'}"
    logger = Logger(run_name) if args.log else None

    X = torch.cartesian_prod(torch.arange(P), torch.arange(P))
    y = (X[:, 0] + X[:, 1]) % P
    shuffle = torch.randperm(len(X))
    X, y = X[shuffle], y[shuffle]
    X_train = X[: int(train_frac * len(X))]
    X_val = X[int(train_frac * len(X)) :]
    y_train = y[: int(train_frac * len(y))]
    y_val = y[int(train_frac * len(y)) :]

    arch = {"mlp": MLPModel, "cnn": CNNModel, "fft": FFTModel}[args.arch]
    model = arch(hidden_dim=args.hidden_dim)
    criterion = nn.CrossEntropyLoss()

    import math

    n_params = sum(p.numel() for p in model.parameters())
    if args.rank > 0:
        n = math.ceil(math.sqrt(n_params))
        search_dim = 2 * n * args.rank
        print(
            f"Model parameters: {n_params:,} | search dim: {search_dim:,} (low-rank rank={args.rank})"
        )
    else:
        print(f"Model parameters: {n_params:,} | search dim: {n_params:,}")

    if args.lon:
        # The landscape is a property of the (model, data, weight_decay) problem,
        # not of any optimizer, so name the run by architecture + regime.
        lon_name = f"{args.arch}_{'grokking' if args.grok else 'comprehension'}"
        lon_alg.run(
            model=model,
            criterion=criterion,
            X_train=X_train,
            y_train=y_train,
            weight_decay=weight_decay,
            run_name=lon_name,
            bound_norm=args.bound_norm,
            n_runs=args.lon_runs,
            n_iter_no_change=args.lon_no_change,
            step_size=args.lon_step_size,
            seed=args.seed,
        )
        raise SystemExit

    print("Train Loss, Acc | Val Loss, Acc")

    common = dict(
        model=model,
        criterion=criterion,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_epochs=args.epochs,
        weight_decay=weight_decay,
        logger=logger,
    )

    if args.algo == "gradient":
        gradient_alg.run(**common, lr=args.lr, rank=args.rank)
    elif args.algo == "cmaes":
        cmaes_alg.run(
            **common,
            sigma=args.sigma,
            bound_norm=args.bound_norm,
            n_individuals=args.n_individuals,
            seed=args.seed,
            rank=args.rank,
        )
    elif args.algo == "cpso":
        cpso_alg.run(
            **common,
            cognition=args.cognition,
            society=args.society,
            sigma=args.sigma,
            bound_norm=args.bound_norm,
            n_individuals=args.n_individuals,
            seed=args.seed,
        )
    elif args.algo == "de":
        de_alg.run(
            **common,
            f=args.f,
            cr=args.cr,
            tm=args.tm,
            sigma=args.sigma,
            bound_norm=args.bound_norm,
            n_individuals=args.n_individuals,
            seed=args.seed,
        )
    elif args.algo == "g3pcx":
        g3pcx_alg.run(
            **common,
            n_offsprings=args.n_offsprings,
            n_parents=args.n_parents,
            sigma=args.sigma,
            bound_norm=args.bound_norm,
            n_individuals=args.n_individuals,
            seed=args.seed,
        )
    elif args.algo == "moea":
        moea_alg.run(
            **common,
            sigma=args.sigma,
            bound_norm=args.bound_norm,
            n_individuals=args.n_individuals,
            seed=args.seed,
        )
