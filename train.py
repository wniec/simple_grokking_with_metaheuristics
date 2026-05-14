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

torch.manual_seed(2)

P = 53
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
        "--epochs",
        type=int,
        default=10_000,
        help="Training epochs / generations (default: 10000)",
    )

    g = parser.add_argument_group("gradient options")
    g.add_argument(
        "--lr", type=float, default=3e-2, help="Learning rate for AdamW (default: 3e-2)"
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

    return parser.parse_args()


class Model(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(P, hidden_dim)
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, P)

    def forward(self, x):
        x = self.embedding(x).flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.readout(x)
        return x


if __name__ == "__main__":
    args = parse_args()

    weight_decay = 3e-2 if args.grok else 5
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

    model = Model()
    criterion = nn.CrossEntropyLoss()

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
        gradient_alg.run(**common, lr=args.lr)
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
