import torch
import torch.nn as nn
import numpy as np
import tqdm
from log import Logger
from sys import argv
from pypop7.optimizers.es.lmcmaes import LMCMAES

torch.manual_seed(2)
grok = "--grok" in argv
do_log = "--log" in argv
use_cmaes = "--cmaes" in argv
logger = Logger("grokking" if grok else "comprehension") if do_log else None

# Hyperparameters
num_epochs = int(argv[argv.index("--epochs") + 1]) if "--epochs" in argv else 10_000
learning_rate = 3e-2
weight_decay = 3e-2 if grok else 5
sigma = 0.5       # LMCMAES: initial step size
bound_norm = 5.0  # LMCMAES: search space bounded to [-bound_norm, bound_norm]
P = 53
train_frac = 0.6

# Data - sum of two numbers mod 53
X = torch.cartesian_prod(torch.arange(P), torch.arange(P))
y = (X[:, 0] + X[:, 1]) % P
shuffle = torch.randperm(len(X))
X, y = X[shuffle], y[shuffle]
X_train, X_val = X[: int(train_frac * len(X))], X[int(train_frac * len(X)) :]
y_train, y_val = y[: int(train_frac * len(y))], y[int(train_frac * len(y)) :]


# Model
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


model = Model()
criterion = nn.CrossEntropyLoss()


def train_gradient():
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    pbar = tqdm.trange(num_epochs, leave=True, position=0)
    for epoch in pbar:
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_acc = (y_pred.argmax(dim=1) == y_train).float().mean() * 100
            y_pred_val = model(X_val)
            val_loss = criterion(y_pred_val, y_val)
            val_acc = (y_pred_val.argmax(dim=1) == y_val).float().mean() * 100

        pbar.set_description(
            f"{loss:10.2f}, {train_acc:>3.0f} | {val_loss:>8.2f}, {val_acc:>4.0f}"
        )
        if do_log:
            logger.log(
                model=model,
                epoch=epoch,
                train_loss=loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
            )


def train_cmaes():
    ndim = sum(p.numel() for p in model.parameters())
    pop_size = 4 + int(3 * np.log(ndim))

    def set_params(x):
        x_t = torch.from_numpy(x.astype(np.float32))
        idx = 0
        with torch.no_grad():
            for p in model.parameters():
                n = p.numel()
                p.copy_(x_t[idx : idx + n].reshape(p.shape))
                idx += n

    def get_params():
        return np.concatenate([p.detach().numpy().ravel() for p in model.parameters()])

    eval_count = [0]
    best_loss = [float("inf")]
    best_x = [get_params()]
    pbar = tqdm.tqdm(total=num_epochs, leave=True, position=0)

    def fitness(x):
        set_params(x)
        with torch.no_grad():
            logits = model(X_train)
            loss = criterion(logits, y_train).item() + np.linalg.norm(x) * weight_decay

        if loss < best_loss[0]:
            best_loss[0] = loss
            best_x[0] = x.copy()

        eval_count[0] += 1
        if eval_count[0] % pop_size == 0:
            gen = eval_count[0] // pop_size
            set_params(best_x[0])
            with torch.no_grad():
                tr_logits = model(X_train)
                tr_loss = criterion(tr_logits, y_train)
                tr_acc = (tr_logits.argmax(1) == y_train).float().mean() * 100
                vl_logits = model(X_val)
                vl_loss = criterion(vl_logits, y_val)
                vl_acc = (vl_logits.argmax(1) == y_val).float().mean() * 100
            pbar.set_description(
                f"{tr_loss:10.2f}, {tr_acc:>3.0f} | {vl_loss:>8.2f}, {vl_acc:>4.0f}"
            )
            pbar.update(1)
            if do_log:
                logger.log(
                    model=model,
                    epoch=gen,
                    train_loss=tr_loss,
                    train_acc=tr_acc,
                    val_loss=vl_loss,
                    val_acc=vl_acc,
                )

        return loss

    print(f"ndim={ndim}, pop_size={pop_size}, sigma={sigma}, bound_norm={bound_norm}")
    problem = {
        "fitness_function": fitness,
        "ndim_problem": ndim,
        "lower_boundary": -bound_norm,
        "upper_boundary": bound_norm,
        "initial_lower_boundary": -bound_norm,
        "initial_upper_boundary": bound_norm,
    }
    options = {
        "max_function_evaluations": num_epochs * pop_size,
        "x": get_params(),
        "sigma": sigma,
        "seed_rng": 2,
        "verbose": False,
        "verbose_frequency": int(1e9),
    }
    results = LMCMAES(problem, options).optimize()
    pbar.close()

    set_params(results["best_so_far_x"])
    print(f"Best train loss: {results['best_so_far_y']:.4f}")


if __name__ == "__main__":
    print("Train Loss, Acc | Val Loss, Acc")
    if use_cmaes:
        train_cmaes()
    else:
        train_gradient()