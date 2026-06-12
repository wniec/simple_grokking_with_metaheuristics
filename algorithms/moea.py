import numpy as np
import torch
import tqdm

from algorithms._common import weight_stats
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize

from algorithms._common import get_params, set_params


class _GaussianSampling(Sampling):
    """Initialise population as Gaussian noise around the current model weights."""

    def __init__(self, x0, sigma):
        super().__init__()
        self.x0 = x0
        self.sigma = sigma

    def _do(self, problem, n_samples, **kwargs):
        X = self.x0 + np.random.randn(n_samples, len(self.x0)) * self.sigma
        return np.clip(X, problem.xl, problem.xu)


class _BiObjectiveProblem(Problem):
    """
    Objective 1: training loss (cross-entropy).
    Objective 2: mean squared weight norm  mean(x²).
    """

    def __init__(self, model, criterion, X_train, y_train, ndim, bound_norm):
        super().__init__(n_var=ndim, n_obj=2, xl=-bound_norm, xu=bound_norm)
        self.model = model
        self.criterion = criterion
        self.X_train = X_train
        self.y_train = y_train

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.empty((len(X), 2))
        for i, x in enumerate(X):
            set_params(self.model, x)
            with torch.no_grad():
                logits = self.model(self.X_train)
                F[i, 0] = self.criterion(logits, self.y_train).item()
            F[i, 1] = float(np.mean(x**2))
        out["F"] = F


class _GenCallback(Callback):
    """Log metrics each generation; select the best trade-off via weight_decay."""

    def __init__(
        self,
        model,
        criterion,
        X_train,
        y_train,
        X_val,
        y_val,
        weight_decay,
        logger,
        pbar,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.weight_decay = weight_decay
        self.logger = logger
        self.pbar = pbar
        self.gen = 0

    def notify(self, algorithm):
        self.gen += 1
        F = algorithm.pop.get("F")
        X_pop = algorithm.pop.get("X")

        # Pick the solution that best mirrors the original regularised loss
        best_idx = int(np.argmin(F[:, 0] + self.weight_decay * F[:, 1]))
        set_params(self.model, X_pop[best_idx])

        with torch.no_grad():
            tr_logits = self.model(self.X_train)
            tr_loss = self.criterion(tr_logits, self.y_train)
            tr_acc = (tr_logits.argmax(1) == self.y_train).float().mean() * 100
            vl_logits = self.model(self.X_val)
            vl_loss = self.criterion(vl_logits, self.y_val)
            vl_acc = (vl_logits.argmax(1) == self.y_val).float().mean() * 100

        self.pbar.set_description(
            f"{tr_loss:10.2f}, {tr_acc:>3.0f} | {vl_loss:>8.2f}, {vl_acc:>4.0f}"
            f" | |w|²={F[best_idx, 1]:.3f}"
        )
        self.pbar.update(1)
        if self.logger is not None:
            self.logger.log(
                model=self.model,
                epoch=self.gen,
                train_loss=tr_loss,
                train_acc=tr_acc,
                val_loss=vl_loss,
                val_acc=vl_acc,
                **weight_stats(self.model),
            )


def run(
    model,
    criterion,
    X_train,
    y_train,
    X_val,
    y_val,
    num_epochs,
    weight_decay,
    logger,
    sigma=0.5,
    bound_norm=5.0,
    n_individuals=None,
    seed=2,
):
    ndim = sum(p.numel() for p in model.parameters())
    pop_size = n_individuals if n_individuals is not None else 4 + int(3 * np.log(ndim))
    x0 = get_params(model)

    problem = _BiObjectiveProblem(model, criterion, X_train, y_train, ndim, bound_norm)
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=_GaussianSampling(x0, sigma),
    )

    pbar = tqdm.tqdm(total=num_epochs, leave=True, position=0)
    callback = _GenCallback(
        model,
        criterion,
        X_train,
        y_train,
        X_val,
        y_val,
        weight_decay,
        logger,
        pbar,
    )

    print(f"ndim={ndim}, pop_size={pop_size}, sigma={sigma}, bound_norm={bound_norm}")

    result = minimize(
        problem,
        algorithm,
        ("n_gen", num_epochs),
        callback=callback,
        seed=seed,
        verbose=False,
    )
    pbar.close()

    # Select from Pareto front using the same weight_decay trade-off
    F, X_res = result.F, result.X
    best_idx = int(np.argmin(F[:, 0] + weight_decay * F[:, 1]))
    set_params(model, X_res[best_idx])

    print(f"Pareto front size: {len(F)}")
    print(
        f"Selected: train_loss={F[best_idx, 0]:.4f}, weight_norm²={F[best_idx, 1]:.4f}"
    )
