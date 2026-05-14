import numpy as np
from pypop7.optimizers.es.lmcmaes import LMCMAES
from algorithms._common import get_params, set_params, make_fitness


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

    fitness, pbar, best_x = make_fitness(
        model,
        criterion,
        X_train,
        y_train,
        X_val,
        y_val,
        weight_decay,
        logger,
        pop_size,
        num_epochs,
    )

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
        "x": get_params(model),
        "sigma": sigma,
        "seed_rng": seed,
        "verbose": False,
        "verbose_frequency": int(1e9),
    }
    results = LMCMAES(problem, options).optimize()
    pbar.close()

    set_params(model, results["best_so_far_x"])
    print(f"Best train loss: {results['best_so_far_y']:.4f}")
