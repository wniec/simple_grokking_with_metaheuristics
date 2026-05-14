import numpy as np
from pypop7.optimizers.pso.cpso import CPSO
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
    cognition=1.49,
    society=1.49,
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

    # CPSO samples the initial swarm from the initial bounds (ignores options['x']).
    # Centering around model weights keeps ||x|| small so the L2 penalty stays
    # comparable to the task loss throughout optimization.
    x0 = get_params(model)
    bound = np.full(ndim, bound_norm)
    print(
        f"ndim={ndim}, pop_size={pop_size}, cognition={cognition}, society={society}, sigma={sigma}, bound_norm={bound_norm}"
    )
    problem = {
        "fitness_function": fitness,
        "ndim_problem": ndim,
        "lower_boundary": -bound,
        "upper_boundary": bound,
        "initial_lower_boundary": np.clip(x0 - sigma, -bound_norm, bound_norm),
        "initial_upper_boundary": np.clip(x0 + sigma, -bound_norm, bound_norm),
    }
    options = {
        "max_function_evaluations": num_epochs * pop_size,
        "x": x0,
        "n_individuals": pop_size,
        "cognition": cognition,
        "society": society,
        "seed_rng": seed,
        "verbose": False,
        "verbose_frequency": int(1e9),
    }
    results = CPSO(problem, options).optimize()
    pbar.close()

    set_params(model, results["best_so_far_x"])
    print(f"Best train loss: {results['best_so_far_y']:.4f}")
