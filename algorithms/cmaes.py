import numpy as np
from pypop7.optimizers.es.lmcmaes import LMCMAES
from algorithms._common import get_params, set_params, make_fitness


def _lr_encode(params, n, rank):
    """Project initial params into low-rank factor space via truncated SVD."""
    padded = np.zeros(n * n)
    padded[: len(params)] = params
    U, s, Vt = np.linalg.svd(padded.reshape(n, n), full_matrices=False)
    sq_s = np.sqrt(s[:rank])
    return np.concatenate([(U[:, :rank] * sq_s).ravel(), (Vt[:rank].T * sq_s).ravel()])


def _lr_decode(z, n, rank, ndim):
    """Reconstruct parameter vector from flat (U, V) low-rank factors."""
    U = z[: n * rank].reshape(n, rank)
    V = z[n * rank :].reshape(n, rank)
    return (U @ V.T).ravel()[:ndim]


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
    rank=0,
):
    ndim = sum(p.numel() for p in model.parameters())

    if rank > 0:
        n = int(np.ceil(np.sqrt(ndim)))
        search_dim = 2 * n * rank
        pop_size = n_individuals if n_individuals is not None else 4 + int(3 * np.log(search_dim))

        raw_fitness, pbar, best_x = make_fitness(
            model, criterion, X_train, y_train, X_val, y_val,
            weight_decay, logger, pop_size, num_epochs,
        )

        def fitness(z):
            return raw_fitness(_lr_decode(z, n, rank, ndim))

        z0 = _lr_encode(get_params(model), n, rank)

        print(f"ndim={ndim}, rank={rank}, n={n}, search_dim={search_dim}, pop_size={pop_size}, sigma={sigma}")
        problem = {
            "fitness_function": fitness,
            "ndim_problem": search_dim,
            "lower_boundary": -bound_norm,
            "upper_boundary": bound_norm,
            "initial_lower_boundary": -bound_norm,
            "initial_upper_boundary": bound_norm,
        }
        options = {
            "max_function_evaluations": num_epochs * pop_size,
            "x": z0,
            "sigma": sigma,
            "seed_rng": seed,
            "verbose": False,
            "verbose_frequency": int(1e9),
        }
        results = LMCMAES(problem, options).optimize()
        pbar.close()

        best_params = _lr_decode(results["best_so_far_x"], n, rank, ndim)
        set_params(model, best_params)
        print(f"Best train loss: {results['best_so_far_y']:.4f}")
    else:
        pop_size = n_individuals if n_individuals is not None else 4 + int(3 * np.log(ndim))

        fitness, pbar, best_x = make_fitness(
            model, criterion, X_train, y_train, X_val, y_val,
            weight_decay, logger, pop_size, num_epochs,
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