import numpy as np
from pypop7.optimizers.ga.g3pcx import G3PCX
from algorithms._common import get_params, set_params, make_fitness


class _SafeG3PCX(G3PCX):
    """G3PCX with zero-norm guards in the PCX recombination step.

    pypop7 forces parents[0] = elitist but doesn't exclude the elitist from
    parents[1:], so diff[ii] = x[elitist] - x[elitist] = 0 can occur (~5% of
    generations).  The original code divides by norm(diff[ii]) * d_norm, giving
    NaN offspring that permanently corrupt the population once they enter it.

    Fixes applied inside iterate():
      - diff_norm == 0: perpendicular distance from that parent is defined as 0
        (the duplicate coincides with the elitist, contributes nothing new).
      - d_norm == 0: mean of parents equals elitist, so there is no preferred
        direction; the projection step is skipped and orth stays random.
    """

    def iterate(self, x=None, y=None, args=None):
        self._elitist, fitness = np.argmin(y), []
        parents = self.rng_optimization.choice(
            self.n_individuals, size=self.n_parents, replace=False
        )
        if self._elitist not in parents:
            parents[0] = self._elitist
        xx = np.empty((self.n_offsprings, self.ndim_problem))
        yy = np.empty((self.n_offsprings,))
        g = np.mean(x[parents], axis=0)
        for i in range(self.n_offsprings):
            if self._check_terminations():
                break
            p = self._elitist
            d = g - x[p]
            d_norm = np.linalg.norm(d)
            diff = x[parents[1:]] - x[p]  # shape (n_parents-1, ndim)
            diff_norms = np.linalg.norm(diff, axis=1)  # shape (n_parents-1,)
            # perpendicular distance for each non-elitist parent
            d_mean_vals = np.zeros(self.n_parents - 1)
            if d_norm > 1e-12:
                for ii in range(self.n_parents - 1):
                    if diff_norms[ii] > 1e-12:
                        cos_a = np.clip(
                            np.dot(diff[ii], d) / (diff_norms[ii] * d_norm), -1.0, 1.0
                        )
                        d_mean_vals[ii] = diff_norms[ii] * np.sqrt(1.0 - cos_a**2)
                    # else: duplicate parent → 0 contribution (already zero)
            d_mean = d_mean_vals.mean()
            orth = (
                self._std_pcx_2
                * d_mean
                * self.rng_optimization.standard_normal(self.ndim_problem)
            )
            if d_norm > 1e-12:
                orth -= (np.dot(orth, d) * d) / (d_norm**2)
                xx[i] = (
                    x[p]
                    + self._std_pcx_1 * self.rng_optimization.standard_normal() * d
                    + orth
                )
            else:
                xx[i] = x[p] + orth
            yy[i] = self._evaluate_fitness(xx[i], args)
            fitness.append(yy[i])
        offsprings = self.rng_optimization.choice(
            self.n_individuals, size=2, replace=False
        )
        xx = np.vstack((xx, x[offsprings]))
        yy = np.hstack((yy, y[offsprings]))
        order = np.argsort(yy)[:2]
        x[offsprings], y[offsprings] = xx[order], yy[order]
        self._n_generations += 1
        return fitness


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
    n_offsprings=2,
    n_parents=3,
    sigma=0.5,
    bound_norm=5.0,
    n_individuals=None,
    seed=2,
):
    ndim = sum(p.numel() for p in model.parameters())
    n_pop = n_individuals if n_individuals is not None else 4 + int(3 * np.log(ndim))
    pop_size = n_pop

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

    bound = np.full(ndim, bound_norm)
    x0 = get_params(model)
    print(
        f"ndim={ndim}, pop_size={n_pop}, n_offsprings={n_offsprings}, n_parents={n_parents}, sigma={sigma}, bound_norm={bound_norm}"
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
        "n_individuals": n_pop,
        "n_offsprings": n_offsprings,
        "n_parents": n_parents,
        "seed_rng": seed,
        "verbose": False,
        "verbose_frequency": int(1e9),
    }
    results = _SafeG3PCX(problem, options).optimize()
    pbar.close()

    set_params(model, results["best_so_far_x"])
    print(f"Best train loss: {results['best_so_far_y']:.4f}")
