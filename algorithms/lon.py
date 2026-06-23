"""Fitness-landscape analysis via Local Optima Networks (LONs).

Uses the `lonkit` library to map the structure of the *optimized problem* — the
same regularized objective the evolutionary optimizers minimize:

    fitness(x) = train_loss(model with weights x) + weight_decay * mean(x**2)

Basin-hopping samples local optima of this objective over the model's parameter
space (bounded by ±bound_norm, matching the evolutionary search space) and
connects them into a LON, whose metrics (number of optima, funnels, global
funnels, ...) characterise how rugged / funnelled the landscape is.

LON construction is only practical for low-dimensional models, so this is meant
to be run on a tiny network (e.g. `--arch fft --hidden-dim 4`, 27 parameters).
"""

import os
import json

import numpy as np
import torch

from lonkit import compute_lon, BasinHoppingSamplerConfig, LONVisualizer
from algorithms._common import set_params
from image_paths import image_path


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def run(
    model,
    criterion,
    X_train,
    y_train,
    weight_decay,
    run_name,
    bound_norm=5.0,
    n_runs=100,
    n_iter_no_change=250,
    step_size=0.1,
    seed=2,
):
    ndim = sum(p.numel() for p in model.parameters())

    def fitness(x):
        # Regularized training objective — identical to the evolutionary fitness.
        set_params(model, np.asarray(x))
        with torch.no_grad():
            loss = criterion(model(X_train), y_train).item()
        return loss + float(np.mean(np.asarray(x) ** 2)) * weight_decay

    config = BasinHoppingSamplerConfig(
        n_runs=n_runs,
        n_iter_no_change=n_iter_no_change,
        step_size=step_size,
        seed=seed,
        n_jobs=-1,  # basin-hopping runs are independent — always use all cores
    )

    print(
        f"Building LON: ndim={ndim}, bounds=±{bound_norm}, n_runs={n_runs}, "
        f"n_iter_no_change={n_iter_no_change}, step_size={step_size}"
    )
    if ndim > 50:
        print(
            f"WARNING: ndim={ndim} is high for LON analysis — basin-hopping will be "
            "slow and the network hard to interpret. Consider a smaller model, e.g. "
            "--arch fft --hidden-dim 4 (27 parameters)."
        )

    lon = compute_lon(
        fitness,
        dim=ndim,
        lower_bound=-bound_norm,
        upper_bound=bound_norm,
        config=config,
        verbose=True,
    )

    metrics = lon.compute_metrics()
    cmlon = lon.to_cmlon()

    print(f"\nLON: {lon.n_vertices} optima, {lon.n_edges} edges")
    print("Landscape metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    viz = LONVisualizer()

    plot_2d = image_path(f"lon_{run_name}.png")
    viz.plot_2d(cmlon, output_path=plot_2d, seed=seed)
    print(f"\nSaved {plot_2d}")

    # plot_3d returns a Plotly figure; write it as a self-contained interactive
    # HTML page (avoids the kaleido static-image path, which rejects .html).
    plot_3d = image_path(f"lon3d_{run_name}.html")
    viz.plot_3d(cmlon, seed=seed).write_html(plot_3d)
    print(f"Saved {plot_3d}")

    metrics_path = image_path(f"lon_{run_name}.json")
    with open(metrics_path, "w") as f:
        json.dump({k: _jsonable(v) for k, v in metrics.items()}, f, indent=2)
    print(f"Saved {metrics_path}")
