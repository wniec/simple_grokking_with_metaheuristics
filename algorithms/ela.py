"""Exploratory Landscape Analysis (ELA) features for the optimized problem.

Characterizes the fitness landscape of the *optimized problem* — the same
regularized objective the evolutionary optimizers minimize:

    fitness(x) = train_loss(model with weights x) + weight_decay * mean(x**2)

A space-filling (Latin-hypercube) sample is drawn over the parameter space
(bounded by ±bound_norm, matching the evolutionary search space), evaluated, and
summarized by the standard ELA feature groups.

This is a self-contained numpy/scipy reimplementation of the classical feature
sets (Mersmann et al. 2011 for meta/distribution; Lunacek & Whitley 2006 for
dispersion; Kerschke et al. 2015 for nearest-better-clustering; Muñoz et al.
2015 for information content). It deliberately avoids `pflacco`, which pins
numpy~=1.24 and cannot coexist with this project's numpy 2.x / torch stack. The
definitions follow the literature but may differ from pflacco at the margins.
"""

import json
import os

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde, kurtosis, qmc, skew

from algorithms._common import set_params
from image_paths import image_path


# --------------------------------------------------------------------------- #
# Feature groups
# --------------------------------------------------------------------------- #
def _adj_r2(y, y_hat, n_coef):
    n = len(y)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    denom = n - n_coef - 1
    return float(1.0 - (1.0 - r2) * (n - 1) / denom) if denom > 0 else float(r2)


def _ela_meta(X, y):
    """Linear / quadratic surrogate-model fit quality and coefficients."""
    n = len(y)
    lin = np.hstack([np.ones((n, 1)), X])
    c_lin, *_ = np.linalg.lstsq(lin, y, rcond=None)
    lin_adj_r2 = _adj_r2(y, lin @ c_lin, X.shape[1])
    abscoef = np.abs(c_lin[1:])

    quad = np.hstack([np.ones((n, 1)), X, X**2])
    c_quad, *_ = np.linalg.lstsq(quad, y, rcond=None)
    quad_adj_r2 = _adj_r2(y, quad @ c_quad, 2 * X.shape[1])

    return {
        "ela_meta.lin_simple.adj_r2": lin_adj_r2,
        "ela_meta.lin_simple.intercept": float(c_lin[0]),
        "ela_meta.lin_simple.coef_min": float(abscoef.min()),
        "ela_meta.lin_simple.coef_max": float(abscoef.max()),
        "ela_meta.lin_simple.coef_max_by_min": float(
            abscoef.max() / max(abscoef.min(), 1e-12)
        ),
        "ela_meta.quad_simple.adj_r2": quad_adj_r2,
    }


def _ela_distr(y):
    """Shape of the objective-value distribution."""
    n_peaks = 1
    if np.ptp(y) > 0:
        kde = gaussian_kde(y)
        grid = np.linspace(y.min(), y.max(), 512)
        dens = kde(grid)
        n_peaks = int(np.sum((dens[1:-1] > dens[:-2]) & (dens[1:-1] > dens[2:])))
    return {
        "ela_distr.skewness": float(skew(y)),
        "ela_distr.kurtosis": float(kurtosis(y)),  # excess kurtosis
        "ela_distr.number_of_peaks": max(n_peaks, 1),
    }


def _dispersion(X, y, dists, quantiles=(0.02, 0.05, 0.10, 0.25)):
    """Dispersion of the best quantiles vs. the whole sample (Lunacek & Whitley)."""
    n = len(y)
    all_mean, all_median = float(dists.mean()), float(np.median(dists))
    order = np.argsort(y)
    out = {}
    for q in quantiles:
        k = max(2, int(np.ceil(q * n)))
        d = pdist(X[order[:k]])
        tag = f"{int(q * 100):02d}"
        out[f"disp.ratio_mean_{tag}"] = float(d.mean() / all_mean)
        out[f"disp.ratio_median_{tag}"] = float(np.median(d) / all_median)
        out[f"disp.diff_mean_{tag}"] = float(d.mean() - all_mean)
    return out


def _nbc(X, y, D):
    """Nearest-better-clustering features (Kerschke et al. 2015)."""
    n = len(y)
    Dinf = D.copy()
    np.fill_diagonal(Dinf, np.inf)
    nn_dist = Dinf.min(axis=1)
    nb_dist = np.full(n, np.nan)
    for i in range(n):
        better = y < y[i]
        if better.any():
            nb_dist[i] = Dinf[i, better].min()
    valid = ~np.isnan(nb_dist)
    ratio = nb_dist[valid] / nn_dist[valid]
    return {
        "nbc.nn_nb.mean_ratio": float(np.mean(ratio)) if valid.any() else float("nan"),
        "nbc.nn_nb.sd_ratio": float(np.std(ratio)) if valid.any() else float("nan"),
        "nbc.nb_fitness.cor": (
            float(np.corrcoef(nb_dist[valid], y[valid])[0, 1])
            if valid.sum() > 1
            else float("nan")
        ),
    }


def _entropy(symbols):
    """Shannon entropy (base 6) over consecutive non-equal symbol pairs."""
    pairs = [(a, b) for a, b in zip(symbols[:-1], symbols[1:]) if a != b]
    total = len(pairs)
    if total == 0:
        return 0.0
    counts = {}
    for p in pairs:
        counts[p] = counts.get(p, 0) + 1
    h = 0.0
    for c in counts.values():
        prob = c / total
        h -= prob * np.log(prob) / np.log(6)
    return float(h)


def _partial_info(symbols):
    """Partial information: fraction of slope-sign changes (Muñoz et al. 2015)."""
    s = symbols[symbols != 0]
    if len(s) < 2:
        return 0.0
    changes = np.sum(s[1:] != s[:-1])
    return float(changes / (len(symbols) - 1))


def _information_content(X, y, seed):
    """Information content of the landscape over a nearest-neighbour walk."""
    n = len(y)
    rng = np.random.default_rng(seed)
    # Greedy nearest-neighbour tour to order the sample into a space-filling walk.
    order = [int(rng.integers(n))]
    remaining = set(range(n)) - {order[0]}
    while remaining:
        rem = np.fromiter(remaining, dtype=int)
        nxt = int(rem[np.argmin(np.linalg.norm(X[rem] - X[order[-1]], axis=1))])
        order.append(nxt)
        remaining.discard(nxt)
    order = np.array(order)
    dy = np.diff(y[order])
    dx = np.linalg.norm(np.diff(X[order], axis=0), axis=1)
    dx[dx == 0] = 1e-12
    ratio = dy / dx

    rmax = float(np.max(np.abs(ratio)))
    if rmax <= 0:
        return {
            "ic.h_max": 0.0,
            "ic.eps_s": float("nan"),
            "ic.eps_max": float("nan"),
            "ic.m0": 0.0,
        }
    eps_grid = np.concatenate(
        [[0.0], np.logspace(np.log10(rmax) - 5, np.log10(rmax), 200)]
    )
    H = np.array(
        [
            _entropy(np.sign(np.where(np.abs(ratio) <= e, 0.0, ratio)).astype(int))
            for e in eps_grid
        ]
    )
    M = np.array(
        [
            _partial_info(np.sign(np.where(np.abs(ratio) <= e, 0.0, ratio)).astype(int))
            for e in eps_grid
        ]
    )

    below = np.where(H < 0.05)[0]
    eps_s = (
        float(np.log10(eps_grid[below[0]]))
        if below.size and eps_grid[below[0]] > 0
        else float("nan")
    )
    eps_at_max = eps_grid[int(np.argmax(H))]
    return {
        "ic.h_max": float(H.max()),
        "ic.eps_s": eps_s,
        "ic.eps_max": float(np.log10(eps_at_max)) if eps_at_max > 0 else float("nan"),
        "ic.m0": float(M[0]),
    }


# --------------------------------------------------------------------------- #
# Shared computation / IO
# --------------------------------------------------------------------------- #
def _all_features(X, y, seed):
    """Compute every ELA feature group on a (X, y) sample."""
    dists = pdist(X)
    D = squareform(dists)
    features = {}
    features.update(_ela_meta(X, y))
    features.update(_ela_distr(y))
    features.update(_dispersion(X, y, dists))
    features.update(_nbc(X, y, D))
    features.update(_information_content(X, y, seed))
    return features


def _save(features, run_name, prefix, meta):
    print(f"\n{prefix} features:")
    for k, v in features.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    out_path = image_path(f"{prefix}_{run_name}.json")
    with open(out_path, "w") as f:
        json.dump({**meta, **features}, f, indent=2)
    print(f"\nSaved {out_path}")


def features_from_sample(
    X, y, run_name, seed=2, p=None, prefix="ela_traj", max_points=2000
):
    """Compute ELA features from a precomputed sample (e.g. an optimizer's
    search trajectory). Subsamples to `max_points` to bound the O(n²) feature
    computations."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_total = len(y)
    if n_total > max_points:
        idx = np.random.default_rng(seed).choice(n_total, max_points, replace=False)
        X, y = X[idx], y[idx]
    print(
        f"Computing {prefix} features: {len(y)} of {n_total} trajectory points, "
        f"ndim={X.shape[1]}"
    )
    features = _all_features(X, y, seed)
    _save(
        features,
        run_name,
        prefix,
        {
            "P": p,
            "ndim": int(X.shape[1]),
            "n_points": int(len(y)),
            "n_trajectory": int(n_total),
        },
    )
    return features


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(
    model,
    criterion,
    X_train,
    y_train,
    weight_decay,
    run_name,
    bound_norm=5.0,
    n_samples=None,
    seed=2,
    p=None,
):
    ndim = sum(p.numel() for p in model.parameters())
    if n_samples is None:
        n_samples = max(50 * ndim, 200)

    def fitness(x):
        set_params(model, np.asarray(x))
        with torch.no_grad():
            loss = criterion(model(X_train), y_train).item()
        return loss + float(np.mean(np.asarray(x) ** 2)) * weight_decay

    print(
        f"Sampling landscape: ndim={ndim}, n_samples={n_samples}, bounds=±{bound_norm}"
    )
    sampler = qmc.LatinHypercube(d=ndim, seed=seed)
    X = sampler.random(n_samples) * (2 * bound_norm) - bound_norm
    y = np.array([fitness(row) for row in X])

    features = _all_features(X, y, seed)
    _save(
        features,
        run_name,
        "ela",
        {"P": p, "ndim": ndim, "n_samples": n_samples, "bound_norm": bound_norm},
    )
    return features
