"""Visualize ELA features across moduli P.

Reads the per-P ELA feature files written by `ela_sweep.sh`
(`images/<prefix>_<arch>_P<P>_<regime>.json`) and plots every numeric feature as
a function of P, overlaying the grokking and comprehension regimes. Output:
`images/<prefix>_sweep_<arch>.png`.

Usage: python plot_ela_sweep.py [prefix]
  prefix defaults to "ela" (LHS sweep); pass "ela_traj" for the CMA-ES
  trajectory sweep.
"""

import glob
import json
import os
import re
import sys

import numpy as np
from matplotlib import pyplot as plt

plt.style.use("mplstyle.mplstyle")

from image_paths import IMAGES_DIR, image_path

ARCH = "fft"
META = {"P", "ndim", "n_samples", "bound_norm", "n_points", "n_trajectory"}
COLORS = {"grokking": "C0", "comprehension": "C1"}


def _load(arch, prefix):
    pat = re.compile(rf"{prefix}_{arch}_P(\d+)_(grokking|comprehension)\.json$")
    records = {}  # regime -> list of feature dicts (sorted by P)
    # Per-P feature files live under images/<prefix>/P<P>/.
    for path in glob.glob(
        os.path.join(IMAGES_DIR, prefix, "P*", f"{prefix}_{arch}_P*_*.json")
    ):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        with open(path) as f:
            d = json.load(f)
        d.setdefault("P", int(m.group(1)))
        records.setdefault(m.group(2), []).append(d)
    for recs in records.values():
        recs.sort(key=lambda r: r["P"])
    return records


def main(arch=ARCH, prefix="ela"):
    records = _load(arch, prefix)
    if not records:
        print(
            f"No {prefix}_{arch}_P*_*.json files in {IMAGES_DIR}/. Run ela_sweep.sh first."
        )
        return

    feats = sorted(
        {
            k
            for recs in records.values()
            for r in recs
            for k, v in r.items()
            if k not in META and isinstance(v, (int, float))
        }
    )

    ncol = 4
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), dpi=150)
    axes = np.array(axes).reshape(-1)

    for ax, feat in zip(axes, feats):
        for regime, recs in records.items():
            pts = [
                (r["P"], r[feat]) for r in recs if isinstance(r.get(feat), (int, float))
            ]
            if pts:
                ps, ys = zip(*pts)
                ax.plot(ps, ys, "o-", ms=3, label=regime, color=COLORS.get(regime))
        ax.set_title(feat, fontsize=6)
        ax.set_xlabel("P", fontsize=6)
        ax.tick_params(labelsize=5)

    for ax in axes[len(feats) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    src = "CMA-ES trajectory" if prefix == "ela_traj" else "LHS"
    fig.suptitle(f"ELA features vs P  ({arch}, {src})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = image_path(f"{prefix}_sweep_{arch}.png")
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main(prefix=sys.argv[1] if len(sys.argv) > 1 else "ela")
