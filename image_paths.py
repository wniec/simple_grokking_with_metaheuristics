"""Map an image filename to a tidy location under images/.

Files are grouped by category (the analysis that produced them) and, where
applicable, by the modulus P of the run:

    images/<category>/P<P>/<filename>      e.g. images/metrics/P5/metrics_cmaes_P5_grokking.jpg
    images/sweeps/<filename>               cross-P aggregate plots (no single P)

Categories are inferred from the filename prefix (metrics_, lon_/lon3d_, ela_,
ela_traj_, ela_window_, emb_). P is read from the `_P<n>_` token; files predating
P-keying are assumed to be P=3 (the original default).
"""

import os
import re

IMAGES_DIR = "images"


def categorize(filename):
    """Return (category, P) for a filename; P is None for cross-P aggregates."""
    base = os.path.basename(filename)
    # Cross-P aggregate plots (span every P) — keep out of the P folders.
    if base.startswith(("ela_sweep_", "ela_traj_sweep_")):
        return "sweeps", None
    if base.startswith("metrics_"):
        category = "metrics"
    elif base.startswith(("lon_", "lon3d_")):
        category = "lon"
    elif base.startswith("ela_window_"):
        category = "ela_window"
    elif base.startswith("ela_traj_"):
        category = "ela_traj"
    elif base.startswith("ela_"):
        category = "ela"
    elif base.startswith("emb_"):
        category = "emb"
    else:
        category = "misc"
    m = re.search(r"_P(\d+)_", base)
    return category, (m.group(1) if m else "3")


def image_path(filename, images_dir=IMAGES_DIR, make=True):
    """Full destination path for `filename` under the organized images tree."""
    base = os.path.basename(filename)
    category, p = categorize(base)
    folder = (
        os.path.join(images_dir, category)
        if p is None
        else os.path.join(images_dir, category, f"P{p}")
    )
    if make:
        os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, base)
