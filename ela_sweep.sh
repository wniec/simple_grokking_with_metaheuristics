#!/usr/bin/env bash
set -euo pipefail

# Log ELA landscape features for the CMA-ES problem across several moduli P,
# in both regimes. Two sampling modes:
#   lhs        (default) -- Latin-hypercube sample of the whole search box
#                           (global landscape); writes images/ela_<arch>_P<P>_<regime>.json
#   trajectory           -- the points CMA-ES actually evaluates while optimizing
#                           (landscape it traverses); writes images/ela_traj_<arch>_P<P>_<regime>.json
#
# Usage:  bash ela_sweep.sh [lhs|trajectory]
# Visualize afterwards:
#   python plot_ela_sweep.py            # lhs sweep
#   python plot_ela_sweep.py ela_traj   # trajectory sweep

MODE="${1:-lhs}"
ARCH=fft
HIDDEN_DIM=4
ELA_SAMPLES=1500   # LHS: sample size; trajectory: cap on points used for features
EPOCHS=1000        # CMA-ES generations (trajectory mode only)
P_VALUES="2 3 5 7 11 13"

for p in $P_VALUES; do
  for regime in grok comprehension; do
    grok_flag=""
    [ "$regime" = "grok" ] && grok_flag="--grok"
    echo ">>> ELA[$MODE]: P=$p regime=$regime"
    if [ "$MODE" = "trajectory" ]; then
      uv run train.py --ela-trajectory --algo cmaes --arch "$ARCH" --hidden-dim "$HIDDEN_DIM" \
        --P "$p" --epochs "$EPOCHS" --ela-samples "$ELA_SAMPLES" $grok_flag
    else
      uv run train.py --ela --algo cmaes --arch "$ARCH" --hidden-dim "$HIDDEN_DIM" \
        --P "$p" --ela-samples "$ELA_SAMPLES" $grok_flag
    fi
  done
done

prefix="ela"; [ "$MODE" = "trajectory" ] && prefix="ela_traj"
echo "Done. Visualize with: python plot_ela_sweep.py $([ "$MODE" = "trajectory" ] && echo ela_traj)"