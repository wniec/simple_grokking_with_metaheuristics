#!/usr/bin/env bash
set -euo pipefail

# Sweep the modular-addition task over several moduli P, running every optimizer
# in both regimes. Larger P is a harder problem, so the epoch budget and model
# width scale with P. All the scaling is defined in the three functions below —
# edit them to retune.

ARCH=fft
P_VALUES="3 5 7 11"
ALGOS="gradient cmaes de cpso moea"

# --- epochs = P_C(P) * algo_C(algo) ---------------------------------------- #
# algo_C: base optimization budget per algorithm (gradient counts steps; the
#         population-based optimizers count generations, hence much smaller).
algo_C() {
  case "$1" in
    gradient) echo 30000 ;;
    cmaes)    echo 300 ;;
    de)       echo 600 ;;
    cpso)     echo 1000 ;;
    moea)     echo 600 ;;
    *)        echo 1000 ;;
  esac
}
# P_C: how the budget grows with the modulus (linear by default; try P*P for a
#      steeper schedule, or a constant to disable P-scaling of epochs).
P_C() { echo "$1"; }

# --- hidden_dim(P, algo): model width ------------------------------------- #
# Grows with P (more classes -> more capacity); $2 is the algo if you want to
# give, say, the evolutionary optimizers a smaller search space than gradient.
hidden_dim() {
  p="$1"
  algo="$2"
  echo $(( p < 4 ? 4 : p ))
}
# --------------------------------------------------------------------------- #

for p in $P_VALUES; do
  # A per-P stored optimum enables distance tracking. It must match this run's
  # arch / P / hidden_dim, so it is opt-in: drop an optimum_P<P>.json to use it.
  track=""
  if [ -f "optima/optimum_P${p}.json" ]; then
    track="--track-optimum optima/optimum_P${p}.json"
    echo ">>> P=$p: tracking distance to optima/optimum_P${p}.json"
  else
    echo ">>> P=$p: no optimum_P${p}.json found — distance tracking OFF"
  fi

  for algo in $ALGOS; do
    epochs=$(( $(P_C "$p") * $(algo_C "$algo") ))
    hdim=$(hidden_dim "$p" "$algo")
    echo ">>> P=$p algo=$algo  epochs=$epochs  hidden_dim=$hdim"

    # grokking regime (low weight decay) and comprehension regime (high)
    uv run train.py --arch "$ARCH" --P "$p" --hidden-dim "$hdim" --algo "$algo" \
      --epochs "$epochs" --log $track --grok
    uv run train.py --arch "$ARCH" --P "$p" --hidden-dim "$hdim" --algo "$algo" \
      --epochs "$epochs" --log $track
  done
done

uv run plot.py