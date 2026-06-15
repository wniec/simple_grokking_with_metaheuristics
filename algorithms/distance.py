"""Symmetry-aware comparison of two optima.

Neural-net weight space has symmetries — distinct weight vectors that compute the
identical function — so a raw Euclidean distance between two optima is misleading
(a relabeled copy of the same solution looks far away). Two complementary
symmetry-aware distances are provided:

* ``functional_distance`` — compares what the networks *compute* over all P²
  input pairs: RMSE between softmax probability vectors plus the fraction of
  inputs whose argmax prediction disagrees. Invariant to *every* weight-space
  reparametrization (neuron permutation, per-feature rescaling / GL on the
  embedding space, the fft model's cyclic structure). ``0`` ⟺ same function.
  This is the complete notion of "is this the same optimum?".

* ``aligned_weight_distance`` — Git Re-Basin style (Ainsworth et al., 2022). The
  per-layer neuron permutation that best matches the current weights to the
  reference is found by Hungarian assignment, alternating over the coupled
  permutations (coordinate descent); it is applied and the residual Euclidean
  distance is returned, alongside the raw (unaligned) distance. This only
  quotients out *permutations*:
    - ``mlp`` has hidden-unit and embedding-feature permutation symmetry,
    - ``cnn`` has conv in/out-channel permutation symmetry,
    - ``fft`` has NO neuron-permutation symmetry (FFT is not permutation
      equivariant), so its aligned distance equals the raw distance — for the
      fft model only ``functional_distance`` is meaningful.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_optimum(path):
    """Load an optimum stored as {param_name: flat list of values}."""
    with open(path) as f:
        return json.load(f)


def load_into(model, optimum):
    """Copy a stored optimum into ``model`` (in place); returns the model."""
    model_keys = {name for name, _ in model.named_parameters()}
    if model_keys != set(optimum):
        raise ValueError(
            f"optimum parameters {sorted(optimum)} do not match the model's "
            f"{sorted(model_keys)} — same --arch/--hidden-dim/P required."
        )
    with torch.no_grad():
        for name, p in model.named_parameters():
            vals = torch.tensor(optimum[name], dtype=p.dtype)
            if vals.numel() != p.numel():
                raise ValueError(
                    f"{name}: optimum has {vals.numel()} values but the model "
                    f"expects {p.numel()} — mismatched --hidden-dim/P."
                )
            p.copy_(vals.reshape(p.shape))
    return model


def _state(model):
    return {
        name: p.detach().cpu().numpy().copy() for name, p in model.named_parameters()
    }


# --------------------------------------------------------------------------- #
# Functional distance
# --------------------------------------------------------------------------- #
def _probs_and_pred(model, X_all):
    with torch.no_grad():
        logits = model(X_all)
        probs = F.softmax(logits, dim=1)
    return probs, logits.argmax(1)


def functional_distance(model, X_all, ref_probs, ref_pred):
    """RMSE of softmax probabilities and argmax-disagreement rate over X_all."""
    probs, pred = _probs_and_pred(model, X_all)
    prob_rmse = torch.sqrt(torch.mean((probs - ref_probs) ** 2)).item()
    disagree = (pred != ref_pred).float().mean().item()
    return prob_rmse, disagree


# --------------------------------------------------------------------------- #
# Permutation-aligned weight distance (Git Re-Basin weight matching)
# --------------------------------------------------------------------------- #
def _match(cost):
    """Hungarian assignment maximizing similarity; returns perm with
    perm[i] = current unit assigned to reference slot i."""
    rows, cols = linear_sum_assignment(-cost)
    perm = np.empty(cost.shape[0], dtype=int)
    perm[rows] = cols
    return perm


def _euclid(a, b, names):
    return float(np.sqrt(sum(np.sum((a[n] - b[n]) ** 2) for n in names)))


def _apply_piE_cols(W, piE, h):
    """Permute the two h-blocks of fc1's (h, 2h) input columns by the same piE."""
    out = W.copy()
    out[:, :h] = W[:, piE]
    out[:, h:] = W[:, h + piE]
    return out


def _align_mlp(cur, ref, iters=20):
    h = ref["fc1.bias"].shape[0]
    piH = np.arange(h)
    piE = np.arange(h)
    for _ in range(iters):
        prev = (piH.copy(), piE.copy())

        # hidden units: fc1 rows + bias + readout columns (cols reindexed by piE)
        cur_fc1_e = _apply_piE_cols(cur["fc1.weight"], piE, h)
        cost = ref["fc1.weight"] @ cur_fc1_e.T
        cost += np.outer(ref["fc1.bias"], cur["fc1.bias"])
        cost += ref["readout.weight"].T @ cur["readout.weight"]
        piH = _match(cost)

        # embedding features: embedding cols + fc1 input blocks (rows reindexed by piH)
        cur_fc1_h = cur["fc1.weight"][piH, :]
        cost = ref["embedding.weight"].T @ cur["embedding.weight"]
        cost += ref["fc1.weight"][:, :h].T @ cur_fc1_h[:, :h]
        cost += ref["fc1.weight"][:, h:].T @ cur_fc1_h[:, h:]
        piE = _match(cost)

        if np.array_equal(piH, prev[0]) and np.array_equal(piE, prev[1]):
            break
    return {"H": piH, "E": piE}


def _apply_mlp(cur, perms):
    piH, piE = perms["H"], perms["E"]
    h = len(piH)
    fc1 = _apply_piE_cols(cur["fc1.weight"], piE, h)[piH, :]
    return {
        "embedding.weight": cur["embedding.weight"][:, piE],
        "fc1.weight": fc1,
        "fc1.bias": cur["fc1.bias"][piH],
        "readout.weight": cur["readout.weight"][:, piH],
        "readout.bias": cur["readout.bias"],
    }


def _align_cnn(cur, ref, iters=20):
    hout = ref["conv.bias"].shape[0]
    hin = ref["embedding.weight"].shape[1]
    pin = np.arange(hin)
    pout = np.arange(hout)
    cw_ref = ref["conv.weight"]  # (hout, hin, k)
    for _ in range(iters):
        prev = (pin.copy(), pout.copy())

        # out channels: conv rows + bias + readout columns (in-channels by pin)
        cw_cur = cur["conv.weight"][:, pin, :]
        cost = cw_ref.reshape(hout, -1) @ cw_cur.reshape(hout, -1).T
        cost += np.outer(ref["conv.bias"], cur["conv.bias"])
        cost += ref["readout.weight"].T @ cur["readout.weight"]
        pout = _match(cost)

        # in channels: embedding cols + conv in-channels (out-channels by pout)
        cw_cur2 = cur["conv.weight"][pout, :, :]
        cost = ref["embedding.weight"].T @ cur["embedding.weight"]
        cost += (
            cw_ref.transpose(1, 0, 2).reshape(hin, -1)
            @ cw_cur2.transpose(1, 0, 2).reshape(hin, -1).T
        )
        pin = _match(cost)

        if np.array_equal(pin, prev[0]) and np.array_equal(pout, prev[1]):
            break
    return {"in": pin, "out": pout}


def _apply_cnn(cur, perms):
    pin, pout = perms["in"], perms["out"]
    return {
        "embedding.weight": cur["embedding.weight"][:, pin],
        "conv.weight": cur["conv.weight"][pout, :, :][:, pin, :],
        "conv.bias": cur["conv.bias"][pout],
        "readout.weight": cur["readout.weight"][:, pout],
        "readout.bias": cur["readout.bias"],
    }


def aligned_weight_distance(model, ref_state):
    """Returns (raw_euclidean, permutation_aligned_euclidean)."""
    cur = _state(model)
    names = list(cur.keys())
    raw = _euclid(cur, ref_state, names)

    if "fc1.weight" in cur:
        aligned_state = _apply_mlp(cur, _align_mlp(cur, ref_state))
    elif "conv.weight" in cur:
        aligned_state = _apply_cnn(cur, _align_cnn(cur, ref_state))
    else:  # fft: no neuron-permutation symmetry
        aligned_state = cur

    aligned = _euclid(aligned_state, ref_state, names)
    return raw, aligned


# --------------------------------------------------------------------------- #
# Reference holder for per-generation tracking
# --------------------------------------------------------------------------- #
class Reference:
    """Precomputed reference optimum used to measure distances during a run."""

    def __init__(self, build_model, optimum, X_all):
        self.X_all = X_all
        ref_model = load_into(build_model(), optimum)
        self.ref_state = _state(ref_model)
        self.ref_probs, self.ref_pred = _probs_and_pred(ref_model, X_all)

    def distances(self, model):
        prob_rmse, disagree = functional_distance(
            model, self.X_all, self.ref_probs, self.ref_pred
        )
        raw, aligned = aligned_weight_distance(model, self.ref_state)
        return dict(
            func_prob_rmse=prob_rmse,
            func_disagree=disagree,
            weight_dist_raw=raw,
            weight_dist_aligned=aligned,
        )


_REF = None


def set_reference(reference):
    """Install a Reference so that `track()` reports distances (opt-in)."""
    global _REF
    _REF = reference


def track(model):
    """Distances of `model` to the installed reference, or {} if none set."""
    return {} if _REF is None else _REF.distances(model)


def active():
    """True if a reference is installed (so `track()` will report distances)."""
    return _REF is not None
