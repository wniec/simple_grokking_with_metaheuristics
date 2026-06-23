import torch
import numpy as np
import tqdm

from algorithms import distance


# Optional capture of every evaluated (x, fitness) — the optimizer's search
# trajectory — used for trajectory-based ELA. None = not recording.
_TRAJECTORY = None


def enable_trajectory():
    """Start recording evaluated points (opt-in, before running an optimizer)."""
    global _TRAJECTORY
    _TRAJECTORY = []


def get_trajectory():
    """Return the recorded trajectory as (X, y) arrays, or None if empty."""
    if not _TRAJECTORY:
        return None
    X = np.array([t[0] for t in _TRAJECTORY])
    y = np.array([t[1] for t in _TRAJECTORY])
    return X, y


# Optional per-generation ELA over a moving window of recently evaluated points.
# None = disabled; otherwise dict(n_last_points, seed, every).
_ELA_WINDOW = None


def enable_ela_window(n_last_points, seed=2, every=1):
    """Compute ELA features every `every` generations over a moving window of
    the last `n_last_points` evaluated points, tracking how each feature evolves
    during the search. Implies trajectory recording (the window reads from it)."""
    global _ELA_WINDOW
    enable_trajectory()
    _ELA_WINDOW = dict(n_last_points=int(n_last_points), seed=int(seed), every=int(every))


def _ela_window_features():
    """ELA features over the last `n_last_points` evaluated points, or {} if the
    window is too small / the feature is disabled."""
    if _ELA_WINDOW is None or not _TRAJECTORY:
        return {}
    window = _TRAJECTORY[-_ELA_WINDOW["n_last_points"] :]
    if len(window) < 4:  # need a handful of points for any feature to be meaningful
        return {}
    X = np.array([t[0] for t in window])
    y = np.array([t[1] for t in window])
    from algorithms import ela  # local import avoids a circular import at load time

    return ela.window_features(X, y, seed=_ELA_WINDOW["seed"])


def get_params(model):
    return np.concatenate([p.detach().numpy().ravel() for p in model.parameters()])


def weight_stats(model):
    w = get_params(model)
    return dict(
        weight_max=float(np.max(np.abs(w))),
        weight_mean=float(np.mean(np.abs(w))),
        weight_median=float(np.median(np.abs(w))),
        weight_norm=float(np.linalg.norm(w)),
    )


def set_params(model, x):
    x_t = torch.from_numpy(x.astype(np.float32))
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(x_t[idx : idx + n].reshape(p.shape))
            idx += n


def make_fitness(
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
):
    eval_count = [0]
    best_loss = [float("inf")]
    best_x = [get_params(model)]
    pbar = tqdm.tqdm(total=num_epochs, leave=True, position=0)

    def fitness(x):
        set_params(model, x)
        with torch.no_grad():
            logits = model(X_train)
            loss = criterion(logits, y_train).item() + np.mean(x**2) * weight_decay

        if _TRAJECTORY is not None:
            _TRAJECTORY.append((x.copy(), loss))

        if loss < best_loss[0]:
            best_loss[0] = loss
            best_x[0] = x.copy()

        eval_count[0] += 1
        if eval_count[0] % pop_size == 0:
            gen = eval_count[0] // pop_size
            set_params(model, best_x[0])
            with torch.no_grad():
                tr_logits = model(X_train)
                tr_loss = criterion(tr_logits, y_train)
                tr_acc = (tr_logits.argmax(1) == y_train).float().mean() * 100
                vl_logits = model(X_val)
                vl_loss = criterion(vl_logits, y_val)
                vl_acc = (vl_logits.argmax(1) == y_val).float().mean() * 100
            dist = distance.track(model)
            desc = f"{tr_loss:10.2f}, {tr_acc:>3.0f} | {vl_loss:>8.2f}, {vl_acc:>4.0f}"
            if dist:
                desc += f" | wΔ={dist['weight_dist_aligned']:.2f} fΔ={dist['func_prob_rmse']:.3f}"
            pbar.set_description(desc)
            pbar.update(1)
            if logger is not None:
                logger.log(
                    model=model,
                    epoch=gen,
                    train_loss=tr_loss,
                    train_acc=tr_acc,
                    val_loss=vl_loss,
                    val_acc=vl_acc,
                    **dist,
                    **weight_stats(model),
                )
                if _ELA_WINDOW is not None and gen % _ELA_WINDOW["every"] == 0:
                    ela_feats = _ela_window_features()
                    if ela_feats:
                        logger.log_to("ela_window.jsonl", epoch=gen, **ela_feats)

        return loss

    return fitness, pbar, best_x
