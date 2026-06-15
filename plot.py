import json
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch
import os
from train import MLPModel

model = MLPModel()
import tqdm
from sys import argv

plt.style.use("mplstyle.mplstyle")

from image_paths import image_path


def _get_metrics(log_dir, skip=1):
    """Load metrics for a run, or None if the run logged nothing usable.

    Prefers a non-empty metrics.jsonl; falls back to a legacy metrics.csv.
    """
    jsonl_path = os.path.join(log_dir, "metrics.jsonl")
    csv_path = os.path.join(log_dir, "metrics.csv")

    rows = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            rows = [json.loads(line) for line in f if line.strip()][::skip]

    if rows:
        keys = set().union(*(r.keys() for r in rows))
        data = {k: np.array([r.get(k, float("nan")) for r in rows]) for k in keys}
    elif os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        metrics = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        cols = [
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "weight_max",
            "weight_mean",
            "weight_median",
            "weight_norm",
        ]
        data = {
            c: (
                metrics[::skip, i]
                if metrics.shape[1] > i
                else np.full(len(metrics[::skip, 0]), float("nan"))
            )
            for i, c in enumerate(cols)
        }
    else:
        return None

    if "epoch" not in data or len(data["epoch"]) == 0:
        return None
    data["epoch"] = data["epoch"].astype(int)
    return data


def _load_embeddings(log_dir, epoch=None):
    model.load_state_dict(torch.load(os.path.join(log_dir, "model.pt")))
    return model.embedding.weight.detach().numpy()


def animate_embedddings(log_dir):
    # Load Data
    print(f"Loading {log_dir}...")
    m = _get_metrics(log_dir, skip=1)
    if m is None:
        print("  (no metrics logged, skipping)\n")
        return
    epochs = m["epoch"]
    nan = np.full(len(epochs), float("nan"))
    train_loss = m.get("train_loss", nan)
    train_acc = m.get("train_acc", nan)
    val_loss = m.get("val_loss", nan)
    val_acc = m.get("val_acc", nan)

    # PCA
    all_embeddings = []
    pca = PCA(n_components=2)
    # Load the last model to get the final PC's
    embeddings = _load_embeddings(log_dir, epochs[-1])
    pca.fit(embeddings)
    for epoch in epochs:
        embeddings = _load_embeddings(log_dir, epoch)
        all_embeddings.append(pca.transform(embeddings))
    all_embeddings = np.array(all_embeddings)
    orig_shape = all_embeddings.shape
    all_embeddings = all_embeddings.reshape(-1, all_embeddings.shape[-1])
    min_ = np.min(all_embeddings, axis=0)
    max_ = np.max(all_embeddings, axis=0)
    all_embeddings = (all_embeddings - min_) / (max_ - min_) * 2 - 1
    all_embeddings = all_embeddings.reshape(orig_shape)
    print(f"Loaded {len(all_embeddings)} embeddings")

    # Plot
    fig, ax = plt.subplots(dpi=300)
    # ax.set_xlim(min(all_embeddings[0][:, 0]), max(all_embeddings[0][:, 0]))
    # ax.set_ylim(min(all_embeddings[0][:, 1]), max(all_embeddings[0][:, 1]))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_title("Embeddings")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    P = embeddings.shape[0]
    colors = plt.cm.viridis(np.linspace(0, 1, P))

    sc = ax.scatter([0] * P, [0] * P, c=colors, s=2)
    # annotate each embedding with its index
    an = [ax.annotate(i, (0, 0), color=colors[i], fontsize=6) for i in range(P)]
    metrics = ax.text(0.0, 0.975, "", transform=ax.transAxes, fontsize=6)

    pbar = tqdm.trange(len(epochs), desc="Plotting...", leave=False)

    def update(i):
        pbar.update()
        sc.set_offsets(all_embeddings[i])
        for j, txt in enumerate(an):
            txt.set_position(all_embeddings[i][j])
        msg = (
            f"Epoch: {epochs[i]} Train Loss: {train_loss[i]:.2f} "
            + f"Acc: {train_acc[i]:.0f} | "
            + f"Val Loss: {val_loss[i]:.2f} "
            + f"Acc: {val_acc[i]:.0f} "
        )
        metrics.set_text(msg)
        return sc, *an, metrics

    anim = FuncAnimation(fig, update, frames=len(epochs), blit=True)
    name = os.path.basename(log_dir)
    savefile = image_path(f"emb_{name}.mp4")
    anim.save(savefile, writer="ffmpeg", fps=len(epochs) // 10)
    print(f"Saved {savefile}\n")


def plot_metrics(log_dir):
    print(f"Loading {log_dir}...")
    m = _get_metrics(log_dir)
    if m is None:
        print("  (no metrics logged, skipping)\n")
        return
    epochs = m["epoch"]
    train_loss = m["train_loss"]
    val_loss = m["val_loss"]

    def present(key):
        return key in m and np.any(np.isfinite(m[key]))

    # The accuracy panel is replaced by tracked optimum-distance series
    # (logged only when training was run with --track-optimum).
    tracked_weight = [
        k for k in ("weight_dist_raw", "weight_dist_aligned") if present(k)
    ]
    tracked_func = [k for k in ("func_prob_rmse", "func_disagree") if present(k)]
    has_tracked = bool(tracked_weight or tracked_func)

    nrows = 3 if has_tracked else 2
    fig, ax = plt.subplots(nrows, 1, sharex=True, dpi=200)

    ax[0].plot(epochs, train_loss, label="Train")
    ax[0].plot(epochs, val_loss, label="Val")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].set_yscale("log")
    ax[0].set_ylim(train_loss.min() * 0.9, val_loss.max() * 1.0)

    if has_tracked:
        dax = ax[1]
        wlabels = {"weight_dist_raw": "raw", "weight_dist_aligned": "aligned"}
        for k in tracked_weight:
            dax.plot(epochs, m[k], label=f"weight {wlabels[k]}")
        dax.set_ylabel("Weight dist")
        if tracked_weight:
            dax.legend(loc="upper left", fontsize=6)
        if tracked_func:
            fax = dax.twinx()
            fstyles = {
                "func_prob_rmse": ("prob rmse", "C2--"),
                "func_disagree": ("argmax disagree", "C3:"),
            }
            for k in tracked_func:
                lbl, style = fstyles[k]
                fax.plot(epochs, m[k], style, label=lbl)
            fax.set_ylabel("Functional dist")
            fax.legend(loc="upper right", fontsize=6)

    weights_ax = ax[2] if has_tracked else ax[1]
    for stat, lab in [
        ("weight_max", "max |w|"),
        ("weight_mean", "mean |w|"),
        ("weight_median", "median |w|"),
        ("weight_norm", "‖w‖"),
    ]:
        if present(stat):
            weights_ax.plot(epochs, m[stat], label=lab)
    weights_ax.set_ylabel("Weight stats")
    weights_ax.set_xlabel("Epoch")
    weights_ax.set_yscale("log")
    weights_ax.legend()

    ax[0].set_xscale("log")
    fig.tight_layout()
    name = os.path.basename(log_dir)
    savefile = image_path(f"metrics_{name}.jpg")
    fig.savefig(savefile)
    print(f"Saved {savefile}\n")


if __name__ == "__main__":
    logs = os.listdir("log")
    if not os.path.exists("log") or len(logs) == 0:
        print("No logs found. Run train.py first.")
        exit()

    for log in logs:
        log_dir = os.path.join("log", log)
        if "--anim" in argv:
            from sklearn.decomposition import PCA

            animate_embedddings(log_dir)
        plot_metrics(log_dir)
