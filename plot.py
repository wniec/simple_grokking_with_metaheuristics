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

IMAGES_DIR = "images"


def _get_metrics(log_dir, skip=1):
    jsonl_path = os.path.join(log_dir, "metrics.jsonl")
    csv_path = os.path.join(log_dir, "metrics.csv")
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            rows = [json.loads(line) for line in f][::skip]

        def col(key):
            return np.array([r.get(key, float("nan")) for r in rows])

        return (
            col("epoch").astype(int),
            col("train_loss"),
            col("train_acc"),
            col("val_loss"),
            col("val_acc"),
            col("weight_max"),
            col("weight_mean"),
            col("weight_median"),
            col("weight_norm"),
        )
    metrics = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    def csv_col(i, default=float("nan")):
        return (
            metrics[::skip, i]
            if metrics.shape[1] > i
            else np.full(len(metrics[::skip, 0]), default)
        )

    return (
        metrics[::skip, 0].astype(int),
        csv_col(1),
        csv_col(2),
        csv_col(3),
        csv_col(4),
        csv_col(5),
        csv_col(6),
        csv_col(7),
        csv_col(8),
    )


def _load_embeddings(log_dir, epoch=None):
    model.load_state_dict(torch.load(os.path.join(log_dir, "model.pt")))
    return model.embedding.weight.detach().numpy()


def animate_embedddings(log_dir):
    # Load Data
    print(f"Loading {log_dir}...")
    epochs, train_loss, train_acc, val_loss, val_acc, *_ = _get_metrics(log_dir, skip=1)

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
    savefile = os.path.join(IMAGES_DIR, f"emb_{name}.mp4")
    anim.save(savefile, writer="ffmpeg", fps=len(epochs) // 10)
    print(f"Saved {savefile}\n")


def plot_metrics(log_dir):
    print(f"Loading {log_dir}...")
    (
        epochs,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
        weight_max,
        weight_mean,
        weight_median,
        weight_norm,
    ) = _get_metrics(log_dir)

    fig, ax = plt.subplots(3, 1, sharex=True, dpi=200)
    ax[0].plot(epochs, train_loss, label="Train")
    ax[0].plot(epochs, val_loss, label="Val")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].set_yscale("log")
    ax[0].set_ylim(train_loss.min() * 0.9, val_loss.max() * 1.0)

    ax[1].plot(epochs, train_acc, label="Train")
    ax[1].plot(epochs, val_acc, label="Val")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    ax[2].plot(epochs, weight_max, label="max |w|")
    ax[2].plot(epochs, weight_mean, label="mean |w|")
    ax[2].plot(epochs, weight_median, label="median |w|")
    ax[2].plot(epochs, weight_norm, label="‖w‖")
    ax[2].set_ylabel("Weight stats")
    ax[2].set_xlabel("Epoch")
    ax[2].set_yscale("log")
    ax[2].legend()

    ax[0].set_xscale("log")
    fig.tight_layout()
    name = os.path.basename(log_dir)
    savefile = os.path.join(IMAGES_DIR, f"metrics_{name}.jpg")
    fig.savefig(savefile)
    print(f"Saved {savefile}\n")


if __name__ == "__main__":
    logs = os.listdir("log")
    if not os.path.exists("log") or len(logs) == 0:
        print("No logs found. Run train.py first.")
        exit()

    os.makedirs(IMAGES_DIR, exist_ok=True)

    for log in logs:
        log_dir = os.path.join("log", log)
        if "--anim" in argv:
            from sklearn.decomposition import PCA

            animate_embedddings(log_dir)
        plot_metrics(log_dir)
