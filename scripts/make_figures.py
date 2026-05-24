from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/synthetic_small")
    parser.add_argument("--figure-dir", default="docs/figures")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    _plot_history(run_dir / "history.csv", figure_dir / "training_curve.png")
    _plot_graphs(
        run_dir / "ground_truth_graph.npy",
        run_dir / "learned_graph.npy",
        figure_dir / "graph_recovery.png",
    )
    _plot_predictions(run_dir / "test_predictions.npz", figure_dir / "population_projection.png")
    _plot_metrics(run_dir / "metrics.json", figure_dir / "heldout_metrics.png")
    _plot_architecture(figure_dir / "architecture.png")
    _plot_hero(figure_dir / "repo_hero.png")
    print(f"wrote figures to {figure_dir}")


def _plot_history(path: Path, output: Path) -> None:
    history = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=160)
    ax.plot(history["epoch"], history["train_loss"], label="train loss", color="#171717", linewidth=2)
    ax.plot(history["epoch"], history["val_mmd"], label="val MMD", color="#c45f2d", linewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.set_title("Training signal")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _plot_graphs(truth_path: Path, learned_path: Path, output: Path) -> None:
    truth = np.load(truth_path)
    learned = np.load(learned_path)
    vmax = float(max(np.abs(truth).max(), np.abs(learned).max()))
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.1), dpi=160)
    for ax, matrix, title in zip(axes, [truth, learned], ["ground truth", "learned"]):
        image = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("source gene")
        ax.set_ylabel("target gene")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="signed edge")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_predictions(path: Path, output: Path) -> None:
    data = np.load(path)
    target = data["target"].reshape(-1, data["target"].shape[-1])
    prediction = data["prediction"].reshape(-1, data["prediction"].shape[-1])
    combined = np.concatenate([target, prediction], axis=0)
    coords = PCA(n_components=2).fit_transform(combined)
    n = target.shape[0]

    fig, ax = plt.subplots(figsize=(6.6, 5.2), dpi=160)
    ax.scatter(coords[:n, 0], coords[:n, 1], s=9, alpha=0.42, label="target", color="#171717")
    ax.scatter(coords[n:, 0], coords[n:, 1], s=9, alpha=0.42, label="prediction", color="#c45f2d")
    ax.set_title("Held-out population projection")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False)
    ax.grid(alpha=0.16)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _plot_metrics(path: Path, output: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    rows: Dict[str, Dict[str, float]] = {"model": report["test_metrics"]}
    rows.update(report["baseline_metrics"])
    names = list(rows.keys())
    mmd = [rows[name]["mmd"] for name in names]
    mae = [rows[name]["mae"] for name in names]

    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=160)
    ax.bar(x - width / 2, mmd, width, label="MMD", color="#171717")
    ax.bar(x + width / 2, mae, width, label="MAE", color="#c45f2d")
    ax.set_xticks(x, names, rotation=12)
    ax.set_title("Held-out combinatorial perturbations")
    ax.set_ylabel("lower is better")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _plot_architecture(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.4), dpi=160)
    ax.axis("off")
    boxes = [
        ("source cells", 0.05, 0.55, "#f2ede6"),
        ("population stats", 0.24, 0.75, "#f8f5ef"),
        ("gene graph", 0.24, 0.35, "#f8f5ef"),
        ("intervention context", 0.45, 0.75, "#f8f5ef"),
        ("latent transition", 0.45, 0.35, "#f8f5ef"),
        ("probabilistic decoder", 0.67, 0.55, "#f8f5ef"),
        ("counterfactual cells", 0.84, 0.55, "#f2ede6"),
    ]
    for label, x, y, color in boxes:
        ax.add_patch(
            plt.Rectangle((x, y - 0.09), 0.14, 0.18, facecolor=color, edgecolor="#171717", lw=1.2)
        )
        ax.text(x + 0.07, y, label, ha="center", va="center", fontsize=9)
    arrows = [
        ((0.19, 0.55), (0.24, 0.75)),
        ((0.19, 0.55), (0.24, 0.35)),
        ((0.38, 0.75), (0.45, 0.75)),
        ((0.38, 0.35), (0.45, 0.35)),
        ((0.59, 0.75), (0.67, 0.55)),
        ((0.59, 0.35), (0.67, 0.55)),
        ((0.81, 0.55), (0.84, 0.55)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.set_title("Counterfactual Cell World model path", loc="left", fontsize=13, pad=14)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _plot_hero(output: Path) -> None:
    rng = np.random.default_rng(17)
    left = rng.normal(loc=(-1.7, 0.0), scale=(0.35, 0.48), size=(360, 2))
    right = rng.normal(loc=(1.7, 0.0), scale=(0.48, 0.52), size=(360, 2))
    anchors = rng.uniform(low=(-2.4, -1.1), high=(2.4, 1.1), size=(34, 2))

    fig, ax = plt.subplots(figsize=(11.0, 4.7), dpi=170)
    ax.set_facecolor("#fbf9f5")
    fig.patch.set_facecolor("#fbf9f5")
    ax.axis("off")

    for _ in range(120):
        start = left[int(rng.integers(0, len(left)))]
        end = right[int(rng.integers(0, len(right)))]
        bend = rng.normal(0.0, 0.16)
        xs = np.linspace(start[0], end[0], 80)
        ys = np.linspace(start[1], end[1], 80) + bend * np.sin(np.linspace(0, np.pi, 80))
        ax.plot(xs, ys, color="#d88948", alpha=0.12, lw=0.8)

    for anchor in anchors:
        neighbors = anchors[np.argsort(np.linalg.norm(anchors - anchor, axis=1))[1:4]]
        for neighbor in neighbors:
            color = "#171717" if anchor[0] < 0 else "#c45f2d"
            ax.plot([anchor[0], neighbor[0]], [anchor[1], neighbor[1]], color=color, alpha=0.12, lw=0.7)
    ax.scatter(anchors[:, 0], anchors[:, 1], s=rng.uniform(16, 70, size=len(anchors)), color="#171717", alpha=0.26)
    ax.scatter(left[:, 0], left[:, 1], s=rng.uniform(12, 48, size=len(left)), color="#4b5b64", alpha=0.26)
    ax.scatter(right[:, 0], right[:, 1], s=rng.uniform(12, 48, size=len(right)), color="#c45f2d", alpha=0.34)
    ax.set_xlim(-2.75, 2.75)
    ax.set_ylim(-1.35, 1.35)
    fig.tight_layout(pad=0)
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
