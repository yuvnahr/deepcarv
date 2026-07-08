"""
src/visualization/plots.py
------------------------------
Publication-quality plotting utilities with consistent styling, shared
across all models/runs. No model-specific logic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_STYLE = {
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
}


def _apply_style() -> None:
    plt.rcParams.update(_STYLE)  # type: ignore[arg-type]


def plot_loss_curve(train_loss: list[float], val_loss: list[float], out_path: Path, title: str = "Loss per Epoch") -> Path:
    _apply_style()
    epochs = range(1, len(train_loss) + 1)
    fig, ax = plt.subplots()
    ax.plot(epochs, train_loss, label="Train", linewidth=2)
    ax.plot(epochs, val_loss, label="Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return Path(out_path)


def plot_accuracy_curve(train_acc: list[float], val_acc: list[float], out_path: Path, title: str = "Accuracy per Epoch") -> Path:
    _apply_style()
    epochs = range(1, len(train_acc) + 1)
    fig, ax = plt.subplots()
    ax.plot(epochs, train_acc, label="Train", linewidth=2)
    ax.plot(epochs, val_acc, label="Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return Path(out_path)


def plot_lr_curve(lr_history: list[float], out_path: Path, title: str = "Learning Rate per Epoch") -> Path | None:
    """Returns None (and skips writing) if there's no meaningful LR history."""
    if not lr_history or len(set(lr_history)) <= 1:
        return None
    _apply_style()
    epochs = range(1, len(lr_history) + 1)
    fig, ax = plt.subplots()
    ax.plot(epochs, lr_history, linewidth=2, color="darkorange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return Path(out_path)


def plot_class_distribution(labels: np.ndarray, out_path: Path, title: str = "Class Distribution") -> Path:
    _apply_style()
    values, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(max(8, len(values) * 0.15), 5))
    ax.bar(values, counts, color="steelblue")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return Path(out_path)
