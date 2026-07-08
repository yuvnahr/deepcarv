"""
src/visualization/confusion.py
----------------------------------
Confusion matrix rendering, split out from plots.py since it needs
label-count-aware sizing and optional normalization.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    cm: np.ndarray,
    out_path: Path,
    class_names: list[str] | None = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> Path:
    """Render a confusion matrix. Falls back to no tick labels for large
    class counts (e.g. 75 FFT-75 classes) to keep the figure readable."""
    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums

    n = cm.shape[0]
    fig_size = min(max(6, n * 0.18), 20)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    if class_names is not None and n <= 40:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)
