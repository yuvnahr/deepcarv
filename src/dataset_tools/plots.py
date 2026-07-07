"""Plot generation for FFT-75 dataset audit reports."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "deepcarv_mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "deepcarv_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.dataset_tools.io import ensure_output_dir
from src.dataset_tools.statistics import DatasetStatistics


def _save(fig: plt.Figure, output_dir: Path, stem: str, save_pdf: bool = True) -> list[str]:
    paths = [output_dir / f"{stem}.png"]
    fig.savefig(paths[0], dpi=180, bbox_inches="tight")
    if save_pdf:
        pdf_path = output_dir / f"{stem}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        paths.append(pdf_path)
    plt.close(fig)
    return [str(path) for path in paths]


def plot_class_distribution(stats: DatasetStatistics, output_dir: Path | str) -> list[str]:
    """Plot class counts per split and overall class distribution."""
    out = ensure_output_dir(output_dir)
    labels = sorted(stats.overall_class_distribution)
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.35), 4.8))
    for offset, split in enumerate(("train", "val", "test")):
        counts = [stats.splits[split].per_class_counts.get(label, 0) for label in labels]
        ax.bar(x + (offset - 1) * width, counts, width=width, label=split)
    ax.set_title("Class Distribution by Split")
    ax.set_xlabel("Class label")
    ax.set_ylabel("Samples")
    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in labels], rotation=45, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, out, "class_distribution")


def plot_overall_class_distribution(stats: DatasetStatistics, output_dir: Path | str) -> list[str]:
    """Plot overall class counts."""
    out = ensure_output_dir(output_dir)
    labels = sorted(stats.overall_class_distribution)
    counts = [stats.overall_class_distribution[label] for label in labels]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.32), 4.5))
    ax.bar([str(label) for label in labels], counts, color="#2f6f8f")
    ax.set_title("Overall Class Distribution")
    ax.set_xlabel("Class label")
    ax.set_ylabel("Samples")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, out, "overall_class_distribution")


def plot_byte_histogram(stats: DatasetStatistics, output_dir: Path | str) -> list[str]:
    """Plot aggregate byte-value histogram."""
    out = ensure_output_dir(output_dir)
    hist = np.zeros(256, dtype=np.int64)
    for split_stats in stats.splits.values():
        hist += np.array(split_stats.byte_histogram, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(256), hist, color="#2a6f5f", linewidth=1.8)
    ax.set_title("Byte Value Histogram")
    ax.set_xlabel("Byte value")
    ax.set_ylabel("Frequency")
    ax.set_xlim(0, 255)
    ax.grid(alpha=0.25)
    return _save(fig, out, "byte_histogram")


def plot_entropy_distribution(stats: DatasetStatistics, output_dir: Path | str) -> list[str]:
    """Plot entropy histograms per split."""
    out = ensure_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for split, split_stats in stats.splits.items():
        edges = np.array(split_stats.entropy_histogram["bin_edges"], dtype=np.float64)
        counts = np.array(split_stats.entropy_histogram["counts"], dtype=np.int64)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, counts, linewidth=1.8, label=split)
    ax.set_title("Fragment Entropy Distribution")
    ax.set_xlabel("Shannon entropy")
    ax.set_ylabel("Fragments")
    ax.set_xlim(0, 8)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    return _save(fig, out, "entropy_histogram")


def plot_split_counts(stats: DatasetStatistics, output_dir: Path | str) -> list[str]:
    """Plot sample count by split."""
    out = ensure_output_dir(output_dir)
    splits = list(stats.splits)
    counts = [stats.splits[split].sample_count for split in splits]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(splits, counts, color=["#3f6c9b", "#b45b48", "#5f8b4c"])
    ax.set_title("Split Sample Counts")
    ax.set_xlabel("Split")
    ax.set_ylabel("Samples")
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, out, "split_sample_counts")


def plot_duplicate_summary(stats: DatasetStatistics, output_dir: Path | str) -> list[str]:
    """Plot within-split duplicate samples and cross-split overlap hashes."""
    out = ensure_output_dir(output_dir)
    leakage = stats.leakage_report
    labels = list(leakage.duplicate_samples_per_split)
    values = [leakage.duplicate_samples_per_split[label] for label in labels]
    overlap_labels = [f"{item.left_split}/{item.right_split}" for item in leakage.overlaps]
    overlap_values = [item.overlapping_hashes for item in leakage.overlaps]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, values, color="#7a5c9e")
    axes[0].set_title("Within-Split Duplicates")
    axes[0].set_ylabel("Duplicate samples")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(overlap_labels, overlap_values, color="#a3673f")
    axes[1].set_title("Cross-Split Overlap")
    axes[1].set_ylabel("Overlapping hashes")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.25)
    return _save(fig, out, "duplicate_summary")


def generate_all_plots(stats: DatasetStatistics, output_dir: Path | str) -> dict[str, list[str]]:
    """Generate all standard audit plots."""
    return {
        "class_distribution": plot_class_distribution(stats, output_dir),
        "overall_class_distribution": plot_overall_class_distribution(stats, output_dir),
        "byte_histogram": plot_byte_histogram(stats, output_dir),
        "entropy_histogram": plot_entropy_distribution(stats, output_dir),
        "split_sample_counts": plot_split_counts(stats, output_dir),
        "duplicate_summary": plot_duplicate_summary(stats, output_dir),
    }
