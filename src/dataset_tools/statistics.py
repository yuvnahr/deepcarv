"""Statistics for FFT-75 NPZ datasets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.dataset_tools.io import SPLITS, class_counts, estimate_nbytes, load_npz, split_path
from src.dataset_tools.leakage import LeakageReport, detect_leakage


@dataclass(frozen=True)
class SplitStatistics:
    """Statistics for one dataset split."""

    split: str
    sample_count: int
    class_count: int
    per_class_counts: dict[int, int]
    class_imbalance_ratio: float
    mean_fragment_value: float
    std_fragment_value: float
    min_byte_value: float
    max_byte_value: float
    median_fragment_entropy: float
    mean_fragment_entropy: float
    duplicate_rate: float
    approximate_memory_bytes: int
    byte_histogram: list[int] = field(default_factory=list)
    entropy_histogram: dict[str, list[float | int]] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetStatistics:
    """Statistics across all dataset splits."""

    root: str
    fragment_size: int
    splits: dict[str, SplitStatistics]
    total_sample_count: int
    shared_label_space: list[int]
    label_coverage_consistency: bool
    overall_class_distribution: dict[int, int]
    imbalance_severity: str
    duplicate_rate_across_splits: float
    leakage_report: LeakageReport


def fragment_entropy(x: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy per fragment row."""
    rows = x.reshape((x.shape[0], -1))
    entropies = np.zeros(rows.shape[0], dtype=np.float64)
    for index, row in enumerate(rows):
        counts = np.bincount(row.astype(np.uint8, copy=False), minlength=256)
        probabilities = counts[counts > 0] / row.size
        entropies[index] = -float(np.sum(probabilities * np.log2(probabilities)))
    return entropies


def imbalance_ratio(counts: dict[int, int]) -> float:
    """Return max/min class count ratio, or 0.0 when unavailable."""
    nonzero = [count for count in counts.values() if count > 0]
    if not nonzero:
        return 0.0
    return float(max(nonzero) / min(nonzero))


def _severity(ratio: float) -> str:
    if ratio <= 1.5:
        return "low"
    if ratio <= 3.0:
        return "moderate"
    return "high"


def compute_split_statistics(
    path: Path | str,
    split: str,
    duplicate_samples: int = 0,
) -> SplitStatistics:
    """Compute class, byte, entropy, duplicate, and memory stats for one split."""
    with load_npz(path) as data:
        x = data["X"]
        y = data["y"].reshape(-1)
        counts = class_counts(y)
        entropies = fragment_entropy(x) if len(x) else np.array([], dtype=np.float64)
        byte_hist = np.bincount(x.astype(np.uint8, copy=False).reshape(-1), minlength=256)
        entropy_counts, entropy_edges = np.histogram(entropies, bins=20, range=(0.0, 8.0))
        duplicate_rate = float(duplicate_samples / len(x)) if len(x) else 0.0
        return SplitStatistics(
            split=split,
            sample_count=int(len(x)),
            class_count=int(len(counts)),
            per_class_counts=counts,
            class_imbalance_ratio=imbalance_ratio(counts),
            mean_fragment_value=float(np.mean(x)) if x.size else 0.0,
            std_fragment_value=float(np.std(x)) if x.size else 0.0,
            min_byte_value=float(np.min(x)) if x.size else 0.0,
            max_byte_value=float(np.max(x)) if x.size else 0.0,
            median_fragment_entropy=float(np.median(entropies)) if entropies.size else 0.0,
            mean_fragment_entropy=float(np.mean(entropies)) if entropies.size else 0.0,
            duplicate_rate=duplicate_rate,
            approximate_memory_bytes=estimate_nbytes(x, y),
            byte_histogram=[int(item) for item in byte_hist.tolist()],
            entropy_histogram={
                "counts": [int(item) for item in entropy_counts.tolist()],
                "bin_edges": [float(item) for item in entropy_edges.tolist()],
            },
        )


def compute_dataset_statistics(root: Path | str, fragment_size: int) -> DatasetStatistics:
    """Compute split-level and dataset-level FFT-75 statistics."""
    leakage = detect_leakage(root, fragment_size)
    split_stats: dict[str, SplitStatistics] = {}
    overall_counts: Counter[int] = Counter()
    label_spaces: list[set[int]] = []

    for split in SPLITS:
        stats = compute_split_statistics(
            split_path(root, fragment_size, split),
            split,
            duplicate_samples=leakage.duplicate_samples_per_split.get(split, 0),
        )
        split_stats[split] = stats
        overall_counts.update(stats.per_class_counts)
        label_spaces.append(set(stats.per_class_counts))

    shared = sorted(set.intersection(*label_spaces)) if label_spaces else []
    union = set.union(*label_spaces) if label_spaces else set()
    total = sum(stats.sample_count for stats in split_stats.values())
    cross_overlap_samples = sum(overlap.overlapping_samples_left for overlap in leakage.overlaps)
    duplicate_rate = float(cross_overlap_samples / total) if total else 0.0
    overall_distribution = {
        int(label): int(count) for label, count in sorted(overall_counts.items())
    }

    return DatasetStatistics(
        root=str(Path(root)),
        fragment_size=fragment_size,
        splits=split_stats,
        total_sample_count=int(total),
        shared_label_space=[int(label) for label in shared],
        label_coverage_consistency=all(labels == union for labels in label_spaces),
        overall_class_distribution=overall_distribution,
        imbalance_severity=_severity(imbalance_ratio(overall_distribution)),
        duplicate_rate_across_splits=duplicate_rate,
        leakage_report=leakage,
    )
