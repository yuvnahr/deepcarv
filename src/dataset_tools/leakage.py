"""Hash-based duplicate and leakage detection for FFT-75 split files."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.dataset_tools.io import SPLITS, load_npz, split_path


@dataclass(frozen=True)
class HashOverlap:
    """Overlap details for two splits."""

    left_split: str
    right_split: str
    overlapping_hashes: int
    overlapping_samples_left: int
    overlapping_samples_right: int
    examples: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class LeakageReport:
    """Duplicate and leakage summary for a dataset fragment size."""

    duplicate_hashes_per_split: dict[str, int]
    duplicate_samples_per_split: dict[str, int]
    pair_duplicate_hashes_per_split: dict[str, int]
    overlaps: list[HashOverlap]
    pair_overlaps: list[HashOverlap]
    warnings: list[str]

    @property
    def has_leakage(self) -> bool:
        """Return whether any split-to-split overlap was found."""
        return any(overlap.overlapping_hashes > 0 for overlap in self.overlaps)


def _hash_array_row(row: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(row)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _hash_fragment_label(row: np.ndarray, label: int) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(_hash_array_row(row).encode("utf-8"))
    digest.update(str(label).encode("utf-8"))
    return digest.hexdigest()


def split_hashes(path: Path | str) -> tuple[list[str], list[str]]:
    """Return fragment hashes and fragment-label pair hashes for one split."""
    with load_npz(path) as data:
        x = data["X"]
        y = data["y"].reshape(-1)
        fragment_hashes = [_hash_array_row(row) for row in x]
        pair_hashes = [_hash_fragment_label(row, int(label)) for row, label in zip(x, y)]
    return fragment_hashes, pair_hashes


def detect_leakage(root: Path | str, fragment_size: int, max_examples: int = 5) -> LeakageReport:
    """Detect exact duplicate fragments within and across train/val/test splits."""
    hash_lists: dict[str, list[str]] = {}
    pair_hash_lists: dict[str, list[str]] = {}
    duplicate_hashes_per_split: dict[str, int] = {}
    duplicate_samples_per_split: dict[str, int] = {}
    pair_duplicate_hashes_per_split: dict[str, int] = {}
    warnings: list[str] = []

    for split in SPLITS:
        path = split_path(root, fragment_size, split)
        fragment_hashes, pair_hashes = split_hashes(path)
        hash_lists[split] = fragment_hashes
        pair_hash_lists[split] = pair_hashes
        counts = Counter(fragment_hashes)
        pair_counts = Counter(pair_hashes)
        duplicate_hashes_per_split[split] = sum(1 for count in counts.values() if count > 1)
        duplicate_samples_per_split[split] = sum(
            count - 1 for count in counts.values() if count > 1
        )
        pair_duplicate_hashes_per_split[split] = sum(
            1 for count in pair_counts.values() if count > 1
        )
        if duplicate_hashes_per_split[split]:
            warnings.append(
                f"{split}: {duplicate_hashes_per_split[split]} repeated fragment hash(es) "
                f"covering {duplicate_samples_per_split[split]} duplicate sample(s)"
            )

    overlaps: list[HashOverlap] = []
    pair_overlaps: list[HashOverlap] = []
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            left_positions: dict[str, list[int]] = defaultdict(list)
            right_positions: dict[str, list[int]] = defaultdict(list)
            for row_index, digest in enumerate(hash_lists[left]):
                left_positions[digest].append(row_index)
            for row_index, digest in enumerate(hash_lists[right]):
                right_positions[digest].append(row_index)

            overlap_hashes = sorted(set(left_positions) & set(right_positions))
            examples = [
                {
                    "hash": digest,
                    f"{left}_indices": left_positions[digest][:max_examples],
                    f"{right}_indices": right_positions[digest][:max_examples],
                }
                for digest in overlap_hashes[:max_examples]
            ]
            overlap = HashOverlap(
                left_split=left,
                right_split=right,
                overlapping_hashes=len(overlap_hashes),
                overlapping_samples_left=sum(
                    len(left_positions[digest]) for digest in overlap_hashes
                ),
                overlapping_samples_right=sum(
                    len(right_positions[digest]) for digest in overlap_hashes
                ),
                examples=examples,
            )
            overlaps.append(overlap)
            if overlap.overlapping_hashes:
                warnings.append(
                    f"{left}/{right}: {overlap.overlapping_hashes} overlapping fragment hash(es)"
                )

            left_pair_positions: dict[str, list[int]] = defaultdict(list)
            right_pair_positions: dict[str, list[int]] = defaultdict(list)
            for row_index, digest in enumerate(pair_hash_lists[left]):
                left_pair_positions[digest].append(row_index)
            for row_index, digest in enumerate(pair_hash_lists[right]):
                right_pair_positions[digest].append(row_index)

            pair_overlap_hashes = sorted(set(left_pair_positions) & set(right_pair_positions))
            pair_examples = [
                {
                    "hash": digest,
                    f"{left}_indices": left_pair_positions[digest][:max_examples],
                    f"{right}_indices": right_pair_positions[digest][:max_examples],
                }
                for digest in pair_overlap_hashes[:max_examples]
            ]
            pair_overlap = HashOverlap(
                left_split=left,
                right_split=right,
                overlapping_hashes=len(pair_overlap_hashes),
                overlapping_samples_left=sum(
                    len(left_pair_positions[digest]) for digest in pair_overlap_hashes
                ),
                overlapping_samples_right=sum(
                    len(right_pair_positions[digest]) for digest in pair_overlap_hashes
                ),
                examples=pair_examples,
            )
            pair_overlaps.append(pair_overlap)
            if pair_overlap.overlapping_hashes:
                warnings.append(
                    f"{left}/{right}: {pair_overlap.overlapping_hashes} overlapping "
                    "(fragment, label) pair hash(es)"
                )

    return LeakageReport(
        duplicate_hashes_per_split=duplicate_hashes_per_split,
        duplicate_samples_per_split=duplicate_samples_per_split,
        pair_duplicate_hashes_per_split=pair_duplicate_hashes_per_split,
        overlaps=overlaps,
        pair_overlaps=pair_overlaps,
        warnings=warnings,
    )
