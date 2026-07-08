"""Dataset fingerprint generation for FFT-75 NPZ splits."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.dataset_tools.io import SPLITS, class_counts, load_npz, split_path, write_json


@dataclass(frozen=True)
class SplitFingerprint:
    """Fingerprint details for one split file."""

    split: str
    path: str
    sha256: str
    sample_count: int
    class_count: int
    class_counts: dict[int, int]


@dataclass(frozen=True)
class DatasetFingerprint:
    """Reproducibility fingerprint for an FFT-75 fragment-size dataset."""

    dataset: str
    root: str
    fragment_size: int
    fft75_version: str
    timestamp: str
    splits: dict[str, SplitFingerprint] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "dataset": self.dataset,
            "root": self.root,
            "fragment_size": self.fragment_size,
            "fft75_version": self.fft75_version,
            "timestamp": self.timestamp,
            "splits": {
                split: {
                    "split": data.split,
                    "path": data.path,
                    "sha256": data.sha256,
                    "sample_count": data.sample_count,
                    "class_count": data.class_count,
                    "class_counts": data.class_counts,
                }
                for split, data in self.splits.items()
            },
        }


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA256 digest for a file without loading it all at once."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def generate_dataset_fingerprint(
    root: Path | str,
    fragment_size: int,
    fft75_version: str = "unknown",
) -> DatasetFingerprint:
    """Generate a fingerprint for train/val/test NPZ files."""
    dataset_root = Path(root)
    splits: dict[str, SplitFingerprint] = {}
    for split in SPLITS:
        path = split_path(dataset_root, fragment_size, split)
        with load_npz(path) as data:
            y = data["y"].reshape(-1)
            counts = class_counts(y)
            splits[split] = SplitFingerprint(
                split=split,
                path=str(path),
                sha256=sha256_file(path),
                sample_count=int(len(y)),
                class_count=int(len(counts)),
                class_counts=counts,
            )
    return DatasetFingerprint(
        dataset="FFT-75",
        root=str(dataset_root),
        fragment_size=fragment_size,
        fft75_version=fft75_version,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        splits=splits,
    )


def write_dataset_fingerprint(
    root: Path | str,
    fragment_size: int,
    output_path: Path | str,
    fft75_version: str = "unknown",
) -> DatasetFingerprint:
    """Generate and write dataset_fingerprint.json."""
    fingerprint = generate_dataset_fingerprint(root, fragment_size, fft75_version)
    write_json(output_path, fingerprint.to_dict())
    return fingerprint
