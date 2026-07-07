"""Shared IO helpers for FFT-75 dataset tooling."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

SPLITS: tuple[str, str, str] = ("train", "val", "test")
REQUIRED_KEYS: tuple[str, str] = ("X", "y")


def fragment_dir(root: Path | str, fragment_size: int) -> Path:
    """Return the directory containing split NPZ files for a fragment size."""
    return Path(root) / str(fragment_size)


def split_path(root: Path | str, fragment_size: int, split: str) -> Path:
    """Return the NPZ path for one split."""
    return fragment_dir(root, fragment_size) / f"{split}.npz"


def load_npz(path: Path | str) -> np.lib.npyio.NpzFile:
    """Open an NPZ file without copying arrays into memory."""
    return np.load(Path(path), allow_pickle=False)


def estimate_nbytes(*arrays: np.ndarray) -> int:
    """Return the total memory footprint in bytes for arrays."""
    return int(sum(array.nbytes for array in arrays))


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses, NumPy values, and paths into JSON-safe objects."""
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path | str, data: Any) -> None:
    """Write JSON data with stable formatting."""
    Path(path).write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n")


def class_counts(y: np.ndarray) -> dict[int, int]:
    """Return per-class sample counts for a label vector."""
    labels, counts = np.unique(y.reshape(-1), return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def ensure_output_dir(path: Path | str) -> Path:
    """Create and return an output directory."""
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output
