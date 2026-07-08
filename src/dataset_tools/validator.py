"""Validation for FFT-75 NPZ dataset directories."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.dataset_tools.io import SPLITS, fragment_dir, load_npz, split_path
from src.dataset_tools.npz_inspector import inspect_npz


class DatasetValidationError(RuntimeError):
    """Raised when a dataset fails validation."""


@dataclass(frozen=True)
class DatasetValidationResult:
    """Structured validation result for an FFT-75 fragment directory."""

    ok: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def raise_if_invalid(self) -> None:
        """Raise DatasetValidationError if validation failed."""
        if not self.ok:
            details = "\n".join(f"- {error}" for error in self.errors)
            raise DatasetValidationError(f"Dataset validation failed:\n{details}")


def _is_integer_labels(y: np.ndarray) -> bool:
    if np.issubdtype(y.dtype, np.integer):
        return True
    if np.issubdtype(y.dtype, np.floating):
        return bool(np.all(np.isfinite(y)) and np.all(y == np.floor(y)))
    return False


def _validate_arrays(
    path: Path,
    fragment_size: int,
    split: str,
) -> tuple[list[str], list[str], dict[str, Any], set[int]]:
    errors: list[str] = []
    warnings: list[str] = []
    labels: set[int] = set()
    summary: dict[str, Any] = {}

    inspection = inspect_npz(path)
    summary.update(inspection.to_dict())
    if inspection.errors:
        errors.extend(f"{split}: {error}" for error in inspection.errors)
        return errors, warnings, summary, labels
    extra_keys = sorted(set(inspection.keys) - {"X", "y"})
    if extra_keys:
        errors.append(f"{split}: NPZ must contain exactly X and y; extra key(s): {extra_keys}")

    with load_npz(path) as data:
        x = data["X"]
        y = data["y"]
        y_flat = y.reshape(-1)
        summary["flattened_y_shape"] = tuple(int(item) for item in y_flat.shape)

        if len(x) == 0:
            errors.append(f"{split}: split is empty")
        if len(x) != len(y_flat):
            errors.append(f"{split}: len(X)={len(x)} does not match len(y)={len(y_flat)}")
        if not np.issubdtype(x.dtype, np.number):
            errors.append(f"{split}: X must be numeric, got {x.dtype}")
        if not np.issubdtype(y.dtype, np.number):
            errors.append(f"{split}: y must be numeric, got {y.dtype}")
        if x.ndim < 2:
            errors.append(f"{split}: X must have samples and fragment dimensions, got {x.shape}")
        elif int(np.prod(x.shape[1:])) != fragment_size:
            errors.append(
                f"{split}: fragment length mismatch, expected {fragment_size}, got "
                f"{int(np.prod(x.shape[1:]))} from X.shape={x.shape}"
            )
        if y.ndim > 2 or (y.ndim == 2 and 1 not in y.shape):
            errors.append(f"{split}: y must be 1D or flattenable singleton 2D, got {y.shape}")
        if x.size and (np.nanmin(x) < 0 or np.nanmax(x) > 255):
            errors.append(f"{split}: X values must stay in byte range [0, 255]")
        if x.size and not np.all(np.isfinite(x)):
            errors.append(f"{split}: X contains NaN or infinite values")
        if y.size and not np.all(np.isfinite(y)):
            errors.append(f"{split}: y contains NaN or infinite values")
        if y.size and not _is_integer_labels(y):
            errors.append(f"{split}: label values must be valid integers")

        if y.size and _is_integer_labels(y):
            labels = {int(label) for label in np.unique(y_flat)}
            if any(label < 0 for label in labels):
                warnings.append(f"{split}: labels include negative values")

    return errors, warnings, summary, labels


def validate_dataset(root: Path | str, fragment_size: int) -> DatasetValidationResult:
    """Validate one FFT-75 fragment-size directory containing train/val/test NPZ files."""
    dataset_root = Path(root)
    errors: list[str] = []
    warnings: list[str] = []
    summary: dict[str, Any] = {
        "root": str(dataset_root),
        "fragment_size": fragment_size,
        "splits": {},
    }

    available_fragment_dirs = [
        int(path.name) for path in dataset_root.iterdir() if path.is_dir() and path.name.isdigit()
    ] if dataset_root.exists() else []
    summary["available_fragment_sizes"] = sorted(available_fragment_dirs)

    frag_dir = fragment_dir(dataset_root, fragment_size)
    if not dataset_root.exists():
        errors.append(f"Dataset root does not exist: {dataset_root}")
    if not frag_dir.exists():
        errors.append(f"Fragment directory does not exist: {frag_dir}")
        return DatasetValidationResult(False, warnings, errors, summary)

    label_spaces: dict[str, set[int]] = {}
    for split in SPLITS:
        path = split_path(dataset_root, fragment_size, split)
        if not path.exists():
            errors.append(f"{split}: missing required file {path}")
            continue
        split_errors, split_warnings, split_summary, labels = _validate_arrays(
            path, fragment_size, split
        )
        errors.extend(split_errors)
        warnings.extend(split_warnings)
        summary["splits"][split] = split_summary
        label_spaces[split] = labels

    non_empty_label_spaces = [labels for labels in label_spaces.values() if labels]
    if non_empty_label_spaces:
        shared = set.intersection(*non_empty_label_spaces)
        union = set.union(*non_empty_label_spaces)
        summary["shared_label_space"] = sorted(shared)
        summary["label_space_union"] = sorted(union)
        for split, labels in label_spaces.items():
            if labels and labels != union:
                missing = sorted(union - labels)
                warnings.append(f"{split}: missing labels present in another split: {missing}")

    return DatasetValidationResult(ok=not errors, warnings=warnings, errors=errors, summary=summary)
