"""Inspection utilities for FFT-75 NPZ split files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.dataset_tools.io import REQUIRED_KEYS, class_counts, estimate_nbytes, load_npz


@dataclass(frozen=True)
class NpzInspectionResult:
    """Summary of one NPZ split file."""

    path: str
    keys: list[str]
    has_x: bool
    has_y: bool
    x_shape: tuple[int, ...] | None
    y_shape: tuple[int, ...] | None
    x_dtype: str | None
    y_dtype: str | None
    num_samples: int | None
    fragment_length: int | None
    label_count: int | None
    memory_bytes: int | None
    class_counts: dict[int, int] = field(default_factory=dict)
    value_min: float | None = None
    value_max: float | None = None
    value_mean: float | None = None
    value_std: float | None = None
    values_in_byte_range: bool | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return whether the file has the minimum inspectable structure."""
        return self.has_x and self.has_y and not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""
        return {
            "path": self.path,
            "keys": self.keys,
            "has_x": self.has_x,
            "has_y": self.has_y,
            "x_shape": self.x_shape,
            "y_shape": self.y_shape,
            "x_dtype": self.x_dtype,
            "y_dtype": self.y_dtype,
            "num_samples": self.num_samples,
            "fragment_length": self.fragment_length,
            "label_count": self.label_count,
            "memory_bytes": self.memory_bytes,
            "class_counts": self.class_counts,
            "value_min": self.value_min,
            "value_max": self.value_max,
            "value_mean": self.value_mean,
            "value_std": self.value_std,
            "values_in_byte_range": self.values_in_byte_range,
            "errors": self.errors,
            "ok": self.ok,
        }


def _fragment_length(shape: tuple[int, ...]) -> int | None:
    if len(shape) < 2:
        return None
    if len(shape) == 2:
        return int(shape[1])
    return int(np.prod(shape[1:]))


def inspect_npz(path: Path | str) -> NpzInspectionResult:
    """Inspect a single NPZ file and return shape, dtype, label, and byte stats."""
    npz_path = Path(path)
    errors: list[str] = []
    if not npz_path.exists():
        return NpzInspectionResult(
            path=str(npz_path),
            keys=[],
            has_x=False,
            has_y=False,
            x_shape=None,
            y_shape=None,
            x_dtype=None,
            y_dtype=None,
            num_samples=None,
            fragment_length=None,
            label_count=None,
            memory_bytes=None,
            errors=[f"File does not exist: {npz_path}"],
        )

    try:
        with load_npz(npz_path) as data:
            keys = list(data.files)
            missing = [key for key in REQUIRED_KEYS if key not in keys]
            if missing:
                errors.append(f"Missing required key(s): {', '.join(missing)}")
                return NpzInspectionResult(
                    path=str(npz_path),
                    keys=keys,
                    has_x="X" in keys,
                    has_y="y" in keys,
                    x_shape=None,
                    y_shape=None,
                    x_dtype=None,
                    y_dtype=None,
                    num_samples=None,
                    fragment_length=None,
                    label_count=None,
                    memory_bytes=None,
                    errors=errors,
                )

            x = data["X"]
            y = data["y"]
            y_flat = y.reshape(-1)
            if len(x) != len(y_flat):
                errors.append(f"Length mismatch: len(X)={len(x)} len(y)={len(y_flat)}")

            value_min = float(np.min(x)) if x.size else None
            value_max = float(np.max(x)) if x.size else None
            values_in_range = (
                value_min is not None
                and value_max is not None
                and value_min >= 0
                and value_max <= 255
            )
            return NpzInspectionResult(
                path=str(npz_path),
                keys=keys,
                has_x=True,
                has_y=True,
                x_shape=tuple(int(item) for item in x.shape),
                y_shape=tuple(int(item) for item in y.shape),
                x_dtype=str(x.dtype),
                y_dtype=str(y.dtype),
                num_samples=int(len(x)),
                fragment_length=_fragment_length(tuple(x.shape)),
                label_count=int(np.unique(y_flat).size),
                memory_bytes=estimate_nbytes(x, y),
                class_counts=class_counts(y_flat),
                value_min=value_min,
                value_max=value_max,
                value_mean=float(np.mean(x)) if x.size else None,
                value_std=float(np.std(x)) if x.size else None,
                values_in_byte_range=values_in_range,
                errors=errors,
            )
    except Exception as exc:  # pragma: no cover - defensive, message is surfaced.
        return NpzInspectionResult(
            path=str(npz_path),
            keys=[],
            has_x=False,
            has_y=False,
            x_shape=None,
            y_shape=None,
            x_dtype=None,
            y_dtype=None,
            num_samples=None,
            fragment_length=None,
            label_count=None,
            memory_bytes=None,
            errors=[f"Could not read NPZ file: {exc}"],
        )
