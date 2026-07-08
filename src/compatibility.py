"""Benchmark compatibility API for future DeepCarv model adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class CompatibilityReport:
    """Model compatibility and resource estimate for one benchmark setup."""

    model_name: str
    fragment_size_supported: bool
    num_classes_supported: bool
    dataset_supported: bool
    dtype_supported: bool
    gpu_supported: bool
    estimated_memory_mb: float | None = None
    estimated_flops: float | None = None
    warnings: list[str] | None = None

    @property
    def ok(self) -> bool:
        """Return whether all hard compatibility checks passed."""
        return all(
            [
                self.fragment_size_supported,
                self.num_classes_supported,
                self.dataset_supported,
                self.dtype_supported,
                self.gpu_supported,
            ]
        )


class BenchmarkCompatible(Protocol):
    """Protocol future model wrappers can implement for benchmark checks."""

    def supports_fragment_size(self, fragment_size: int) -> bool:
        """Return whether this model supports a fragment length."""

    def supports_num_classes(self, num_classes: int) -> bool:
        """Return whether this model supports a number of output classes."""

    def supports_dataset(self, dataset_name: str) -> bool:
        """Return whether this model supports a dataset."""

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        """Return whether this model supports an input dtype."""

    def supports_gpu(self) -> bool:
        """Return whether this model can run on CUDA."""

    def estimated_memory(self, batch_size: int, fragment_size: int) -> float | None:
        """Return estimated memory in MB, if known."""

    def estimated_flops(self, fragment_size: int) -> float | None:
        """Return estimated FLOPs per sample, if known."""


def _call_bool(model: object, method: str, default: bool, *args: object) -> bool:
    fn = getattr(model, method, None)
    if fn is None:
        return default
    try:
        return bool(fn(*args))
    except Exception:
        return False


def build_compatibility_report(
    model: object,
    model_name: str,
    dataset_name: str,
    fragment_size: int,
    num_classes: int,
    dtype: torch.dtype = torch.uint8,
    batch_size: int = 1,
) -> CompatibilityReport:
    """Evaluate a model object against the standard compatibility API."""
    warnings: list[str] = []
    if not hasattr(model, "supports_fragment_size"):
        warnings.append("Model does not expose supports_fragment_size(); assuming compatible.")
    if not hasattr(model, "supports_num_classes"):
        warnings.append("Model does not expose supports_num_classes(); assuming compatible.")

    memory_fn = getattr(model, "estimated_memory", None)
    flops_fn = getattr(model, "estimated_flops", None)
    memory = memory_fn(batch_size, fragment_size) if memory_fn else None
    flops = flops_fn(fragment_size) if flops_fn else None

    return CompatibilityReport(
        model_name=model_name,
        fragment_size_supported=_call_bool(model, "supports_fragment_size", True, fragment_size),
        num_classes_supported=_call_bool(model, "supports_num_classes", True, num_classes),
        dataset_supported=_call_bool(model, "supports_dataset", True, dataset_name),
        dtype_supported=_call_bool(model, "supports_dtype", True, dtype),
        gpu_supported=_call_bool(model, "supports_gpu", True),
        estimated_memory_mb=memory,
        estimated_flops=flops,
        warnings=warnings,
    )
