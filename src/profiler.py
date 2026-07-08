"""Lightweight performance profiler for benchmark runs."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch

try:
    import psutil
except ImportError:  # pragma: no cover - optional runtime dependency.
    psutil = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerformanceProfile:
    """Performance metrics captured for a run or batch loop."""

    training_throughput_samples_s: float | None = None
    inference_throughput_samples_s: float | None = None
    gpu_utilization: float | None = None
    peak_gpu_memory_mb: float | None = None
    cpu_ram_mb: float | None = None
    samples_per_sec: float | None = None
    time_per_batch_s: float | None = None
    time_per_epoch_s: float | None = None
    flops_estimate: float | None = None
    parameter_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


class PerformanceProfiler:
    """Collects simple timing, memory, and throughput metrics."""

    def __init__(self, device: torch.device | str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._start: float | None = None

    def start(self) -> None:
        """Start a timing window and reset CUDA peak memory when available."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._start = time.perf_counter()

    def stop(
        self,
        samples: int | None = None,
        batches: int | None = None,
        parameter_count: int | None = None,
        flops_estimate: float | None = None,
    ) -> PerformanceProfile:
        """Stop timing and return a performance profile."""
        if self._start is None:
            raise RuntimeError("PerformanceProfiler.stop() called before start().")
        elapsed = time.perf_counter() - self._start
        samples_per_sec = samples / elapsed if samples and elapsed > 0 else None
        time_per_batch = elapsed / batches if batches and batches > 0 else None
        peak_gpu = None
        if self.device.type == "cuda":
            peak_gpu = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        cpu_ram = None
        if psutil is not None:
            cpu_ram = psutil.Process().memory_info().rss / (1024**2)
        return PerformanceProfile(
            samples_per_sec=samples_per_sec,
            time_per_batch_s=time_per_batch,
            time_per_epoch_s=elapsed,
            peak_gpu_memory_mb=peak_gpu,
            cpu_ram_mb=cpu_ram,
            flops_estimate=flops_estimate,
            parameter_count=parameter_count,
        )


def count_parameters(model: torch.nn.Module) -> int:
    """Return trainable parameter count."""
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def save_performance_profile(profile: PerformanceProfile, path: Path | str) -> Path:
    """Write performance.json."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile.to_dict(), indent=2) + "\n")
    logger.info("Performance profile written to %s", out)
    return out
