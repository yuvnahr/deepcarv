"""Reproducibility metadata helpers for benchmark runs."""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.dataset_tools.fingerprint import DatasetFingerprint, generate_dataset_fingerprint
from src.utils.environment import collect_environment_info, get_git_commit

try:
    import psutil
except ImportError:  # pragma: no cover - optional runtime dependency.
    psutil = None  # type: ignore[assignment]


@dataclass(frozen=True)
class BenchmarkMetadata:
    """Full reproducibility metadata for a benchmark run."""

    dataset_hash: dict[str, str] | None
    dataset_version: str
    git_hash: str
    torch_version: str
    cuda_version: str | None
    python_version: str
    gpu: str | None
    cpu: str
    ram_gb: float | None
    hostname: str
    os: str
    command_line: list[str]
    runtime_s: float | None = None
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return asdict(self)


def collect_benchmark_metadata(
    dataset_root: Path | str | None = None,
    fragment_size: int | None = None,
    dataset_version: str = "unknown",
    started_at: float | None = None,
) -> BenchmarkMetadata:
    """Collect benchmark_metadata.json fields."""
    fingerprint: DatasetFingerprint | None = None
    if dataset_root is not None and fragment_size is not None:
        try:
            fingerprint = generate_dataset_fingerprint(dataset_root, fragment_size, dataset_version)
        except Exception:
            fingerprint = None
    dataset_hash = None
    if fingerprint is not None:
        dataset_hash = {
            split: split_fp.sha256 for split, split_fp in fingerprint.splits.items()
        }

    cuda_available = torch.cuda.is_available()
    ram_gb = None
    if psutil is not None:
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 3)
    return BenchmarkMetadata(
        dataset_hash=dataset_hash,
        dataset_version=dataset_version,
        git_hash=get_git_commit(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda if cuda_available else None,
        python_version=sys.version.split()[0],
        gpu=torch.cuda.get_device_name(0) if cuda_available else None,
        cpu=platform.processor() or platform.machine() or "unknown",
        ram_gb=ram_gb,
        hostname=socket.gethostname(),
        os=platform.platform(),
        command_line=sys.argv,
        runtime_s=(time.time() - started_at) if started_at is not None else None,
        environment={**collect_environment_info(), "pid": os.getpid()},
    )


def write_benchmark_metadata(
    path: Path | str,
    metadata: BenchmarkMetadata,
) -> Path:
    """Write benchmark_metadata.json."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metadata.to_dict(), indent=2, default=str) + "\n")
    return out
