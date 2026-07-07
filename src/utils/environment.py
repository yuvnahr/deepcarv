"""
src/utils/environment.py
-------------------------
Captures a reproducibility fingerprint of the current run environment.
Used by the ExperimentManager to write environment.json for every run.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any

import torch

from src.utils.paths import REPO_ROOT


def get_git_commit() -> str:
    """Return the current git commit hash, or 'unknown' if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_git_branch() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_cpu_info() -> str:
    try:
        return platform.processor() or platform.machine() or "unknown"
    except Exception:
        return "unknown"


def collect_environment_info() -> dict[str, Any]:
    """Collect a JSON-serializable snapshot of the run environment."""
    cuda_available = torch.cuda.is_available()
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "cpu_info": get_cpu_info(),
        "git_commit": get_git_commit(),
        "git_branch": get_git_branch(),
    }
