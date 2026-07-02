"""
src/utils/paths.py
------------------
Canonical path resolution for the DeepCarv / ByteRCNN benchmark.

All paths are resolved relative to the repository root.
Nothing is hard-coded to an absolute filesystem location so the repo works
both locally and inside Kaggle /kaggle/working/.
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_repo_root() -> Path:
    """Walk up from this file until we find the repo root."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / ".git").exists() or (current / "configs").exists():
            return current
        current = current.parent
    return Path.cwd()


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
REPO_ROOT: Path = _find_repo_root()

# ---------------------------------------------------------------------------
# Source / config directories (inside the repo)
# ---------------------------------------------------------------------------
SRC_DIR: Path = REPO_ROOT / "src"
CONFIGS_DIR: Path = REPO_ROOT / "configs"
NOTEBOOKS_DIR: Path = REPO_ROOT / "notebooks"
BENCHMARKS_DIR: Path = REPO_ROOT / "benchmarks"

# ---------------------------------------------------------------------------
# Data directories (outside the repo; created at runtime)
# ---------------------------------------------------------------------------
DATA_ROOT: Path = Path(os.environ.get("DEEPCARV_DATA_ROOT", str(REPO_ROOT / "data")))

# FFT-75 NPZ root — the extracted FFT-75/ folder
# Layout: FFT75_DATA_DIR / "512" / "train.npz"  etc.
FFT75_DATA_DIR: Path = DATA_ROOT / "FFT-75"

# ---------------------------------------------------------------------------
# Output directories (created at runtime)
# ---------------------------------------------------------------------------
OUTPUTS_DIR: Path = Path(
    os.environ.get("DEEPCARV_OUTPUTS_DIR", str(REPO_ROOT / "outputs"))
)
CHECKPOINTS_DIR: Path = Path(
    os.environ.get("DEEPCARV_CHECKPOINTS_DIR", str(REPO_ROOT / "checkpoints"))
)
LOGS_DIR: Path = Path(
    os.environ.get("DEEPCARV_LOGS_DIR", str(REPO_ROOT / "logs"))
)

# ---------------------------------------------------------------------------
# Benchmark checkpoint + output paths
# ---------------------------------------------------------------------------
BYTERCNN_RUN_DIR: Path = OUTPUTS_DIR / "bytercnn_fft75"
BYTERCNN_BEST_CKPT: Path = CHECKPOINTS_DIR / "best_bytercnn_fft75.pt"
BYTERCNN_SANITY_CKPT: Path = CHECKPOINTS_DIR / "sanity_bytercnn_fft75.pt"

# ---------------------------------------------------------------------------
# Kaggle-specific overrides
# ---------------------------------------------------------------------------
if os.environ.get("KAGGLE_RUNTIME", "0") == "1":
    _kw = Path("/kaggle/working")
    DATA_ROOT       = _kw / "data"
    FFT75_DATA_DIR  = DATA_ROOT / "FFT-75"
    OUTPUTS_DIR     = _kw / "outputs"
    CHECKPOINTS_DIR = _kw / "checkpoints"
    LOGS_DIR        = _kw / "logs"
    BYTERCNN_RUN_DIR   = OUTPUTS_DIR / "bytercnn_fft75"
    BYTERCNN_BEST_CKPT = CHECKPOINTS_DIR / "best_bytercnn_fft75.pt"
    BYTERCNN_SANITY_CKPT = CHECKPOINTS_DIR / "sanity_bytercnn_fft75.pt"


def ensure_dirs(*dirs: Path) -> None:
    """Create directories (and parents) if they don't already exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def print_paths() -> None:
    """Pretty-print all canonical paths."""
    labels = {
        "REPO_ROOT":           REPO_ROOT,
        "FFT75_DATA_DIR":      FFT75_DATA_DIR,
        "OUTPUTS_DIR":         OUTPUTS_DIR,
        "CHECKPOINTS_DIR":     CHECKPOINTS_DIR,
        "LOGS_DIR":            LOGS_DIR,
        "BYTERCNN_BEST_CKPT":  BYTERCNN_BEST_CKPT,
    }
    print("\n=== DeepCarv Paths ===")
    for name, path in labels.items():
        print(f"  {name:<25} {path}")
    print()
