"""
src/utils/paths.py
------------------
Canonical path resolution for the DeepCarv / ByteRCNN benchmark.

All paths are resolved relative to the repository root (the directory that
contains this very file's grandparent — i.e., two levels up from src/utils/).
Nothing is hard-coded to an absolute filesystem location so the repo works
both locally and inside Kaggle /kaggle/working/.
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_repo_root() -> Path:
    """Walk up from this file until we find the repo root.

    The repo root is identified by the presence of a `.git` directory
    or a `configs/` directory (whichever comes first).
    """
    current = Path(__file__).resolve().parent
    for _ in range(10):  # guard against infinite climb
        if (current / ".git").exists() or (current / "configs").exists():
            return current
        current = current.parent
    # Fallback: assume cwd
    return Path.cwd()


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
REPO_ROOT: Path = _find_repo_root()

# ---------------------------------------------------------------------------
# Source directories (inside the repo)
# ---------------------------------------------------------------------------
SRC_DIR: Path = REPO_ROOT / "src"
CONFIGS_DIR: Path = REPO_ROOT / "configs"
NOTEBOOKS_DIR: Path = REPO_ROOT / "notebooks"
BENCHMARKS_DIR: Path = REPO_ROOT / "benchmarks"

# ---------------------------------------------------------------------------
# Data directories (outside the repo; created at runtime)
# ---------------------------------------------------------------------------
# These can be overridden by setting environment variables, which the Kaggle
# notebook does automatically.
DATA_ROOT: Path = Path(os.environ.get("DEEPCARV_DATA_ROOT", str(REPO_ROOT / "data")))
DATA_RAW_DIR: Path = DATA_ROOT / "raw"
DATA_SPLITS_DIR: Path = DATA_ROOT / "splits"

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
# Frozen benchmark split paths
# ---------------------------------------------------------------------------
FFT75_SPLITS_DIR: Path = DATA_SPLITS_DIR / "fft75_s1_512"
FFT75_TRAIN_CSV: Path = FFT75_SPLITS_DIR / "train.csv"
FFT75_VAL_CSV: Path = FFT75_SPLITS_DIR / "val.csv"
FFT75_TEST_CSV: Path = FFT75_SPLITS_DIR / "test.csv"
FFT75_CLASS_MAP: Path = FFT75_SPLITS_DIR / "class_map.json"
FFT75_MANIFEST: Path = FFT75_SPLITS_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# Frozen benchmark checkpoint + output paths
# ---------------------------------------------------------------------------
BYTERCNN_RUN_DIR: Path = OUTPUTS_DIR / "bytercnn_fft75"
BYTERCNN_BEST_CKPT: Path = CHECKPOINTS_DIR / "best_bytercnn_fft75.pt"
BYTERCNN_SANITY_CKPT: Path = CHECKPOINTS_DIR / "sanity_bytercnn_fft75.pt"

# ---------------------------------------------------------------------------
# Kaggle-specific overrides
# ---------------------------------------------------------------------------
# The Kaggle notebook sets KAGGLE_RUNTIME=1 in the environment before calling
# any Python scripts.  Paths are re-rooted to /kaggle/working/ in that case.
if os.environ.get("KAGGLE_RUNTIME", "0") == "1":
    _kaggle_working = Path("/kaggle/working")
    DATA_ROOT = _kaggle_working / "data"
    DATA_RAW_DIR = DATA_ROOT / "raw"
    DATA_SPLITS_DIR = DATA_ROOT / "splits"
    FFT75_SPLITS_DIR = DATA_SPLITS_DIR / "fft75_s1_512"
    FFT75_TRAIN_CSV = FFT75_SPLITS_DIR / "train.csv"
    FFT75_VAL_CSV = FFT75_SPLITS_DIR / "val.csv"
    FFT75_TEST_CSV = FFT75_SPLITS_DIR / "test.csv"
    FFT75_CLASS_MAP = FFT75_SPLITS_DIR / "class_map.json"
    FFT75_MANIFEST = FFT75_SPLITS_DIR / "manifest.json"
    OUTPUTS_DIR = _kaggle_working / "outputs"
    CHECKPOINTS_DIR = _kaggle_working / "checkpoints"
    LOGS_DIR = _kaggle_working / "logs"
    BYTERCNN_RUN_DIR = OUTPUTS_DIR / "bytercnn_fft75"
    BYTERCNN_BEST_CKPT = CHECKPOINTS_DIR / "best_bytercnn_fft75.pt"
    BYTERCNN_SANITY_CKPT = CHECKPOINTS_DIR / "sanity_bytercnn_fft75.pt"


def ensure_dirs(*dirs: Path) -> None:
    """Create directories (and parents) if they don't already exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def print_paths() -> None:
    """Pretty-print all canonical paths (useful for debugging)."""
    labels = {
        "REPO_ROOT": REPO_ROOT,
        "DATA_RAW_DIR": DATA_RAW_DIR,
        "DATA_SPLITS_DIR": DATA_SPLITS_DIR,
        "FFT75_SPLITS_DIR": FFT75_SPLITS_DIR,
        "FFT75_TRAIN_CSV": FFT75_TRAIN_CSV,
        "FFT75_VAL_CSV": FFT75_VAL_CSV,
        "FFT75_TEST_CSV": FFT75_TEST_CSV,
        "OUTPUTS_DIR": OUTPUTS_DIR,
        "CHECKPOINTS_DIR": CHECKPOINTS_DIR,
        "LOGS_DIR": LOGS_DIR,
        "BYTERCNN_BEST_CKPT": BYTERCNN_BEST_CKPT,
    }
    print("\n=== DeepCarv Paths ===")
    for name, path in labels.items():
        print(f"  {name:<25} {path}")
    print()
