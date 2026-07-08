"""Shared fixtures for framework smoke tests: builds a tiny synthetic
FFT-75-shaped NPZ dataset on disk so tests never depend on the real,
large dataset being present."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tiny_npz_dataset(tmp_path: Path) -> Path:
    """Create <tmp_path>/FFT-75/512/{train,val,test}.npz with tiny random data."""
    root = tmp_path / "FFT-75"
    frag_dir = root / "512"
    frag_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    num_classes = 5
    fragment_size = 512

    for split, n in (("train", 40), ("val", 20), ("test", 20)):
        X = rng.integers(0, 256, size=(n, fragment_size), dtype=np.uint8)
        y = rng.integers(0, num_classes, size=(n,), dtype=np.int64)
        np.savez(frag_dir / f"{split}.npz", X=X, y=y)

    return root
