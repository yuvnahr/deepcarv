"""
src/data/dataset.py
--------------------
PyTorch Dataset for the FFT-75 benchmark — NPZ edition.

Loads the official pre-split .npz files directly.  No CSV generation,
no split logic.  The benchmark splits are authoritative.

Expected layout:
    {root_dir}/
    └── {fragment_size}/
        ├── train.npz
        ├── val.npz
        └── test.npz

Each .npz contains:
    X : np.ndarray  shape [N, fragment_size]  — byte values
    y : np.ndarray  shape [N]                 — integer class labels

Classes
-------
    FragmentDataset  — main dataset
    build_dataloader — convenience DataLoader factory

Usage
-----
    from src.data.dataset import FragmentDataset, build_dataloader

    train_ds = FragmentDataset(
        root_dir      = "data/FFT-75",
        split         = "train",
        fragment_size = 512,
        cache         = True,
    )
    loader = build_dataloader(train_ds, batch_size=256, shuffle=True)

    fragment, label = train_ds[0]
    # fragment : torch.long  shape [512]
    # label    : torch.long  scalar
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_SPLITS: frozenset[str] = frozenset({"train", "val", "test"})


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FragmentDataset(Dataset):
    """PyTorch Dataset that reads FFT-75 .npz split files.

    Parameters
    ----------
    root_dir : Path | str
        FFT-75 root directory (e.g. ``data/FFT-75`` or
        ``/kaggle/working/data/FFT-75``).
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    fragment_size : int
        Fragment length in bytes (512 or 4096).  Selects the subdirectory.
    cache : bool
        If True (default), load the entire split into RAM on construction.
        Recommended — FFT-75 NPZ files fit comfortably in memory.
    tiny_subset : bool | int
        If True, keep only the first 2 000 samples (sanity runs).
        If a positive int, keep that many samples instead.
    """

    def __init__(
        self,
        root_dir: Path | str,
        split: str,
        fragment_size: int = 512,
        cache: bool = True,
        tiny_subset: bool | int = False,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(
                f"split must be one of {sorted(VALID_SPLITS)}, got '{split}'"
            )

        self.root_dir = Path(root_dir)
        self.split = split
        self.fragment_size = fragment_size

        npz_path = self.root_dir / str(fragment_size) / f"{split}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"NPZ file not found: {npz_path}\n"
                f"  Check that the FFT-75 dataset was downloaded and extracted\n"
                f"  to: {self.root_dir}"
            )

        # ---- Load arrays -------------------------------------------------
        data = np.load(str(npz_path))
        if set(data.files) != {"x", "y"}:
            raise ValueError(
                f"{npz_path.name} must contain exactly keys ['X', 'y'], "
                f"got {data.files}"
            )

        X: np.ndarray = data["x"]
        y: np.ndarray = data["y"]

        if X.ndim != 2 or X.shape[1] != fragment_size:
            raise ValueError(
                f"X has wrong shape {X.shape}; "
                f"expected (N, {fragment_size})"
            )
        if len(X) != len(y):
            raise ValueError(f"len(X)={len(X)} != len(y)={len(y)}")

        # ---- tiny_subset -------------------------------------------------
        if tiny_subset is True:
            n = 2_000
        elif isinstance(tiny_subset, int) and tiny_subset > 0:
            n = tiny_subset
        else:
            n = len(X)

        n = min(n, len(X))
        X = X[:n]
        y = y[:n]

        # ---- Class info --------------------------------------------------
        self.num_classes: int = int(y.max()) + 1
        unique_labels = np.unique(y)
        self.class_labels: list[int] = unique_labels.tolist()

        # ---- Optional caching (always True by default) -------------------
        if cache:
            # Store X as int16 (byte values 0-255 fit comfortably — 4x less RAM
            # than int64). Cast to int64 lazily in __getitem__ per batch.
            self._X = torch.from_numpy(X.astype(np.int16))   # [N, L]  ~2 bytes/elem
            self._y = torch.from_numpy(y.astype(np.int64))   # [N]
            self._cached = True
        else:
            # Keep as numpy, convert per-item
            self._X_np = X
            self._y_np = y
            self._cached = False

        self._n = n

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cached:
            # Cast X from int16 → int64 here (per batch, negligible overhead)
            return self._X[idx].to(torch.int64), self._y[idx]
        return (
            torch.from_numpy(self._X_np[idx].astype(np.int64)),
            torch.tensor(int(self._y_np[idx]), dtype=torch.long),
        )

    def __repr__(self) -> str:
        return (
            f"FragmentDataset("
            f"split={self.split!r}, "
            f"n={self._n:,}, "
            f"fragment_size={self.fragment_size}, "
            f"num_classes={self.num_classes}, "
            f"cached={self._cached})"
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def build_dataloader(
    dataset: FragmentDataset,
    batch_size: int = 256,
    shuffle: bool = False,
    num_workers: int | None = None,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Return a DataLoader with sane defaults.

    Parameters
    ----------
    dataset : FragmentDataset
    batch_size : int
    shuffle : bool
        True for training, False for val/test.
    num_workers : int | None
        Defaults to min(4, cpu_count // 2).
        If the dataset is fully cached in RAM (cache=True), num_workers=0
        is often faster because data is already a tensor.
    pin_memory : bool
        Auto-disabled if CUDA is not available.
    drop_last : bool
        Drop last incomplete batch (useful during training).
    """
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        # With full RAM cache, inter-process communication overhead
        # outweighs any loading benefit.
        num_workers = 0 if dataset._cached else min(4, max(0, cpu_count // 2))

    _pin = pin_memory and torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=_pin,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )
