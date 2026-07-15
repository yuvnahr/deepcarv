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

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.npz_mmap import is_mmappable, mmap_npz_member
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
        mmap: bool = True,
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
        # `mmap` mode maps the arrays straight off disk instead of reading them
        # into RAM. This is what makes full-scale FFT-75 (esp. 4096-byte
        # fragments, ~25 GB train split) trainable on a ~13 GB Kaggle instance:
        # the OS pages in only the batches actually touched, so resident memory
        # stays near-flat regardless of dataset size.
        self._mmapped = False
        X: np.ndarray
        y: np.ndarray

        if mmap and is_mmappable(npz_path):
            X_mm = mmap_npz_member(npz_path, "x")
            y_mm = mmap_npz_member(npz_path, "y")
            if X_mm is not None and y_mm is not None:
                X, y = X_mm, y_mm
                self._mmapped = True

        if not self._mmapped:
            data = np.load(str(npz_path))
            keys = {k.lower(): k for k in data.files}
            if "x" not in keys or "y" not in keys:
                raise ValueError(
                    f"{npz_path.name} must contain x/y arrays, got {data.files}"
                )
            X = data[keys["x"]]
            y = data[keys["y"]]

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

        # ---- Storage mode ------------------------------------------------
        # Priority: mmap (zero RAM) > cache (RAM-resident) > lazy numpy.
        #
        # NOTE: the previous implementation cached X as int16, which DOUBLED
        # memory for no benefit — byte values are 0-255 and fit in uint8. At
        # 4096-byte fragments that upcast alone turned a 25 GB train split into
        # 50 GB (75 GB peak while both copies coexisted), which is the direct
        # cause of the Kaggle RAM overflows. X is now kept in its native uint8
        # and cast to int64 per batch in __getitem__ (negligible cost).
        if self._mmapped:
            # Keep the memmap views; do NOT copy into RAM.
            self._X_np = X
            self._y_np = y
            self._cached = False
        elif cache:
            self._X = torch.from_numpy(np.ascontiguousarray(X))  # native uint8
            self._y = torch.from_numpy(y.astype(np.int64))
            self._cached = True
        else:
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
            # Stored natively as uint8; cast to int64 per item (cheap).
            return self._X[idx].to(torch.int64), self._y[idx]
        # mmap / lazy-numpy path. np.asarray() on a memmap row copies just that
        # row (a few KB), so RAM stays flat no matter how large the file is.
        row = np.asarray(self._X_np[idx], dtype=np.int64)
        return (
            torch.from_numpy(row),
            torch.tensor(int(self._y_np[idx]), dtype=torch.long),
        )

    def __repr__(self) -> str:
        return (
            f"FragmentDataset("
            f"split={self.split!r}, "
            f"n={self._n:,}, "
            f"fragment_size={self.fragment_size}, "
            f"num_classes={self.num_classes}, "
            f"cached={self._cached}, mmapped={self._mmapped})"
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
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = 4,
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
        Defaults to 4 on CUDA (Kaggle T4-friendly), otherwise 0.
    pin_memory : bool
        Auto-disabled if CUDA is not available.
    persistent_workers : bool | None
        Keep worker processes alive between epochs when num_workers > 0.
        Defaults to True for worker-based loading.
    prefetch_factor : int | None
        Number of batches each worker preloads. Only valid when
        num_workers > 0.
    drop_last : bool
        Drop last incomplete batch (useful during training).
    """
    if num_workers is None:
        # Kaggle T4 runs are input-pipeline bound with the old cached=RAM
        # default of 0 workers. Four workers is usually the best trade-off
        # there; keep CPU-only/local runs simple by defaulting to 0.
        num_workers = 4 if torch.cuda.is_available() else 0

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    _pin = pin_memory and torch.cuda.is_available()
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": _pin,
        "drop_last": drop_last,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **loader_kwargs)
