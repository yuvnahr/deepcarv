"""
src/data/dataset_factory.py
============================
Config-driven construction of train/val/test datasets and dataloaders.

This module exists so the trainer, evaluator, and unified runner never
construct :class:`~src.data.dataset.FragmentDataset` directly — dataset-layout
knowledge stays isolated here. Adding a new dataset format later means adding a
branch in this file, not touching the trainer.

Memory note
-----------
``mmap`` defaults to ``True``. FFT-75 at 4096-byte fragments has a ~25 GB
training split, which cannot be read into a ~13 GB Kaggle instance. Memory-
mapping streams batches straight off disk and keeps resident memory near-flat,
which is what makes full-scale training possible. See ``src/data/npz_mmap.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from src.data.dataset import FragmentDataset, build_dataloader


def build_datasets(dataset_cfg: dict[str, Any]) -> dict[str, FragmentDataset]:
    """Build ``{"train": ds, "val": ds, "test": ds}`` from a dataset config.

    Recognised keys
    ---------------
    root_dir:
        Root FFT-75 directory (``{root_dir}/{fragment_size}/{split}.npz``).
    fragment_size:
        512 or 4096.
    cache:
        Hold the split in RAM (ignored when ``mmap`` succeeds).
    mmap:
        Memory-map the NPZ instead of loading it into RAM (default ``True``).
    tiny_subset:
        Optional cap on sample count, for smoke tests only.
    """
    fmt = dataset_cfg.get("format", "npz")
    if fmt != "npz":
        raise ValueError(
            f"Unsupported dataset format '{fmt}'. Only 'npz' is implemented. "
            "Add a new branch here (not in the trainer) to support new formats."
        )

    root_dir = Path(dataset_cfg.get("root_dir", "data/FFT-75"))
    fragment_size = int(dataset_cfg.get("fragment_size", 512))
    cache = bool(dataset_cfg.get("cache", True))
    mmap = bool(dataset_cfg.get("mmap", True))
    tiny_subset: bool | int = dataset_cfg.get("tiny_subset") or False

    datasets: dict[str, FragmentDataset] = {}
    for split in ("train", "val", "test"):
        datasets[split] = FragmentDataset(
            root_dir=root_dir,
            split=split,
            fragment_size=fragment_size,
            cache=cache,
            mmap=mmap,
            tiny_subset=tiny_subset,
        )
    return datasets


def build_dataloaders(
    datasets: dict[str, FragmentDataset],
    training_cfg: dict[str, Any],
    eval_cfg: dict[str, Any] | None = None,
) -> dict[str, DataLoader]:
    """Build ``{"train": loader, "val": loader, "test": loader}``."""
    eval_cfg = eval_cfg or {}
    train_bs = int(training_cfg.get("batch_size", 256))
    eval_bs = int(eval_cfg.get("batch_size", train_bs))
    num_workers = int(training_cfg.get("num_workers", 0))

    return {
        "train": build_dataloader(
            datasets["train"], batch_size=train_bs, shuffle=True,
            drop_last=True, num_workers=num_workers,
        ),
        "val": build_dataloader(
            datasets["val"], batch_size=eval_bs, shuffle=False,
            num_workers=num_workers,
        ),
        "test": build_dataloader(
            datasets["test"], batch_size=eval_bs, shuffle=False,
            num_workers=num_workers,
        ),
    }
