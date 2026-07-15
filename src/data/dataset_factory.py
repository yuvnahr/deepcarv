"""
src/data/dataset_factory.py
------------------------------
Dataset factory for the DeepCarv benchmark framework.

Provides:
    build_datasets  — construct train/val/test FragmentDataset objects from
                      a dataset config dict.
    build_dataloaders — wrap the datasets in DataLoader objects, using
                        training and evaluation configs for batch sizes /
                        worker counts.

This module is the ONLY supported way for the generic runner to instantiate
datasets.  Dataset code (FragmentDataset) lives in src/data/dataset.py;
no model-specific dataset handling belongs here.

Config keys expected in `dataset_config`
-----------------------------------------
    root_dir      : str | Path  — FFT-75 root directory
    fragment_size : int         — 512 or 4096
    version       : str         — dataset version tag (optional)
    tiny_subset   : bool | int  — restrict to N samples (for sanity runs)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from src.data.dataset import FragmentDataset, build_dataloader
from src.utils.paths import FFT75_DATA_DIR


def build_datasets(
    dataset_config: dict[str, Any],
    tiny_subset: bool | int = False,
) -> dict[str, FragmentDataset]:
    """Construct train/val/test datasets from a dataset config dict.

    Parameters
    ----------
    dataset_config : dict
        Must contain 'fragment_size'.  'root_dir' defaults to FFT75_DATA_DIR.
    tiny_subset : bool | int
        If True, keep 2 000 samples per split (sanity mode).
        If a positive int, keep that many samples.

    Returns
    -------
    dict with keys "train", "val", "test".
    """
    root_dir = Path(dataset_config.get("root_dir", FFT75_DATA_DIR))
    if not root_dir.is_absolute():
        # Resolve relative paths against the repo root via FFT75_DATA_DIR's parent
        from src.utils.paths import REPO_ROOT
        root_dir = REPO_ROOT / root_dir

    fragment_size = int(dataset_config["fragment_size"])
    tiny = dataset_config.get("tiny_subset", tiny_subset)
    cache = dataset_config.get("cache", True)
    mmap = dataset_config.get("mmap", False)

    datasets: dict[str, FragmentDataset] = {}
    for split in ("train", "val", "test"):
        datasets[split] = FragmentDataset(
            root_dir=root_dir,
            split=split,
            fragment_size=fragment_size,
            cache=cache,
            mmap=mmap,
            tiny_subset=tiny,
        )
    return datasets


def build_dataloaders(
    datasets: dict[str, FragmentDataset],
    training_config: dict[str, Any] | None = None,
    evaluation_config: dict[str, Any] | None = None,
    # Legacy aliases accepted for backwards compatibility with pre-existing tests
    training_cfg: dict[str, Any] | None = None,
    eval_cfg: dict[str, Any] | None = None,
) -> dict[str, DataLoader]:
    """Wrap datasets in DataLoaders using training/evaluation config values.

    Parameters
    ----------
    datasets : dict
        Mapping of split name -> FragmentDataset.
    training_config : dict
        Training config; used for batch_size, num_workers for train split.
        Alias: training_cfg (accepted for backwards compatibility).
    evaluation_config : dict
        Evaluation config; used for batch_size for val/test splits.
        Alias: eval_cfg (accepted for backwards compatibility).

    Returns
    -------
    dict with keys "train", "val", "test".
    """
    # Accept legacy kwarg names
    _train_cfg: dict[str, Any] = training_config or training_cfg or {}
    _eval_cfg: dict[str, Any] = evaluation_config or eval_cfg or {}

    train_bs = int(_train_cfg.get("batch_size", 256))
    eval_bs = int(_eval_cfg.get("batch_size", train_bs))
    num_workers = int(_train_cfg.get("num_workers", 0))

    loaders: dict[str, DataLoader] = {}

    loaders["train"] = build_dataloader(
        datasets["train"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    for split in ("val", "test"):
        loaders[split] = build_dataloader(
            datasets[split],
            batch_size=eval_bs,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
    return loaders
