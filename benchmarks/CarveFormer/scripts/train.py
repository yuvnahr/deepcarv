"""
benchmarks/CarveFormer/scripts/train.py
=======================================
CarveFormer training entrypoint.

This is a thin wrapper around the shared DeepCarv framework: it loads the
benchmark config, builds datasets via the framework ``FragmentDataset``, builds
the model via the framework registry (key ``"carveformer"``), and trains with
the shared ``Trainer``. It does NOT duplicate any training logic and does NOT
create a second training system.

Usage
-----
    python -m benchmarks.CarveFormer.scripts.train \
        --config benchmarks/CarveFormer/configs/benchmark.yaml

    # Override anything from the CLI:
    python -m benchmarks.CarveFormer.scripts.train \
        --config benchmarks/CarveFormer/configs/benchmark.yaml \
        --fragment-size 4096 --epochs 2 --no-pretrained

Note on effective batch size
----------------------------
The paper uses an effective batch size of 1024. The shared ``Trainer`` does
not implement gradient accumulation, so ``batch_size`` here is the *real*
batch size the hardware must hold. To reach the paper's effective 1024 you
need a large-memory GPU (or a future accumulation feature in the shared
trainer). This is a documented deviation, not a silent simplification.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from src.data.dataset import FragmentDataset, build_dataloader
from src.evaluation.evaluator import Evaluator
from src.models.registry import build_model
from src.training.trainer import Trainer, TrainerConfig
from src.utils.paths import REPO_ROOT, ensure_dirs
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the benchmark YAML config."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _resolve(root: Path, value: str) -> Path:
    """Resolve a possibly-relative path against the repo root."""
    p = Path(value)
    return p if p.is_absolute() else (root / p)


def run_training(config: dict[str, Any], overrides: dict[str, Any] | None = None) -> Path:
    """Train CarveFormer end-to-end using the shared framework.

    Returns the run output directory containing the standardized outputs.
    """
    overrides = overrides or {}
    model_cfg = dict(config.get("model", {}))
    ds_cfg = dict(config.get("dataset", {}))
    train_cfg = dict(config.get("training", {}))
    path_cfg = dict(config.get("paths", {}))

    # CLI overrides
    fragment_size = int(overrides.get("fragment_size") or ds_cfg.get("fragment_size") or 512)
    if "epochs" in overrides:
        train_cfg["epochs"] = int(overrides["epochs"])
    if "pretrained" in overrides:
        model_cfg["pretrained"] = bool(overrides["pretrained"])
    data_dir = Path(str(overrides.get("data_dir") or ds_cfg.get("root_dir") or "data/FFT-75"))
    data_dir = _resolve(REPO_ROOT, str(data_dir))

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    run_outputs = _resolve(REPO_ROOT, str(path_cfg.get("run_outputs", "benchmarks/CarveFormer/outputs")))
    ensure_dirs(run_outputs)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    logger.info("=== CarveFormer training | fragment_size=%d | data=%s ===", fragment_size, data_dir)

    # ---- Datasets (shared framework) -----------------------------------
    cache = bool(ds_cfg.get("cache", True))
    train_ds = FragmentDataset(root_dir=data_dir, split="train", fragment_size=fragment_size, cache=cache)
    val_ds = FragmentDataset(root_dir=data_dir, split="val", fragment_size=fragment_size, cache=cache)
    test_ds = FragmentDataset(root_dir=data_dir, split="test", fragment_size=fragment_size, cache=cache)

    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 0))
    train_loader = build_dataloader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = build_dataloader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = build_dataloader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_classes = train_ds.num_classes
    logger.info("Dataset: num_classes=%d train=%d val=%d test=%d",
                num_classes, len(train_ds), len(val_ds), len(test_ds))

    # ---- Model (shared registry) ---------------------------------------
    model_kwargs = {k: v for k, v in model_cfg.items() if k not in ("name",)}
    model = build_model("carveformer", num_classes=num_classes, fragment_size=fragment_size, **model_kwargs)
    logger.info("CarveFormer built: %d trainable params", model.num_parameters())

    # ---- Train (shared Trainer) ----------------------------------------
    trainer_config = TrainerConfig.from_dict(train_cfg)
    trainer = Trainer(model, trainer_config, run_outputs)
    trainer.fit(train_loader, val_loader)

    # ---- Evaluate best checkpoint (shared Evaluator) -------------------
    from src.training.checkpointing import load_checkpoint

    load_checkpoint(trainer.best_ckpt_path, model, map_location=trainer.device)
    evaluator = Evaluator(model, device=trainer_config.device)
    evaluator.evaluate_and_save(test_loader, run_outputs, run_name=f"carveformer_fft75_{fragment_size}")

    logger.info("CarveFormer run complete: %s", run_outputs)
    return run_outputs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CarveFormer via the DeepCarv framework.")
    p.add_argument("--config", type=str, default="benchmarks/CarveFormer/configs/benchmark.yaml")
    p.add_argument("--fragment-size", type=int, default=None, help="512 or 4096 (overrides config).")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet1k init (offline).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = load_config(args.config)
    overrides: dict[str, Any] = {}
    if args.fragment_size is not None:
        overrides["fragment_size"] = args.fragment_size
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.data_dir is not None:
        overrides["data_dir"] = args.data_dir
    if args.no_pretrained:
        overrides["pretrained"] = False
    run_training(config, overrides)


if __name__ == "__main__":
    main()
