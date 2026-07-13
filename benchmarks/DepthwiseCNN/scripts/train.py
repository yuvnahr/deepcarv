"""
benchmarks/DepthwiseCNN/scripts/train.py
-----------------------------------------
Training entry-point for the DepthwiseCNN benchmark.

This script is intentionally thin — it reads configuration, wires together
the shared DeepCarv framework components, and hands off to the generic
:class:`~src.training.trainer.Trainer`.  It contains **no** custom training
loop of its own.

Usage
-----
From the repository root (so that `src` and `benchmarks` are importable)::

    # Default (512-byte, DSC variant):
    python -m benchmarks.DepthwiseCNN.scripts.train \\
        --config benchmarks/DepthwiseCNN/configs/benchmark.yaml

    # Override fragment size at the command line without editing the YAML:
    python -m benchmarks.DepthwiseCNN.scripts.train \\
        --config benchmarks/DepthwiseCNN/configs/benchmark.yaml \\
        --fragment-size 512 \\
        --variant dsc-se

    # Resume from a previous checkpoint:
    python -m benchmarks.DepthwiseCNN.scripts.train \\
        --config benchmarks/DepthwiseCNN/configs/benchmark.yaml \\
        --resume outputs/DepthwiseCNN/checkpoint_last.pt

Config keys read
----------------
    dataset.root_dir        — FFT-75 data root
    dataset.fragment_size   — 512 or 4096
    dataset.cache           — load NPZ into RAM (bool)
    model.name              — registry key ("depthwisecnn")
    model.kwargs            — forwarded to build_model (variant, dropout_p, …)
    training.*              — forwarded to TrainerConfig.from_dict()
    paths.run_outputs       — where checkpoints + curves are written
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from src.data.dataset import FragmentDataset, build_dataloader
from src.models.registry import build_model
from src.training.trainer import Trainer, TrainerConfig
from src.utils.logging import RunLogger
from src.utils.paths import FFT75_DATA_DIR, LOGS_DIR, OUTPUTS_DIR, ensure_dirs
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DepthwiseCNN on FFT-75.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to benchmark.yaml (or any compatible YAML config).",
    )
    # Optional CLI overrides — these shadow the YAML values.
    p.add_argument(
        "--fragment-size",
        type=int,
        default=None,
        dest="fragment_size",
        choices=[512, 4096],
        help="Override dataset.fragment_size from the config.",
    )
    p.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=["dsc", "dsc-se", "m-dsc"],
        help="Override model.kwargs.variant from the config.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs from the config.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        dest="batch_size",
        help="Override training.batch_size from the config.",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        dest="run_name",
        help=(
            "Human-readable run identifier used in log directory names.  "
            "Defaults to depthwisecnn_<variant>_<fragment_size>b."
        ),
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Config loading and override application
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"[train] Config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        sys.exit(f"[train] Config must be a YAML mapping: {path}")
    return cfg


def _apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply CLI overrides onto the loaded config dict (in place)."""
    if args.fragment_size is not None:
        cfg.setdefault("dataset", {})["fragment_size"] = args.fragment_size
    if args.variant is not None:
        cfg.setdefault("model", {}).setdefault("kwargs", {})["variant"] = args.variant
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg.setdefault("training", {})["batch_size"] = args.batch_size
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = _load_config(args.config)
    cfg = _apply_cli_overrides(cfg, args)

    # --- Extract sections (never mutate the originals) ---
    ds_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    path_cfg = cfg.get("paths", {})

    # --- Reproducibility ---
    seed: int = train_cfg.get("seed", 42)
    set_seed(seed)

    # --- Resolve paths ---
    data_dir = Path(ds_cfg.get("root_dir", str(FFT75_DATA_DIR)))
    fragment_size: int = int(ds_cfg.get("fragment_size", 4096))
    cache: bool = bool(ds_cfg.get("cache", True))

    run_outputs = Path(path_cfg.get("run_outputs", str(OUTPUTS_DIR / "DepthwiseCNN")))
    ensure_dirs(run_outputs, LOGS_DIR)

    # --- Run identifier ---
    variant = model_cfg.get("kwargs", {}).get("variant", "dsc")
    run_name = args.run_name or f"depthwisecnn_{variant}_{fragment_size}b"

    # --- Logging ---
    with RunLogger(run_name=run_name, base_dir=LOGS_DIR, config_path=args.config) as run_logger:
        run_logger.save_config(cfg)
        run_logger.info(
            "=== DepthwiseCNN Training  variant=%s  fragment_size=%d ===",
            variant, fragment_size,
        )

        # --- Datasets ---
        run_logger.info("Loading datasets from %s …", data_dir)
        train_ds = FragmentDataset(
            root_dir=data_dir,
            split="train",
            fragment_size=fragment_size,
            cache=cache,
        )
        val_ds = FragmentDataset(
            root_dir=data_dir,
            split="val",
            fragment_size=fragment_size,
            cache=cache,
        )
        run_logger.info("Train: %s", train_ds)
        run_logger.info("Val  : %s", val_ds)

        batch_size: int = int(train_cfg.get("batch_size", 256))
        train_loader = build_dataloader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = build_dataloader(
            val_ds, batch_size=batch_size, shuffle=False, drop_last=False
        )

        # --- Model (via registry — no direct adapter import needed) ---
        model = build_model(
            model_cfg["name"],
            num_classes=train_ds.num_classes,
            **model_cfg.get("kwargs", {}),
        )
        run_logger.info(
            "Model: %s | classes=%d | params=%s",
            model.name,
            train_ds.num_classes,
            f"{model.num_parameters():,}",
        )

        # --- Trainer ---
        trainer_config = TrainerConfig.from_dict(train_cfg)
        trainer = Trainer(model, trainer_config, run_outputs)

        # --- Optional resume ---
        start_epoch = 1
        if args.resume is not None:
            start_epoch = trainer.resume_from(args.resume)
            run_logger.info("Resumed from %s (starting at epoch %d)", args.resume, start_epoch)

        # --- Train ---
        history = trainer.fit(train_loader, val_loader, start_epoch=start_epoch)

        # --- Summary ---
        best_acc: float = trainer._best_metric or 0.0
        run_logger.info(
            "=== Training complete  best_val_acc=%.4f  best_epoch=%d ===",
            best_acc,
            trainer.best_epoch,
        )
        run_logger.info("Best checkpoint → %s", trainer.best_ckpt_path)
        run_logger.save_eval_summary({
            "best_val_acc": best_acc,
            "best_epoch": trainer.best_epoch,
            "total_epochs": len(history.val_acc),
            "variant": variant,
            "fragment_size": fragment_size,
            "num_classes": train_ds.num_classes,
            "param_count": model.num_parameters(),
        })


if __name__ == "__main__":
    main()
