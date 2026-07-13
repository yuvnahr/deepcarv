"""
benchmarks/DepthwiseCNN/scripts/train.py
------------------------------------------
Training entry point for the DepthwiseCNN benchmark on FFT-75.

This script wires the DepthwiseCNN model into the DeepCarv generic training
framework (Trainer + ExperimentManager) and produces all standardized outputs.

All three paper variants (DSC, DSC-SE, M-DSC) are supported via --variant.

Usage
-----
    # Full benchmark run — 512-byte fragments, DSC variant:
    python benchmarks/DepthwiseCNN/scripts/train.py \\
        --data_dir data/FFT-75 \\
        --fragment_size 512 \\
        --variant dsc

    # From a composed experiment config (recommended):
    python benchmarks/DepthwiseCNN/scripts/train.py \\
        --config configs/experiments/depthwisecnn_fft75_512.yaml

    # Sanity run (2 epochs, 2000 samples):
    python benchmarks/DepthwiseCNN/scripts/train.py \\
        --data_dir data/FFT-75 --sanity

    # Via the generic DeepCarv runner:
    python -m src.core.runner --experiment depthwisecnn_fft75_512

Constraints
-----------
- Does NOT regenerate FFT-75 splits.
- Does NOT hardcode dataset paths.
- Loads pre-split .npz files via FragmentDataset.
- Delegates model construction to the registry.
- Delegates training to the generic Trainer.
- Delegates outputs/checkpointing to ExperimentManager + Trainer.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import yaml

import torch

from src.core.experiment_manager import ExperimentManager
from src.data.dataset import FragmentDataset, build_dataloader
from src.evaluation.evaluator import Evaluator
from src.models.registry import build_model
from src.training.checkpointing import load_checkpoint
from src.training.trainer import Trainer, TrainerConfig
from src.utils.config import ExperimentConfig, load_experiment_config
from src.utils.paths import FFT75_DATA_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR, ensure_dirs
from src.utils.seed import set_seed
from src.visualization.confusion import plot_confusion_matrix
from src.visualization.plots import plot_accuracy_curve, plot_loss_curve, plot_lr_curve

logger = logging.getLogger("depthwisecnn_train")

# ---------------------------------------------------------------------------
# Paper-derived training defaults (§III)
# Assumptions documented in configs/training/depthwisecnn.yaml
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "epochs": 50,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "optimizer": "adamw",
    "scheduler": "reduce_on_plateau",
    "scheduler_patience": 3,
    "grad_clip": 1.0,
    "patience": 10,
    "fragment_size": 512,
    "variant": "dsc",
    "embed_dim": 64,
    "channels": [64, 128, 256, 256],
    "kernel_size": 3,
    "se_reduction": 16,
    "p_dropout": 0.5,
    "amp": True,
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="DepthwiseCNN benchmark training on FFT-75."
    )
    p.add_argument("--config", type=Path, default=None,
                   help="Experiment YAML config (e.g. configs/experiments/depthwisecnn_fft75_512.yaml)")
    p.add_argument("--data_dir", type=Path, default=None,
                   help="FFT-75 root directory (contains 512/ and 4096/ subdirs).")
    p.add_argument("--fragment_size", type=int, default=None, choices=[512, 4096],
                   help="Fragment size in bytes (512 or 4096).")
    p.add_argument("--variant", type=str, default=None,
                   choices=["dsc", "dsc_se", "m_dsc"],
                   help="DepthwiseCNN variant: dsc | dsc_se | m_dsc")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run_dir", type=Path, default=None,
                   help="Output directory for this run.")
    p.add_argument("--resume", type=Path, default=None,
                   help="Resume training from a checkpoint file.")
    p.add_argument("--sanity", action="store_true",
                   help="Quick sanity run: 2 epochs, 2000 samples.")
    p.add_argument("--no_timestamp", action="store_true",
                   help="Do not append timestamp to the run directory name.")
    return p.parse_args(argv)


def _load_yaml_file(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _r(cli_val, cfg_val, default):
    """Priority: CLI → config → default."""
    return cli_val if cli_val is not None else (cfg_val if cfg_val is not None else default)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    )

    args = _parse_args(argv)

    # ---- Load experiment config if provided --------------------------------
    train_cfg: dict = {}
    model_cfg: dict = {}
    ds_cfg: dict = {}

    if args.config:
        try:
            ecfg = load_experiment_config(str(args.config))
            train_cfg = ecfg.training
            model_cfg = ecfg.model
            ds_cfg = ecfg.dataset
        except Exception as exc:
            logger.warning("Config load failed (%s); falling back to defaults.", exc)

    # ---- Resolve hyperparameters (CLI > config > paper defaults) -----------
    seed         = _r(args.seed,         train_cfg.get("seed"),         _DEFAULTS["seed"])
    epochs       = _r(args.epochs,       train_cfg.get("epochs"),       _DEFAULTS["epochs"])
    batch_size   = _r(args.batch_size,   train_cfg.get("batch_size"),   _DEFAULTS["batch_size"])
    lr           = _r(args.lr,           train_cfg.get("lr"),           _DEFAULTS["lr"])
    weight_decay = _r(None,              train_cfg.get("weight_decay"), _DEFAULTS["weight_decay"])
    grad_clip    = _r(None,              train_cfg.get("grad_clip"),    _DEFAULTS["grad_clip"])
    patience     = _r(None,              train_cfg.get("patience"),     _DEFAULTS["patience"])
    fragment_size = _r(
        args.fragment_size,
        ds_cfg.get("fragment_size") or model_cfg.get("fragment_size"),
        _DEFAULTS["fragment_size"],
    )
    variant = _r(args.variant, model_cfg.get("variant"), _DEFAULTS["variant"])
    embed_dim    = model_cfg.get("embed_dim", _DEFAULTS["embed_dim"])
    channels     = model_cfg.get("channels",  _DEFAULTS["channels"])
    kernel_size  = model_cfg.get("kernel_size", _DEFAULTS["kernel_size"])
    se_reduction = model_cfg.get("se_reduction", _DEFAULTS["se_reduction"])
    p_dropout    = model_cfg.get("p_dropout", _DEFAULTS["p_dropout"])

    if args.sanity:
        epochs = 2
        batch_size = 64
        patience = 999   # disable early stopping for sanity

    # ---- Paths -------------------------------------------------------------
    data_dir = _r(
        args.data_dir,
        Path(ds_cfg["root_dir"]) if "root_dir" in ds_cfg else None,
        FFT75_DATA_DIR,
    )
    run_name = f"depthwisecnn_{variant}_fft75_{fragment_size}b"

    # ---- Reproducibility ---------------------------------------------------
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    logger.info(
        "=== DepthwiseCNN Training: variant=%s | fragment_size=%d | device=%s ===",
        variant, fragment_size, device
    )

    # ---- Datasets ----------------------------------------------------------
    tiny = 2000 if args.sanity else False
    train_ds = FragmentDataset(data_dir, "train", fragment_size, cache=True, tiny_subset=tiny)
    val_ds   = FragmentDataset(data_dir, "val",   fragment_size, cache=True, tiny_subset=tiny)
    test_ds  = FragmentDataset(data_dir, "test",  fragment_size, cache=True, tiny_subset=tiny)
    logger.info("Train: %s", train_ds)
    logger.info("Val  : %s", val_ds)
    logger.info("Test : %s", test_ds)

    train_loader = build_dataloader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = build_dataloader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = build_dataloader(test_ds,  batch_size=batch_size, shuffle=False)

    # ---- Model (via registry) ---------------------------------------------
    model_kwargs: dict[str, Any] = {
        "variant": variant,
        "fragment_size": fragment_size,
        "embed_dim": embed_dim,
        "channels": channels,
        "kernel_size": kernel_size,
        "se_reduction": se_reduction,
        "p_dropout": p_dropout,
    }
    model = build_model("depthwisecnn", num_classes=train_ds.num_classes, **model_kwargs)
    logger.info(
        "Model: depthwisecnn (%s) | classes=%d | params=%s",
        variant, train_ds.num_classes, f"{model.num_parameters():,}"
    )

    # ---- Run directory -----------------------------------------------------
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    else:
        run_dir = OUTPUTS_DIR / run_name
    ensure_dirs(CHECKPOINTS_DIR, run_dir)

    # ---- Trainer config ----------------------------------------------------
    trainer_config_dict: dict[str, Any] = {
        "epochs": epochs,
        "lr": lr,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "optimizer": train_cfg.get("optimizer", _DEFAULTS["optimizer"]),
        "scheduler": train_cfg.get("scheduler", _DEFAULTS["scheduler"]),
        "scheduler_factor": train_cfg.get("scheduler_factor", 0.5),
        "scheduler_patience": train_cfg.get("scheduler_patience", _DEFAULTS["scheduler_patience"]),
        "grad_clip": grad_clip,
        "patience": patience,
        "monitor": "val_acc",
        "monitor_mode": "max",
        "amp": train_cfg.get("amp", _DEFAULTS["amp"]),
        "seed": seed,
    }
    trainer_cfg = TrainerConfig.from_dict(trainer_config_dict)
    trainer = Trainer(model, trainer_cfg, run_dir)

    # ---- Resume ------------------------------------------------------------
    start_epoch = 1
    if args.resume and args.resume.exists():
        start_epoch = trainer.resume_from(args.resume)
        logger.info("Resumed from %s (starting epoch %d)", args.resume, start_epoch)

    # ---- Train -------------------------------------------------------------
    history = trainer.fit(train_loader, val_loader, start_epoch=start_epoch)

    # ---- Plots -------------------------------------------------------------
    plot_loss_curve(history.train_loss, history.val_loss, run_dir / "loss_curve.png")
    plot_accuracy_curve(history.train_acc, history.val_acc, run_dir / "accuracy_curve.png")
    plot_lr_curve(history.lr, run_dir / "lr_curve.png")

    # ---- Evaluate on test split (best checkpoint) --------------------------
    load_checkpoint(trainer.best_ckpt_path, model, map_location=trainer.device)
    evaluator = Evaluator(model, device=str(trainer.device))
    result = evaluator.evaluate_and_save(test_loader, run_dir, run_name=run_name)

    plot_confusion_matrix(result["metrics"].confusion, run_dir / "confusion_matrix.png")

    # ---- Save best checkpoint to canonical location ------------------------
    best_ckpt_canonical = CHECKPOINTS_DIR / f"best_{run_name}.pt"
    if trainer.best_ckpt_path.exists():
        import shutil
        shutil.copy2(trainer.best_ckpt_path, best_ckpt_canonical)
        logger.info("Best checkpoint → %s", best_ckpt_canonical)

    logger.info("=== DepthwiseCNN training complete. Run dir: %s ===", run_dir)
    logger.info(
        "  Best val_acc=%.4f (epoch %d)",
        trainer._best_metric or 0.0, trainer.best_epoch
    )
    logger.info("  Test accuracy=%.4f", result["metrics"].accuracy)


if __name__ == "__main__":
    main()
