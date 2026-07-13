"""
benchmarks/DepthwiseCNN/scripts/evaluate.py
---------------------------------------------
Evaluation entry point for the DepthwiseCNN benchmark.

Loads a trained checkpoint and runs full evaluation on the test split,
producing all standardized benchmark outputs:

    metrics.json          — full metrics (accuracy, F1, per-class)
    summary.json          — high-level summary
    predictions.csv       — per-sample y_true, y_pred, confidence
    confusion_matrix.csv  — raw NxN confusion matrix
    per_class_metrics.csv — per-class precision/recall/F1
    classification_report.txt
    confusion_matrix.png

Usage
-----
    # Evaluate a specific checkpoint:
    python benchmarks/DepthwiseCNN/scripts/evaluate.py \\
        --checkpoint checkpoints/best_depthwisecnn_dsc_fft75_512b.pt \\
        --data_dir data/FFT-75 \\
        --fragment_size 512 \\
        --variant dsc

    # From a composed experiment config:
    python benchmarks/DepthwiseCNN/scripts/evaluate.py \\
        --config configs/experiments/depthwisecnn_fft75_512.yaml \\
        --checkpoint checkpoints/best_depthwisecnn_dsc_fft75_512b.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import yaml

import torch

from src.data.dataset import FragmentDataset, build_dataloader
from src.evaluation.evaluator import Evaluator
from src.models.registry import build_model
from src.training.checkpointing import load_checkpoint
from src.utils.config import load_experiment_config
from src.utils.paths import FFT75_DATA_DIR, OUTPUTS_DIR, ensure_dirs
from src.visualization.confusion import plot_confusion_matrix

logger = logging.getLogger("depthwisecnn_evaluate")

_DEFAULTS: dict[str, Any] = {
    "fragment_size": 512,
    "variant": "dsc",
    "embed_dim": 64,
    "channels": [64, 128, 256, 256],
    "kernel_size": 3,
    "se_reduction": 16,
    "p_dropout": 0.0,   # Disable dropout at eval time (model.eval() also disables it)
    "batch_size": 256,
}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="DepthwiseCNN benchmark evaluation on FFT-75."
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to a trained .pt checkpoint.")
    p.add_argument("--config", type=Path, default=None,
                   help="Experiment YAML config (optional; used to infer model kwargs).")
    p.add_argument("--data_dir", type=Path, default=None,
                   help="FFT-75 root directory.")
    p.add_argument("--fragment_size", type=int, default=None, choices=[512, 4096])
    p.add_argument("--variant", type=str, default=None,
                   choices=["dsc", "dsc_se", "m_dsc"])
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Output directory for evaluation results.")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "val", "test"],
                   help="Dataset split to evaluate (default: test).")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--sanity", action="store_true",
                   help="Evaluate on a tiny subset (2000 samples).")
    return p.parse_args(argv)


def _r(cli_val, cfg_val, default):
    return cli_val if cli_val is not None else (cfg_val if cfg_val is not None else default)


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    )

    args = _parse_args(argv)

    # ---- Load experiment config if provided --------------------------------
    model_cfg: dict = {}
    ds_cfg: dict = {}
    eval_cfg: dict = {}

    if args.config:
        try:
            ecfg = load_experiment_config(str(args.config))
            model_cfg = ecfg.model
            ds_cfg = ecfg.dataset
            eval_cfg = ecfg.evaluation
        except Exception as exc:
            logger.warning("Config load failed (%s); using defaults.", exc)

    # ---- Resolve parameters ------------------------------------------------
    fragment_size = _r(
        args.fragment_size,
        ds_cfg.get("fragment_size") or model_cfg.get("fragment_size"),
        _DEFAULTS["fragment_size"],
    )
    variant     = _r(args.variant,    model_cfg.get("variant"),     _DEFAULTS["variant"])
    embed_dim   = model_cfg.get("embed_dim",   _DEFAULTS["embed_dim"])
    channels    = model_cfg.get("channels",    _DEFAULTS["channels"])
    kernel_size = model_cfg.get("kernel_size", _DEFAULTS["kernel_size"])
    se_reduction = model_cfg.get("se_reduction", _DEFAULTS["se_reduction"])
    batch_size  = _r(args.batch_size, eval_cfg.get("batch_size"), _DEFAULTS["batch_size"])

    data_dir = _r(
        args.data_dir,
        Path(ds_cfg["root_dir"]) if "root_dir" in ds_cfg else None,
        FFT75_DATA_DIR,
    )

    # ---- Load checkpoint to infer num_classes if needed --------------------
    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(str(args.checkpoint), map_location="cpu")
    # Try to infer num_classes from checkpoint metadata
    num_classes_from_ckpt: int | None = ckpt.get("num_classes") or ckpt.get("extra", {}).get("num_classes")

    # ---- Dataset -----------------------------------------------------------
    tiny = 2000 if args.sanity else False
    dataset = FragmentDataset(data_dir, args.split, fragment_size, cache=True, tiny_subset=tiny)
    loader = build_dataloader(dataset, batch_size=batch_size, shuffle=False)
    num_classes = num_classes_from_ckpt or dataset.num_classes
    logger.info("Dataset: %s", dataset)

    # ---- Model (via registry) ---------------------------------------------
    model_kwargs: dict[str, Any] = {
        "variant": variant,
        "fragment_size": fragment_size,
        "embed_dim": embed_dim,
        "channels": channels,
        "kernel_size": kernel_size,
        "se_reduction": se_reduction,
        "p_dropout": 0.0,
    }
    model = build_model("depthwisecnn", num_classes=num_classes, **model_kwargs)
    load_checkpoint(args.checkpoint, model, map_location="cpu")
    logger.info("Model: depthwisecnn (%s) | params=%s", variant, f"{model.num_parameters():,}")

    # ---- Output directory --------------------------------------------------
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        run_name = f"depthwisecnn_{variant}_fft75_{fragment_size}b_eval"
        out_dir = OUTPUTS_DIR / run_name
    ensure_dirs(out_dir)

    # ---- Evaluate ----------------------------------------------------------
    run_name = out_dir.name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = Evaluator(model, device=device)
    result = evaluator.evaluate_and_save(loader, out_dir, run_name=run_name)

    # ---- Confusion matrix plot ---------------------------------------------
    plot_confusion_matrix(result["metrics"].confusion, out_dir / "confusion_matrix.png")

    # ---- Summary -----------------------------------------------------------
    metrics = result["metrics"]
    logger.info("=== Evaluation complete ===")
    logger.info("  Split       : %s", args.split)
    logger.info("  N samples   : %d", result["n_samples"])
    logger.info("  Accuracy    : %.4f", metrics.accuracy)
    logger.info("  Macro F1    : %.4f", metrics.macro_f1)
    logger.info("  Weighted F1 : %.4f", metrics.weighted_f1)
    logger.info("  Latency     : %.3f ms/sample", result["latency_ms_per_sample"])
    logger.info("  Results dir : %s", out_dir)


if __name__ == "__main__":
    main()
