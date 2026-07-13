"""
benchmarks/DepthwiseCNN/scripts/evaluate.py
--------------------------------------------
Evaluation entry-point for the DepthwiseCNN benchmark.

Loads a trained checkpoint, runs the shared DeepCarv
:class:`~src.evaluation.evaluator.Evaluator` over a chosen split, and
writes the full standard output set (metrics.json, summary.json,
predictions.csv, confusion_matrix.csv, per_class_metrics.csv,
classification_report.txt) to disk.

This script contains **no** custom evaluation logic — all metric
computation lives in :mod:`src.evaluation.evaluator`.

Usage
-----
From the repository root::

    # Evaluate the best checkpoint on the test split:
    python -m benchmarks.DepthwiseCNN.scripts.evaluate \\
        --config benchmarks/DepthwiseCNN/configs/benchmark.yaml \\
        --checkpoint outputs/DepthwiseCNN/checkpoint_best.pt

    # Evaluate on the validation split with a custom output directory:
    python -m benchmarks.DepthwiseCNN.scripts.evaluate \\
        --config benchmarks/DepthwiseCNN/configs/benchmark.yaml \\
        --checkpoint outputs/DepthwiseCNN/checkpoint_best.pt \\
        --split val \\
        --out-dir outputs/DepthwiseCNN/eval_val

Config keys read
----------------
    dataset.root_dir        — FFT-75 data root
    dataset.fragment_size   — 512 or 4096
    dataset.cache           — load NPZ into RAM (bool)
    model.name              — registry key ("depthwisecnn")
    model.kwargs            — forwarded to build_model (variant, dropout_p, …)
    evaluation.batch_size   — inference batch size (default 512)
    evaluation.split        — default split if --split is not given
    paths.eval_outputs      — default output directory
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

from src.data.dataset import FragmentDataset, build_dataloader
from src.evaluation.evaluator import Evaluator
from src.models.registry import build_model
from src.training.checkpointing import load_checkpoint
from src.utils.logging import RunLogger
from src.utils.paths import FFT75_DATA_DIR, LOGS_DIR, OUTPUTS_DIR, ensure_dirs
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained DepthwiseCNN checkpoint on FFT-75.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to the benchmark YAML used during training.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the .pt checkpoint file to evaluate.",
    )
    p.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help=(
            "Dataset split to evaluate.  Defaults to evaluation.split in the "
            "config (which defaults to 'test')."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        dest="out_dir",
        help=(
            "Directory to write evaluation outputs.  Defaults to "
            "paths.eval_outputs in the config."
        ),
    )
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
        "--batch-size",
        type=int,
        default=None,
        dest="batch_size",
        help="Override evaluation.batch_size from the config.",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        dest="run_name",
        help="Human-readable identifier for the log directory.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"[evaluate] Config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        sys.exit(f"[evaluate] Config must be a YAML mapping: {path}")
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = _load_config(args.config)

    ds_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    eval_cfg = cfg.get("evaluation", {})
    path_cfg = cfg.get("paths", {})

    # --- Reproducibility (for consistent class ordering etc.) ---
    set_seed(42)

    # --- Resolve settings ---
    data_dir = Path(ds_cfg.get("root_dir", str(FFT75_DATA_DIR)))
    fragment_size: int = int(args.fragment_size or ds_cfg.get("fragment_size", 4096))
    cache: bool = bool(ds_cfg.get("cache", True))

    split: str = args.split or eval_cfg.get("split", "test")
    batch_size: int = int(args.batch_size or eval_cfg.get("batch_size", 512))

    default_eval_dir = Path(
        path_cfg.get("eval_outputs", str(OUTPUTS_DIR / "DepthwiseCNN" / "eval"))
    )
    out_dir: Path = args.out_dir or default_eval_dir
    ensure_dirs(out_dir, LOGS_DIR)

    # --- Checkpoint validation ---
    if not args.checkpoint.exists():
        sys.exit(f"[evaluate] Checkpoint not found: {args.checkpoint}")

    # --- Run identifier ---
    variant = args.variant or model_cfg.get("kwargs", {}).get("variant", "dsc")
    run_name = args.run_name or f"depthwisecnn_eval_{variant}_{fragment_size}b_{split}"

    # --- Logging ---
    with RunLogger(run_name=run_name, base_dir=LOGS_DIR) as run_logger:
        run_logger.info(
            "=== DepthwiseCNN Evaluation  variant=%s  split=%s  fragment_size=%d ===",
            variant, split, fragment_size,
        )
        run_logger.info("Checkpoint : %s", args.checkpoint)
        run_logger.info("Output dir : %s", out_dir)

        # --- Dataset ---
        run_logger.info("Loading %s split from %s …", split, data_dir)
        eval_ds = FragmentDataset(
            root_dir=data_dir,
            split=split,
            fragment_size=fragment_size,
            cache=cache,
        )
        run_logger.info("%s", eval_ds)

        eval_loader = build_dataloader(
            eval_ds, batch_size=batch_size, shuffle=False, drop_last=False
        )

        # --- Model (via registry) ---
        model_kwargs = model_cfg.get("kwargs", {})
        if args.variant:
            model_kwargs["variant"] = args.variant
            
        model = build_model(
            model_cfg["name"],
            num_classes=eval_ds.num_classes,
            **model_kwargs,
        )
        run_logger.info(
            "Model: %s | classes=%d | params=%s",
            model.name,
            eval_ds.num_classes,
            f"{model.num_parameters():,}",
        )

        # --- Load checkpoint ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = load_checkpoint(args.checkpoint, model, map_location=device)
        trained_epoch = ckpt.get("epoch", "unknown")
        trained_val_acc = ckpt.get("metrics", {}).get("val_acc", "unknown")
        run_logger.info(
            "Checkpoint loaded  (trained epoch=%s, val_acc=%s)",
            trained_epoch, trained_val_acc,
        )

        # --- Evaluate (full output suite via the shared evaluator) ---
        evaluator = Evaluator(model, device=str(device))
        result = evaluator.evaluate_and_save(
            eval_loader,
            out_dir=out_dir,
            run_name=run_name,
        )

        # --- Log top-level summary ---
        metrics = result["metrics"]
        run_logger.info("Accuracy          : %.4f", metrics.accuracy)
        run_logger.info("Macro F1          : %.4f", metrics.macro_f1)
        run_logger.info("Weighted F1       : %.4f", metrics.weighted_f1)
        run_logger.info(
            "Latency (ms/sample): %.3f", result["latency_ms_per_sample"]
        )
        run_logger.info("N samples         : %d", result["n_samples"])
        if result.get("peak_gpu_memory_mb") is not None and result["peak_gpu_memory_mb"] > 0:
            run_logger.info(
                "Peak GPU memory   : %.1f MB", result["peak_gpu_memory_mb"]
            )
        run_logger.info("=== Evaluation complete. Outputs → %s ===", out_dir)

        # --- Save extended eval summary into the log directory ---
        run_logger.save_eval_summary({
            "variant": variant,
            "fragment_size": fragment_size,
            "split": split,
            "num_classes": eval_ds.num_classes,
            "param_count": model.num_parameters(),
            "trained_epoch": trained_epoch,
            "trained_val_acc": trained_val_acc,
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "weighted_f1": metrics.weighted_f1,
            "latency_ms_per_sample": result["latency_ms_per_sample"],
            "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
            "gpu_name": result.get("gpu_name", "N/A"),
            "n_samples": result["n_samples"],
            "checkpoint": str(args.checkpoint),
            "outputs_dir": str(out_dir),
        })


if __name__ == "__main__":
    main()
