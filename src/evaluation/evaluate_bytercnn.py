"""
src/evaluation/evaluate_bytercnn.py
-------------------------------------
Frozen test-set evaluation for the ByteRCNN FFT-75 Scenario #1 baseline.

Outputs written to outputs/bytercnn_fft75/ (or --out_dir):
    metrics.json          — scalar metrics (accuracy, macro P/R/F1, …)
    confusion_matrix.csv  — full num_classes × num_classes matrix
    per_class_metrics.csv — per-class precision, recall, F1, support
    predictions.csv       — sample_id, true_label, pred_label, confidence
    eval_summary.txt      — human-readable table

Measurement protocol:
    1. Warm-up: 5 batches (GPU JIT / caches)
    2. Timed inference: remaining batches
    3. Peak GPU memory recorded after full pass

Usage
-----
    python -m src.evaluation.evaluate_bytercnn \\
        --config configs/fft75_s1_512_bytercnn.yaml

    # Manual override:
    python -m src.evaluation.evaluate_bytercnn \\
        --checkpoint checkpoints/best_bytercnn_fft75.pt \\
        --test_csv   data/splits/fft75_s1_512/test.csv \\
        --class_map  data/splits/fft75_s1_512/class_map.json \\
        --out_dir    outputs/bytercnn_fft75
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.data.dataset import FragmentDataset, build_dataloader
from src.models.bytercnn_wrapper import build_bytercnn
from src.utils.logging import get_simple_logger
from src.utils.paths import (
    BYTERCNN_BEST_CKPT,
    BYTERCNN_RUN_DIR,
    FFT75_CLASS_MAP,
    FFT75_TEST_CSV,
    ensure_dirs,
)
from src.utils.seed import set_seed

logger = get_simple_logger("evaluate_bytercnn")

WARMUP_BATCHES: int = 5


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _run_inference(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Run inference with warmup batches.

    Returns
    -------
    all_true   : shape [N]
    all_pred   : shape [N]
    all_conf   : shape [N]  — max softmax probability
    time_per_sample_ms : float
    peak_gpu_mb : float (0 if no CUDA)
    """
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    all_true: list[int] = []
    all_pred: list[int] = []
    all_conf: list[float] = []

    # ---- Warm-up --------------------------------------------------------
    warmed_up = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            _ = model(x)
            warmed_up += 1
            if warmed_up >= WARMUP_BATCHES:
                break
    logger.info("Warm-up complete (%d batches).", warmed_up)

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    # ---- Timed inference ------------------------------------------------
    n_timed = 0
    t_start = time.perf_counter()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            log_probs = model(x)
            probs = log_probs.exp()

            preds = probs.argmax(dim=1)
            confs = probs.max(dim=1).values

            all_true.extend(y.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
            all_conf.extend(confs.cpu().tolist())
            n_timed += x.size(0)

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    t_end = time.perf_counter()

    elapsed_s = t_end - t_start
    time_per_sample_ms = (elapsed_s / max(n_timed, 1)) * 1000.0

    if torch.cuda.is_available():
        peak_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_gpu_mb = 0.0

    return (
        np.array(all_true, dtype=np.int64),
        np.array(all_pred, dtype=np.int64),
        np.array(all_conf, dtype=np.float32),
        time_per_sample_ms,
        peak_gpu_mb,
    )


# ---------------------------------------------------------------------------
# Metrics & output saving
# ---------------------------------------------------------------------------


def _compute_and_save(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    confidences: np.ndarray,
    class_map: dict[str, int],
    time_per_sample_ms: float,
    peak_gpu_mb: float,
    sample_ids: list[str],
    out_dir: Path,
) -> dict:
    """Compute all metrics and write output files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    id_to_label = {v: k for k, v in class_map.items()}
    label_names = [id_to_label[i] for i in range(len(class_map))]

    acc = accuracy_score(true_labels, pred_labels)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="weighted", zero_division=0
    )
    prec_per, rec_per, f1_per, support_per = precision_recall_fscore_support(
        true_labels, pred_labels, average=None, zero_division=0,
        labels=list(range(len(class_map))),
    )

    # ---- metrics.json ---------------------------------------------------
    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(prec_weighted),
        "weighted_recall": float(rec_weighted),
        "weighted_f1": float(f1_weighted),
        "time_per_sample_ms": float(time_per_sample_ms),
        "peak_gpu_memory_mb": float(peak_gpu_mb),
        "num_test_samples": int(len(true_labels)),
        "num_classes": int(len(class_map)),
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("metrics.json → %s", metrics_path)

    # ---- confusion_matrix.csv ------------------------------------------
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(class_map))))
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    cm_path = out_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    logger.info("confusion_matrix.csv → %s", cm_path)

    # ---- per_class_metrics.csv -----------------------------------------
    per_class_df = pd.DataFrame(
        {
            "label": label_names,
            "label_id": list(range(len(class_map))),
            "precision": prec_per,
            "recall": rec_per,
            "f1": f1_per,
            "support": support_per.astype(int),
        }
    )
    per_class_path = out_dir / "per_class_metrics.csv"
    per_class_df.to_csv(per_class_path, index=False)
    logger.info("per_class_metrics.csv → %s", per_class_path)

    # ---- predictions.csv -----------------------------------------------
    # Truncate sample_ids to match (may differ if test split re-indexed)
    n = len(true_labels)
    sid_col = sample_ids[:n] if len(sample_ids) >= n else sample_ids + [""] * (n - len(sample_ids))

    pred_df = pd.DataFrame(
        {
            "sample_id": sid_col,
            "true_label": [id_to_label.get(t, str(t)) for t in true_labels],
            "pred_label": [id_to_label.get(p, str(p)) for p in pred_labels],
            "true_label_id": true_labels,
            "pred_label_id": pred_labels,
            "confidence": confidences,
            "correct": (true_labels == pred_labels).astype(int),
        }
    )
    pred_path = out_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info("predictions.csv → %s", pred_path)

    # ---- eval_summary.txt ----------------------------------------------
    summary_lines = [
        "=" * 70,
        "ByteRCNN — FFT-75 Scenario #1  (512 bytes, 75 classes)",
        "FROZEN TEST-SET EVALUATION",
        "=" * 70,
        f"{'Accuracy':<35} {acc:.6f}  ({acc*100:.2f}%)",
        f"{'Macro Precision':<35} {prec_macro:.6f}",
        f"{'Macro Recall':<35} {rec_macro:.6f}",
        f"{'Macro F1':<35} {f1_macro:.6f}",
        f"{'Weighted F1':<35} {f1_weighted:.6f}",
        "-" * 70,
        f"{'Inference time / sample (ms)':<35} {time_per_sample_ms:.4f}",
        f"{'Peak GPU memory (MB)':<35} {peak_gpu_mb:.2f}",
        f"{'Test samples':<35} {len(true_labels):,}",
        f"{'Classes':<35} {len(class_map)}",
        "=" * 70,
        "",
        "Per-class Classification Report:",
        "-" * 70,
        classification_report(
            true_labels, pred_labels,
            target_names=label_names,
            zero_division=0,
        ),
    ]
    summary_txt = "\n".join(summary_lines)
    summary_path = out_dir / "eval_summary.txt"
    summary_path.write_text(summary_txt)
    logger.info("eval_summary.txt → %s", summary_path)
    print("\n" + summary_txt)

    return metrics


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ByteRCNN evaluation on frozen test set.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--test_csv", type=Path, default=None)
    p.add_argument("--class_map", type=Path, default=None)
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _load_config(config_path: Path | None) -> dict:
    if config_path is None or not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = _load_config(args.config)
    path_cfg = cfg.get("paths", {})
    eval_cfg = cfg.get("evaluation", {})

    checkpoint_path: Path = (
        args.checkpoint
        or Path(path_cfg.get("best_checkpoint", str(BYTERCNN_BEST_CKPT)))
    )
    test_csv: Path = (
        args.test_csv
        or Path(path_cfg.get("test_csv", str(FFT75_TEST_CSV)))
    )
    class_map_path: Path = (
        args.class_map
        or Path(path_cfg.get("class_map", str(FFT75_CLASS_MAP)))
    )
    out_dir: Path = (
        args.out_dir
        or Path(path_cfg.get("run_outputs", str(BYTERCNN_RUN_DIR)))
    )
    batch_size: int = eval_cfg.get("batch_size", args.batch_size)
    seed: int = args.seed

    # ---- Setup ----------------------------------------------------------
    set_seed(seed)
    ensure_dirs(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=== ByteRCNN Evaluation — FFT-75 Frozen Test Set ===")
    logger.info("Device      : %s", device)
    logger.info("Checkpoint  : %s", checkpoint_path)
    logger.info("Test CSV    : %s", test_csv)
    logger.info("Output dir  : %s", out_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run train_bytercnn.py first."
        )

    # ---- Load checkpoint & model ----------------------------------------
    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes = ckpt.get("num_classes", 75)

    model = build_bytercnn(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_acc = ckpt.get("val_acc", float("nan"))
    logger.info(
        "Checkpoint loaded. Epoch=%s  Val-acc=%.4f",
        ckpt.get("epoch", "?"), val_acc,
    )

    # ---- Dataset & loader -----------------------------------------------
    test_ds = FragmentDataset(
        csv_path=test_csv,
        class_map_path=class_map_path,
        cache=False,
    )
    logger.info("Test dataset: %s", test_ds)

    test_loader = build_dataloader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    sample_ids = test_ds.df["sample_id"].tolist()

    # ---- Inference ------------------------------------------------------
    logger.info("Running inference (warmup=%d batches) …", WARMUP_BATCHES)
    true_arr, pred_arr, conf_arr, time_per_ms, peak_mb = _run_inference(
        model, test_loader, device
    )
    logger.info(
        "Inference done. Time/sample=%.3f ms  Peak GPU=%.1f MB",
        time_per_ms, peak_mb,
    )

    # ---- Compute & save metrics -----------------------------------------
    _compute_and_save(
        true_labels=true_arr,
        pred_labels=pred_arr,
        confidences=conf_arr,
        class_map=test_ds.class_map,
        time_per_sample_ms=time_per_ms,
        peak_gpu_mb=peak_mb,
        sample_ids=sample_ids,
        out_dir=out_dir,
    )

    logger.info("=== Evaluation complete. Outputs in %s ===", out_dir)


if __name__ == "__main__":
    main()
