"""
src/evaluation/evaluate_bytercnn.py
-------------------------------------
Frozen test-set evaluation for the ByteRCNN FFT-75 Scenario #1 baseline.

Loads the official test.npz split.  Does not regenerate or modify splits.

Outputs written to outputs/bytercnn_fft75/ (or --out_dir):
    metrics.json
    confusion_matrix.csv
    per_class_metrics.csv
    predictions.csv
    eval_summary.txt

Usage
-----
    python -m src.evaluation.evaluate_bytercnn \\
        --config configs/fft75_s1_512_bytercnn.yaml

    # Manual override:
    python -m src.evaluation.evaluate_bytercnn \\
        --checkpoint checkpoints/best_bytercnn_fft75.pt \\
        --data_dir   data/FFT-75 \\
        --fragment_size 512 \\
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
    FFT75_DATA_DIR,
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
    """Warm-up then timed inference. Returns (true, pred, conf, ms/sample, peak_mb)."""
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    all_true: list[int] = []
    all_pred: list[int] = []
    all_conf: list[float] = []

    # Warm-up
    warmed = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            _ = model(x)
            warmed += 1
            if warmed >= WARMUP_BATCHES:
                break
    logger.info("Warm-up complete (%d batches).", warmed)

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    # Timed pass
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

    elapsed = time.perf_counter() - t_start
    time_per_ms = (elapsed / max(n_timed, 1)) * 1000.0
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0

    return (
        np.array(all_true, dtype=np.int64),
        np.array(all_pred, dtype=np.int64),
        np.array(all_conf, dtype=np.float32),
        time_per_ms,
        peak_mb,
    )


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------


def _compute_and_save(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    confidences: np.ndarray,
    num_classes: int,
    time_per_sample_ms: float,
    peak_gpu_mb: float,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use integer labels as class names (no external class_map needed)
    label_names = [str(i) for i in range(num_classes)]
    label_ids   = list(range(num_classes))

    acc = accuracy_score(true_labels, pred_labels)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="macro", zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="weighted", zero_division=0
    )
    prec_per, rec_per, f1_per, sup_per = precision_recall_fscore_support(
        true_labels, pred_labels, average=None, zero_division=0, labels=label_ids
    )

    # metrics.json
    metrics = {
        "accuracy":          float(acc),
        "macro_precision":   float(prec_macro),
        "macro_recall":      float(rec_macro),
        "macro_f1":          float(f1_macro),
        "weighted_precision":float(prec_w),
        "weighted_recall":   float(rec_w),
        "weighted_f1":       float(f1_w),
        "time_per_sample_ms":float(time_per_sample_ms),
        "peak_gpu_memory_mb":float(peak_gpu_mb),
        "num_test_samples":  int(len(true_labels)),
        "num_classes":       num_classes,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("metrics.json → %s", out_dir / "metrics.json")

    # confusion_matrix.csv
    cm = confusion_matrix(true_labels, pred_labels, labels=label_ids)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    cm_df.to_csv(out_dir / "confusion_matrix.csv")
    logger.info("confusion_matrix.csv → %s", out_dir / "confusion_matrix.csv")

    # per_class_metrics.csv
    pc_df = pd.DataFrame({
        "label_id":  label_ids,
        "precision": prec_per,
        "recall":    rec_per,
        "f1":        f1_per,
        "support":   sup_per.astype(int),
    })
    pc_df.to_csv(out_dir / "per_class_metrics.csv", index=False)
    logger.info("per_class_metrics.csv → %s", out_dir / "per_class_metrics.csv")

    # predictions.csv
    pred_df = pd.DataFrame({
        "true_label_id":  true_labels,
        "pred_label_id":  pred_labels,
        "confidence":     confidences,
        "correct":        (true_labels == pred_labels).astype(int),
    })
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    logger.info("predictions.csv → %s", out_dir / "predictions.csv")

    # eval_summary.txt
    summary = "\n".join([
        "=" * 70,
        "ByteRCNN — FFT-75 Scenario #1  (512 bytes, 75 classes)",
        "FROZEN TEST-SET EVALUATION",
        "=" * 70,
        f"{'Accuracy':<35} {acc:.6f}  ({acc*100:.2f}%)",
        f"{'Macro Precision':<35} {prec_macro:.6f}",
        f"{'Macro Recall':<35} {rec_macro:.6f}",
        f"{'Macro F1':<35} {f1_macro:.6f}",
        f"{'Weighted F1':<35} {f1_w:.6f}",
        "-" * 70,
        f"{'Inference time / sample (ms)':<35} {time_per_sample_ms:.4f}",
        f"{'Peak GPU memory (MB)':<35} {peak_gpu_mb:.2f}",
        f"{'Test samples':<35} {len(true_labels):,}",
        f"{'Classes':<35} {num_classes}",
        "=" * 70,
        "",
        "Per-class Classification Report:",
        "-" * 70,
        classification_report(true_labels, pred_labels, zero_division=0),
    ])
    (out_dir / "eval_summary.txt").write_text(summary)
    logger.info("eval_summary.txt → %s", out_dir / "eval_summary.txt")
    print("\n" + summary)

    return metrics


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ByteRCNN evaluation on frozen test set.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--data_dir", type=Path, default=None)
    p.add_argument("--fragment_size", type=int, default=None)
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _load_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = _load_config(args.config)

    ds_cfg   = cfg.get("dataset", {})
    path_cfg = cfg.get("paths", {})
    eval_cfg = cfg.get("evaluation", {})

    checkpoint_path = (
        args.checkpoint
        or Path(path_cfg.get("best_checkpoint", str(BYTERCNN_BEST_CKPT)))
    )
    data_dir = (
        args.data_dir
        or Path(ds_cfg.get("root_dir", str(FFT75_DATA_DIR)))
    )
    fragment_size = (
        args.fragment_size
        or int(ds_cfg.get("fragment_size", 512))
    )
    out_dir = (
        args.out_dir
        or Path(path_cfg.get("run_outputs", str(BYTERCNN_RUN_DIR)))
    )
    batch_size = eval_cfg.get("batch_size", args.batch_size)

    set_seed(args.seed)
    ensure_dirs(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=== ByteRCNN Evaluation — FFT-75 Frozen Test Set ===")
    logger.info("Device        : %s", device)
    logger.info("Checkpoint    : %s", checkpoint_path)
    logger.info("Data dir      : %s", data_dir)
    logger.info("Fragment size : %d", fragment_size)
    logger.info("Output dir    : %s", out_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run train_bytercnn.py first."
        )

    # ---- Load model -----------------------------------------------------
    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes = ckpt.get("num_classes", 75)
    model = build_bytercnn(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Checkpoint loaded. Epoch=%s  Val-acc=%.4f",
                ckpt.get("epoch", "?"), ckpt.get("val_acc", float("nan")))

    # ---- Dataset --------------------------------------------------------
    test_ds = FragmentDataset(
        root_dir=data_dir, split="test",
        fragment_size=fragment_size, cache=True,
    )
    logger.info("%s", test_ds)
    test_loader = build_dataloader(test_ds, batch_size=batch_size, shuffle=False)

    # ---- Inference ------------------------------------------------------
    logger.info("Running inference (warmup=%d batches) …", WARMUP_BATCHES)
    true_arr, pred_arr, conf_arr, time_ms, peak_mb = _run_inference(model, test_loader, device)
    logger.info("Inference done. Time/sample=%.3f ms  Peak GPU=%.1f MB", time_ms, peak_mb)

    # ---- Metrics & save -------------------------------------------------
    _compute_and_save(
        true_labels=true_arr,
        pred_labels=pred_arr,
        confidences=conf_arr,
        num_classes=num_classes,
        time_per_sample_ms=time_ms,
        peak_gpu_mb=peak_mb,
        out_dir=out_dir,
    )
    logger.info("=== Evaluation complete. Outputs in %s ===", out_dir)


if __name__ == "__main__":
    main()
