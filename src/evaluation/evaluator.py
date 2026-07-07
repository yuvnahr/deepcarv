"""
src/evaluation/evaluator.py
------------------------------
Generic, model-agnostic evaluator. Depends only on the FragmentClassifier
interface and standard DataLoaders — no ByteRCNN-specific (or any
model-specific) logic.

Produces the standardized output set for every run:
    metrics.json
    summary.json
    predictions.csv
    confusion_matrix.csv
    per_class_metrics.csv
    classification_report.txt
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_classification_metrics, format_classification_report
from src.utils.device import get_device, gpu_name, peak_memory_mb, reset_peak_memory

logger = logging.getLogger(__name__)


class Evaluator:
    """Runs a full test-set evaluation for any registered model."""

    def __init__(self, model: nn.Module, device: str | None = None) -> None:
        self.device = get_device(device)
        self.model = model.to(self.device)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        """Run inference over `loader` and compute the full metric suite.

        Returns a dict with keys: metrics, y_true, y_pred, y_proba,
        latency_ms_per_sample, peak_gpu_memory_mb, n_samples.
        """
        self.model.eval()
        reset_peak_memory(self.device)

        all_true: list[np.ndarray] = []
        all_pred: list[np.ndarray] = []
        all_proba: list[np.ndarray] = []
        latencies_ms: list[float] = []

        for x, y in loader:
            x = x.to(self.device, non_blocking=True)

            t0 = time.perf_counter()
            log_probs = self.model(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            per_sample_ms = (elapsed / max(x.size(0), 1)) * 1000
            latencies_ms.extend([per_sample_ms] * x.size(0))

            proba = log_probs.exp().cpu().numpy()
            pred = proba.argmax(axis=1)

            all_true.append(y.numpy())
            all_pred.append(pred)
            all_proba.append(proba)

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        y_proba = np.concatenate(all_proba)

        num_classes = getattr(self.model, "num_classes", y_proba.shape[1])
        labels = list(range(num_classes))
        metrics = compute_classification_metrics(y_true, y_pred, y_proba, labels=labels)

        return {
            "metrics": metrics,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "latency_ms_per_sample": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
            "peak_gpu_memory_mb": peak_memory_mb(self.device),
            "gpu_name": gpu_name(self.device),
            "n_samples": int(y_true.shape[0]),
        }

    def evaluate_and_save(self, loader: DataLoader, out_dir: Path, run_name: str = "run") -> dict:
        """Evaluate and write the full standardized output set to `out_dir`."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        result = self.evaluate(loader)
        metrics = result["metrics"]

        # metrics.json
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics.to_dict(include_confusion=False), f, indent=2)

        # summary.json
        summary = {
            "run_name": run_name,
            "n_samples": result["n_samples"],
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "weighted_f1": metrics.weighted_f1,
            "latency_ms_per_sample": result["latency_ms_per_sample"],
            "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
            "gpu_name": result["gpu_name"],
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # predictions.csv
        pd.DataFrame({
            "y_true": result["y_true"],
            "y_pred": result["y_pred"],
            "confidence": result["y_proba"].max(axis=1),
        }).to_csv(out_dir / "predictions.csv", index=False)

        # confusion_matrix.csv
        if metrics.confusion is not None:
            pd.DataFrame(metrics.confusion).to_csv(out_dir / "confusion_matrix.csv", index=False)

        # per_class_metrics.csv
        per_class_df = pd.DataFrame.from_dict(metrics.per_class, orient="index")
        per_class_df.index.name = "class"
        per_class_df.to_csv(out_dir / "per_class_metrics.csv")

        # classification_report.txt
        report_str = format_classification_report(result["y_true"], result["y_pred"])
        (out_dir / "classification_report.txt").write_text(report_str)

        logger.info("Evaluation outputs written to %s", out_dir)
        return result
