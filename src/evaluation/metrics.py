"""
src/evaluation/metrics.py
----------------------------
Centralized metrics library. All sklearn metric calls live here — no
other module in the framework should import sklearn.metrics directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: np.ndarray | None = None
    top_k: dict[str, float] = field(default_factory=dict)

    def to_dict(self, include_confusion: bool = False) -> dict[str, Any]:
        d = {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "top_k": self.top_k,
        }
        if include_confusion and self.confusion is not None:
            d["confusion_matrix"] = self.confusion.tolist()
        return d


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    labels: list[int] | None = None,
    top_k: tuple[int, ...] = (1, 5),
) -> ClassificationMetrics:
    """Compute the full standard metric suite for a classification run.

    Parameters
    ----------
    y_true, y_pred : np.ndarray, shape [N]
    y_proba : np.ndarray, shape [N, C], optional
        Needed for top-k accuracy beyond k=1.
    labels : list[int], optional
        Explicit label set (ensures consistent confusion matrix / report
        shape even if some classes are absent from a small eval subset).
    """
    if labels is None:
        labels = sorted(set(np.unique(y_true).tolist()) | set(np.unique(y_pred).tolist()))

    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class: dict[str, dict[str, float]] = {}
    for i, label in enumerate(labels):
        per_class[str(label)] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    top_k_scores: dict[str, float] = {"top_1": accuracy}
    if y_proba is not None:
        for k in top_k:
            if k == 1:
                continue
            try:
                score = float(
                    top_k_accuracy_score(y_true, y_proba, k=k, labels=labels)
                )
                top_k_scores[f"top_{k}"] = score
            except ValueError:
                # e.g. k >= num_classes in a tiny smoke-test run
                continue

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_precision=float(macro_precision),
        macro_recall=float(macro_recall),
        macro_f1=float(macro_f1),
        weighted_f1=weighted_f1,
        per_class=per_class,
        confusion=cm,
        top_k=top_k_scores,
    )


def format_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] | None = None
) -> str:
    """Human-readable sklearn classification report string."""
    return classification_report(y_true, y_pred, labels=labels, zero_division=0)
