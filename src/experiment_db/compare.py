"""Comparison utilities for benchmark experiment records."""

from __future__ import annotations

from dataclasses import dataclass

from src.experiment_db.experiment import ExperimentRecord


@dataclass(frozen=True)
class ExperimentComparison:
    """Metric deltas between two experiment records."""

    left_run_id: str
    right_run_id: str
    accuracy_delta: float | None
    macro_f1_delta: float | None
    weighted_f1_delta: float | None
    latency_delta_ms: float | None


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return right - left


def compare_records(left: ExperimentRecord, right: ExperimentRecord) -> ExperimentComparison:
    """Compare two records, returning right-minus-left metric deltas."""
    return ExperimentComparison(
        left_run_id=left.run_id,
        right_run_id=right.run_id,
        accuracy_delta=_delta(left.accuracy, right.accuracy),
        macro_f1_delta=_delta(left.macro_f1, right.macro_f1),
        weighted_f1_delta=_delta(left.weighted_f1, right.weighted_f1),
        latency_delta_ms=_delta(left.inference_latency_ms, right.inference_latency_ms),
    )
