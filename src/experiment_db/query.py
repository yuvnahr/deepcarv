"""Query helpers for the experiment database."""

from __future__ import annotations

from typing import Iterable

from src.experiment_db.experiment import ExperimentRecord


def filter_records(
    records: Iterable[ExperimentRecord],
    model: str | None = None,
    dataset: str | None = None,
    fragment_size: int | None = None,
    min_accuracy: float | None = None,
) -> list[ExperimentRecord]:
    """Filter records by common benchmark dimensions."""
    result: list[ExperimentRecord] = []
    for record in records:
        if model is not None and record.model != model:
            continue
        if dataset is not None and record.dataset != dataset:
            continue
        if fragment_size is not None and record.fragment_size != fragment_size:
            continue
        if min_accuracy is not None:
            if record.accuracy is None or record.accuracy < min_accuracy:
                continue
        result.append(record)
    return result


def best_by_model(records: Iterable[ExperimentRecord]) -> dict[str, ExperimentRecord]:
    """Return the best accuracy record for each model."""
    best: dict[str, ExperimentRecord] = {}
    for record in records:
        if record.accuracy is None:
            continue
        current = best.get(record.model)
        if current is None or current.accuracy is None or record.accuracy > current.accuracy:
            best[record.model] = record
    return best
