"""
src/utils/statistics.py
------------------------
Small, dependency-light numeric helpers shared by metrics, evaluation,
and reporting modules. Keeps ad-hoc math out of those modules.
"""

from __future__ import annotations

import numpy as np


def mean_std(values: list[float]) -> tuple[float, float]:
    """Return (mean, population std) of a list of floats; (0, 0) if empty."""
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def percentile(values: list[float], q: float) -> float:
    """Return the q-th percentile (0-100) of a list of floats."""
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    """Standard latency summary used in evaluator/reporting outputs."""
    if not latencies_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    return {
        "mean_ms": float(np.mean(latencies_ms)),
        "p50_ms": percentile(latencies_ms, 50),
        "p95_ms": percentile(latencies_ms, 95),
        "p99_ms": percentile(latencies_ms, 99),
    }


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns `default` instead of raising on zero denominator."""
    if denominator == 0:
        return default
    return numerator / denominator
