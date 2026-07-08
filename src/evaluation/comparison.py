"""
src/evaluation/comparison.py
--------------------------------
Reads standardized run outputs (outputs/<run_name>/summary.json,
metrics.json, config.yaml, environment.json) and produces a
cross-run comparison table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_run(run_dir: Path) -> dict[str, Any]:
    """Load one run directory's outputs into a flat dict for comparison."""
    run_dir = Path(run_dir)
    summary = _read_json(run_dir / "summary.json")
    metrics = _read_json(run_dir / "metrics.json")
    config = _read_yaml(run_dir / "config.yaml")
    env = _read_json(run_dir / "environment.json")

    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})

    row = {
        "run_name": run_dir.name,
        "model": model_cfg.get("name", model_cfg.get("model_name", "unknown")),
        "dataset": dataset_cfg.get("name", "FFT-75"),
        "fragment_size": dataset_cfg.get("fragment_size"),
        "accuracy": summary.get("accuracy", metrics.get("accuracy")),
        "macro_precision": metrics.get("macro_precision"),
        "macro_recall": metrics.get("macro_recall"),
        "macro_f1": metrics.get("macro_f1", summary.get("macro_f1")),
        "weighted_f1": metrics.get("weighted_f1", summary.get("weighted_f1")),
        "latency_ms_per_sample": summary.get("latency_ms_per_sample"),
        "peak_gpu_memory_mb": summary.get("peak_gpu_memory_mb"),
        "gpu_name": summary.get("gpu_name"),
        "git_commit": env.get("git_commit"),
    }
    return row


def compare_runs(run_dirs: list[Path]) -> pd.DataFrame:
    """Build a comparison DataFrame across multiple run directories."""
    rows = [load_run(d) for d in run_dirs]
    df = pd.DataFrame(rows)
    if "accuracy" in df.columns:
        df = df.sort_values("accuracy", ascending=False)
    return df


def discover_runs(outputs_dir: Path) -> list[Path]:
    """Find all run directories under outputs_dir that have a summary.json."""
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.exists():
        return []
    return sorted(
        d for d in outputs_dir.iterdir()
        if d.is_dir() and (d / "summary.json").exists()
    )
