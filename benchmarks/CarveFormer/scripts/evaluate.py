"""
benchmarks/CarveFormer/scripts/evaluate.py
==========================================
CarveFormer evaluation entrypoint.

Thin wrapper around the shared DeepCarv ``Evaluator``: loads a trained
checkpoint, builds the test dataset via the framework ``FragmentDataset``,
rebuilds the model via the registry, and writes the standardized output set
(metrics.json, summary.json, predictions.csv, confusion_matrix.csv,
per_class_metrics.csv, classification_report.txt). No evaluation logic is
duplicated here.

Usage
-----
    python -m benchmarks.CarveFormer.scripts.evaluate \
        --config benchmarks/CarveFormer/configs/benchmark.yaml \
        --checkpoint benchmarks/CarveFormer/outputs/checkpoint_best.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from src.data.dataset import FragmentDataset, build_dataloader
from src.evaluation.evaluator import Evaluator
from src.models.registry import build_model
from src.training.checkpointing import load_checkpoint
from src.utils.paths import REPO_ROOT, ensure_dirs

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the benchmark YAML config."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p)


def run_evaluation(
    config: dict[str, Any],
    checkpoint: str | Path,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """Evaluate a trained CarveFormer checkpoint on the test split."""
    overrides = overrides or {}
    model_cfg = dict(config.get("model", {}))
    ds_cfg = dict(config.get("dataset", {}))
    train_cfg = dict(config.get("training", {}))
    path_cfg = dict(config.get("paths", {}))

    fragment_size = int(overrides.get("fragment_size") or ds_cfg.get("fragment_size") or 512)
    data_dir = _resolve(REPO_ROOT, str(overrides.get("data_dir", ds_cfg.get("root_dir", "data/FFT-75"))))
    run_outputs = _resolve(REPO_ROOT, str(path_cfg.get("run_outputs", "benchmarks/CarveFormer/outputs")))
    ensure_dirs(run_outputs)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    cache = bool(ds_cfg.get("cache", True))
    test_ds = FragmentDataset(root_dir=data_dir, split="test", fragment_size=fragment_size, cache=cache)
    batch_size = int(train_cfg.get("batch_size", 64))
    test_loader = build_dataloader(test_ds, batch_size=batch_size, shuffle=False)

    num_classes = test_ds.num_classes
    model_kwargs = {k: v for k, v in model_cfg.items() if k not in ("name",)}
    # Pretrained weights are irrelevant here — the checkpoint is loaded next.
    model_kwargs["pretrained"] = False
    model = build_model("carveformer", num_classes=num_classes, fragment_size=fragment_size, **model_kwargs)

    load_checkpoint(Path(checkpoint), model)
    device = train_cfg.get("device", "auto")
    evaluator = Evaluator(model, device=device)
    result = evaluator.evaluate_and_save(test_loader, run_outputs, run_name=f"carveformer_fft75_{fragment_size}")

    logger.info("CarveFormer evaluation complete: accuracy=%.4f | outputs=%s",
                result["metrics"].accuracy, run_outputs)
    return run_outputs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CarveFormer via the DeepCarv framework.")
    p.add_argument("--config", type=str, default="benchmarks/CarveFormer/configs/benchmark.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--fragment-size", type=int, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = load_config(args.config)
    overrides: dict[str, Any] = {}
    if args.fragment_size is not None:
        overrides["fragment_size"] = args.fragment_size
    if args.data_dir is not None:
        overrides["data_dir"] = args.data_dir
    run_evaluation(config, args.checkpoint, overrides)


if __name__ == "__main__":
    main()
