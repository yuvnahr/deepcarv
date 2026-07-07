"""
src/core/runner.py
---------------------
The single, unified entrypoint for running any registered model through
the benchmark framework end-to-end:

    config -> dataset_factory -> registry -> trainer -> evaluator -> plots

This is the ONLY supported way to run a benchmark experiment. There is no
per-model training/evaluation script — adding a new model means adding an
adapter + registry entry + config, then running this same runner.

Usage
-----
    python -m src.core.runner --experiment configs/experiments/bytercnn_fft75.yaml
    python -m src.core.runner --experiment bytercnn_fft75 --override training.epochs=2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.core.experiment_manager import ExperimentManager
from src.data.dataset_factory import build_dataloaders, build_datasets
from src.evaluation.evaluator import Evaluator
from src.models.registry import build_model
from src.training.trainer import Trainer, TrainerConfig
from src.utils.config import load_experiment_config
from src.utils.paths import OUTPUTS_DIR
from src.visualization.confusion import plot_confusion_matrix
from src.visualization.plots import plot_accuracy_curve, plot_loss_curve, plot_lr_curve

logger = logging.getLogger(__name__)


def run_experiment(
    experiment_name_or_path: str,
    overrides: dict | None = None,
    base_dir: Path | None = None,
) -> Path:
    """Run a full train + evaluate cycle for one experiment config.

    Returns the run directory containing all standardized outputs.
    """
    config = load_experiment_config(experiment_name_or_path, overrides=overrides)

    exp = ExperimentManager(config, base_dir=base_dir or OUTPUTS_DIR)
    run_dir = exp.setup()

    try:
        # ---- Data --------------------------------------------------------
        datasets = build_datasets(config.dataset)
        loaders = build_dataloaders(datasets, config.training, config.evaluation)
        num_classes = datasets["train"].num_classes
        logger.info("Dataset ready: num_classes=%d, train=%d val=%d test=%d",
                    num_classes, len(datasets["train"]), len(datasets["val"]), len(datasets["test"]))

        # ---- Model (via registry only) -----------------------------------
        model_name = config.model.get("name", config.model.get("model_name"))
        if not model_name:
            raise ValueError("model config must specify a 'name' key matching a registry entry.")
        model_kwargs = {k: v for k, v in config.model.items() if k not in ("name", "model_name")}
        model = build_model(model_name, num_classes=num_classes, **model_kwargs)
        logger.info("Model '%s' built: %d trainable parameters", model_name, model.num_parameters())

        # ---- Train ---------------------------------------------------------
        trainer_cfg = TrainerConfig.from_dict(config.training)
        trainer = Trainer(model, trainer_cfg, run_dir)
        history = trainer.fit(loaders["train"], loaders["val"])

        # ---- Plots -----------------------------------------------------------
        plot_loss_curve(history.train_loss, history.val_loss, run_dir / "loss_curve.png")
        plot_accuracy_curve(history.train_acc, history.val_acc, run_dir / "accuracy_curve.png")
        plot_lr_curve(history.lr, run_dir / "lr_curve.png")

        # ---- Evaluate on test split (using the best checkpoint) ---------------
        from src.training.checkpointing import load_checkpoint
        load_checkpoint(trainer.best_ckpt_path, model, map_location=trainer.device)

        evaluator = Evaluator(model, device=trainer_cfg.device)
        result = evaluator.evaluate_and_save(loaders["test"], run_dir, run_name=exp.run_name)

        plot_confusion_matrix(result["metrics"].confusion, run_dir / "confusion_matrix.png")

        exp.finalize(extra_summary={
            "model": model_name,
            "num_classes": num_classes,
            "n_train": len(datasets["train"]),
            "n_val": len(datasets["val"]),
            "n_test": len(datasets["test"]),
            "n_parameters": model.num_parameters(),
            "best_epoch": trainer.best_epoch,
            "best_monitor_value": trainer._best_metric,
        })
        logger.info("Run complete: %s", run_dir)
        return run_dir

    except Exception:
        exp.finalize()
        raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepCarv unified benchmark runner.")
    p.add_argument("--experiment", type=str, required=True,
                    help="Experiment name (configs/experiments/<name>.yaml) or direct path.")
    p.add_argument("--override", action="append", default=[],
                    help="Dotted override, e.g. --override training.epochs=2 (repeatable).")
    return p.parse_args(argv)


def _parse_overrides(raw: list[str]) -> dict:
    overrides = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Invalid --override '{item}', expected key.path=value")
        key, value = item.split("=", 1)
        # Best-effort type coercion
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                continue
        else:
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
        overrides[key] = value
    return overrides


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    args = _parse_args(argv)
    overrides = _parse_overrides(args.override)
    run_experiment(args.experiment, overrides=overrides or None)


if __name__ == "__main__":
    main()
