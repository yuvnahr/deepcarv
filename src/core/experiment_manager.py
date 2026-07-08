"""
src/core/experiment_manager.py
----------------------------------
Makes every benchmark run reproducible by owning the run directory and
everything that must be snapshotted into it:

outputs/<run_name>/
    config.yaml
    metrics.json            (written by the evaluator)
    summary.json            (written by the evaluator)
    train.log
    git_commit.txt
    environment.json
    checkpoint_best.pt      (written by the trainer)
    checkpoint_last.pt      (written by the trainer)
    loss_curve.png
    accuracy_curve.png
    confusion_matrix.png
    per_class_metrics.csv   (written by the evaluator)
    classification_report.txt (written by the evaluator)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.experiment_db.experiment import ExperimentRecord
from src.experiment_db.leaderboard import write_leaderboard
from src.experiment_db.storage import register_experiment
from src.research.benchmark_report import generate_benchmark_report
from src.research.metadata import collect_benchmark_metadata, write_benchmark_metadata
from src.research.output_versioning import update_output_pointers
from src.utils.config import ExperimentConfig
from src.utils.environment import collect_environment_info, get_git_commit
from src.utils.paths import OUTPUTS_DIR, REPO_ROOT
from src.utils.seed import set_seed


class ExperimentManager:
    """Owns the run directory for a single benchmark experiment.

    Usage
    -----
        exp = ExperimentManager(config, base_dir=OUTPUTS_DIR)
        exp.setup()          # creates run_dir, sets seed, snapshots everything
        ... trainer/evaluator write into exp.run_dir ...
        exp.finalize()       # closes logging handlers
    """

    def __init__(
        self,
        config: ExperimentConfig,
        base_dir: Path | None = None,
        run_name: str | None = None,
        timestamp_run: bool = True,
    ) -> None:
        self.config = config
        self.base_dir = Path(base_dir or OUTPUTS_DIR)

        name = run_name or config.name
        if timestamp_run:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"{name}_{ts}"
        self.run_name = name
        self.run_dir = self.base_dir / self.run_name
        self._started_at: float | None = None

        self._log_handler: logging.Handler | None = None

    def setup(self) -> Path:
        """Create the run directory and write all reproducibility artifacts."""
        self._started_at = time.time()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        seed = int(self.config.training.get("seed", 42))
        set_seed(seed)

        # config.yaml
        self.config.save(self.run_dir / "config.yaml")

        # git_commit.txt
        (self.run_dir / "git_commit.txt").write_text(get_git_commit() + "\n")

        # environment.json
        env_info = collect_environment_info()
        with open(self.run_dir / "environment.json", "w") as f:
            json.dump(env_info, f, indent=2)

        self._write_benchmark_metadata()

        # train.log — attach a file handler to the root logger for this run
        log_path = self.run_dir / "train.log"
        handler = logging.FileHandler(str(log_path))
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s: %(message)s")
        )
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        has_stream_handler = any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)
        if not root_logger.handlers or not has_stream_handler:
            stream = logging.StreamHandler(sys.stdout)
            stream.setFormatter(
                logging.Formatter(
                    "%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            root_logger.addHandler(stream)
        root_logger.setLevel(logging.INFO)
        self._log_handler = handler

        logging.getLogger(__name__).info(
            "Experiment '%s' initialized at %s (seed=%d)", self.run_name, self.run_dir, seed
        )
        return self.run_dir

    def finalize(self, extra_summary: dict[str, Any] | None = None) -> None:
        """Detach logging handlers and optionally merge extra fields into summary.json."""
        if extra_summary:
            summary_path = self.run_dir / "summary.json"
            existing: dict[str, Any] = {}
            if summary_path.exists():
                with open(summary_path) as f:
                    existing = json.load(f)
            existing.update(extra_summary)
            with open(summary_path, "w") as f:
                json.dump(existing, f, indent=2)

        self._write_benchmark_metadata()
        self._register_run_if_possible()
        self._write_benchmark_report_if_possible()
        update_output_pointers(self.base_dir, self.run_dir)

        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler.close()
            self._log_handler = None

    def __enter__(self) -> "ExperimentManager":
        self.setup()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.finalize()

    def _write_benchmark_metadata(self) -> None:
        dataset_root = self.config.dataset.get("root_dir")
        if dataset_root is not None:
            dataset_root = Path(dataset_root)
            if not dataset_root.is_absolute():
                dataset_root = REPO_ROOT / dataset_root
        fragment_size = self.config.dataset.get("fragment_size")
        metadata = collect_benchmark_metadata(
            dataset_root=dataset_root,
            fragment_size=int(fragment_size) if fragment_size is not None else None,
            dataset_version=str(self.config.dataset.get("version", "unknown")),
            started_at=self._started_at,
        )
        write_benchmark_metadata(self.run_dir / "benchmark_metadata.json", metadata)

    def _register_run_if_possible(self) -> None:
        summary_path = self.run_dir / "summary.json"
        if not summary_path.exists():
            return
        try:
            summary = json.loads(summary_path.read_text())
            metadata = json.loads((self.run_dir / "benchmark_metadata.json").read_text())
            model = str(summary.get("model") or self.config.model.get("name", "unknown"))
            dataset = str(self.config.dataset.get("name", "unknown"))
            checkpoint = self.run_dir / "checkpoint_best.pt"
            record = ExperimentRecord.create(
                model=model,
                dataset=dataset,
                fragment_size=self.config.dataset.get("fragment_size"),
                seed=self.config.training.get("seed"),
                accuracy=summary.get("accuracy"),
                macro_f1=summary.get("macro_f1"),
                weighted_f1=summary.get("weighted_f1"),
                inference_latency_ms=summary.get("latency_ms_per_sample"),
                gpu=summary.get("gpu_name") or metadata.get("gpu"),
                training_time_s=metadata.get("runtime_s"),
                parameter_count=summary.get("n_parameters"),
                checkpoint=str(checkpoint) if checkpoint.exists() else None,
                git_commit=metadata.get("git_hash"),
                run_dir=self.run_dir,
                metadata={"run_name": self.run_name},
            )
            register_experiment(record)
            write_leaderboard(output_dir=self.base_dir)
        except Exception as exc:  # pragma: no cover - non-critical side effect.
            logging.getLogger(__name__).warning("Experiment registration skipped: %s", exc)

    def _write_benchmark_report_if_possible(self) -> None:
        try:
            generate_benchmark_report(self.run_dir)
        except Exception as exc:  # pragma: no cover - non-critical side effect.
            logging.getLogger(__name__).warning("Benchmark report generation skipped: %s", exc)
