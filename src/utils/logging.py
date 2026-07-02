"""
src/utils/logging.py
--------------------
Run-directory management, config snapshotting, and metric logging for the
ByteRCNN FFT-75 benchmark.

Typical usage
-------------
    from src.utils.logging import RunLogger
    logger = RunLogger(run_name="bytercnn_fft75", base_dir=LOGS_DIR)
    logger.save_config(config_dict)
    for epoch in range(epochs):
        logger.log_epoch(epoch, {"loss": 0.3, "val_acc": 0.82})
    logger.save_eval_summary(eval_metrics)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class RunLogger:
    """Creates a timestamped run directory and manages all log artefacts."""

    def __init__(
        self,
        run_name: str,
        base_dir: Path | str,
        config_path: Path | str | None = None,
    ) -> None:
        self.run_name = run_name
        self.base_dir = Path(base_dir)
        self.timestamp = _now_str()
        self.run_dir = self.base_dir / f"{run_name}_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Set up Python logging to file + stdout
        log_file = self.run_dir / "run.log"
        handlers: list[logging.Handler] = [
            logging.FileHandler(str(log_file)),
            logging.StreamHandler(sys.stdout),
        ]
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%H:%M:%S",
            handlers=handlers,
            force=True,
        )
        self.logger = logging.getLogger(run_name)
        self.logger.info("Run directory: %s", self.run_dir)

        # Epoch CSV state
        self._epoch_csv_path = self.run_dir / "metrics_per_epoch.csv"
        self._epoch_csv_initialized = False
        self._epoch_writer: csv.DictWriter | None = None
        self._epoch_file = None

        # Optionally snapshot the config YAML
        if config_path is not None:
            self.save_config_file(Path(config_path))

    # ------------------------------------------------------------------
    # Config snapshotting
    # ------------------------------------------------------------------

    def save_config(self, config: dict[str, Any]) -> None:
        """Dump a config dictionary as JSON into the run directory."""
        out = self.run_dir / "config_snapshot.json"
        with open(out, "w") as f:
            json.dump(config, f, indent=2, default=str)
        self.logger.info("Config snapshot → %s", out)

    def save_config_file(self, src: Path) -> None:
        """Copy an existing config file into the run directory."""
        if src.exists():
            dst = self.run_dir / src.name
            shutil.copy2(src, dst)
            self.logger.info("Config file copied → %s", dst)

    # ------------------------------------------------------------------
    # Epoch metrics
    # ------------------------------------------------------------------

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        """Append one row to metrics_per_epoch.csv and log to stdout."""
        row = {"epoch": epoch, **metrics}

        if not self._epoch_csv_initialized:
            self._epoch_file = open(self._epoch_csv_path, "w", newline="")
            self._epoch_writer = csv.DictWriter(
                self._epoch_file, fieldnames=list(row.keys())
            )
            self._epoch_writer.writeheader()
            self._epoch_csv_initialized = True

        self._epoch_writer.writerow(row)  # type: ignore[union-attr]
        self._epoch_file.flush()          # type: ignore[union-attr]

        metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.logger.info("Epoch %3d  %s", epoch, metrics_str)

    def close(self) -> None:
        """Flush and close the epoch CSV file."""
        if self._epoch_file is not None:
            self._epoch_file.close()

    # ------------------------------------------------------------------
    # Evaluation summary
    # ------------------------------------------------------------------

    def save_eval_summary(self, metrics: dict[str, Any]) -> None:
        """Write a human-readable evaluation summary text file and JSON."""
        # JSON
        json_path = self.run_dir / "eval_summary.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        # Plain-text summary
        txt_path = self.run_dir / "eval_summary.txt"
        lines = [
            "=" * 60,
            f"Evaluation Summary — {self.run_name}",
            f"Timestamp       : {self.timestamp}",
            "=" * 60,
        ]
        for k, v in metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k:<30} {v:.6f}")
            else:
                lines.append(f"  {k:<30} {v}")
        lines.append("=" * 60)
        txt_path.write_text("\n".join(lines) + "\n")

        self.logger.info("Eval summary → %s", txt_path)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def info(self, msg: str, *args: Any) -> None:
        self.logger.info(msg, *args)

    def warning(self, msg: str, *args: Any) -> None:
        self.logger.warning(msg, *args)

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def get_simple_logger(name: str = "deepcarv") -> logging.Logger:
    """Return a simple stdout logger (no run dir needed)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
