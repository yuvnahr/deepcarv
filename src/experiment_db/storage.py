"""File-backed storage for benchmark experiment records."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from src.experiment_db.experiment import ExperimentRecord
from src.utils.paths import OUTPUTS_DIR

logger = logging.getLogger(__name__)


class ExperimentDatabase:
    """A small JSONL experiment database stored under outputs/."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path or OUTPUTS_DIR / "experiment_db" / "experiments.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, record: ExperimentRecord) -> None:
        """Append an experiment record."""
        with open(self.path, "a") as f:
            f.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        logger.info("Registered experiment %s in %s", record.run_id, self.path)

    def load(self) -> list[ExperimentRecord]:
        """Load all records, skipping blank lines."""
        if not self.path.exists():
            return []
        records: list[ExperimentRecord] = []
        with open(self.path) as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(ExperimentRecord.from_dict(json.loads(line)))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed experiment DB line %d: %s", line_number, exc)
        return records

    def replace_all(self, records: Iterable[ExperimentRecord]) -> None:
        """Rewrite the database with records."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            for record in records:
                f.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    def export_json(self, path: Path | str | None = None) -> Path:
        """Export records to a JSON array."""
        out = Path(path or self.path.with_suffix(".json"))
        out.write_text(json.dumps([asdict(record) for record in self.load()], indent=2) + "\n")
        return out


def register_experiment(
    record: ExperimentRecord,
    db_path: Path | str | None = None,
) -> ExperimentRecord:
    """Register a run in the file-backed experiment database."""
    ExperimentDatabase(db_path).add(record)
    return record
