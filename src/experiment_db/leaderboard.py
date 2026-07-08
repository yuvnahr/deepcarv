"""Leaderboard generation for DeepCarv benchmark records."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.experiment_db.experiment import ExperimentRecord
from src.experiment_db.storage import ExperimentDatabase
from src.utils.paths import OUTPUTS_DIR


LEADERBOARD_FIELDS = [
    "rank",
    "model",
    "dataset",
    "fragment_size",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "inference_latency_ms",
    "parameter_count",
    "run_id",
    "run_dir",
]


def sort_records(records: list[ExperimentRecord]) -> list[ExperimentRecord]:
    """Sort by accuracy, macro F1, then lower inference latency."""
    return sorted(
        records,
        key=lambda item: (
            item.accuracy if item.accuracy is not None else -1.0,
            item.macro_f1 if item.macro_f1 is not None else -1.0,
            -(item.inference_latency_ms if item.inference_latency_ms is not None else float("inf")),
        ),
        reverse=True,
    )


def leaderboard_rows(records: list[ExperimentRecord]) -> list[dict[str, object]]:
    """Return normalized leaderboard rows."""
    rows: list[dict[str, object]] = []
    for rank, record in enumerate(sort_records(records), 1):
        row = record.to_dict()
        row["rank"] = rank
        rows.append({field: row.get(field) for field in LEADERBOARD_FIELDS})
    return rows


def write_leaderboard(
    records: list[ExperimentRecord] | None = None,
    output_dir: Path | str | None = None,
    db_path: Path | str | None = None,
) -> dict[str, str]:
    """Write leaderboard.md, leaderboard.csv, and leaderboard.json."""
    out = Path(output_dir or OUTPUTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    records = records if records is not None else ExperimentDatabase(db_path).load()
    rows = leaderboard_rows(records)

    csv_path = out / "leaderboard.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    json_path = out / "leaderboard.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")

    md_path = out / "leaderboard.md"
    md_path.write_text(_to_markdown(rows) + "\n")
    return {"markdown": str(md_path), "csv": str(csv_path), "json": str(json_path)}


def _to_markdown(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "# DeepCarv Leaderboard\n\n_No benchmark runs registered yet._"
    lines = [
        "# DeepCarv Leaderboard",
        "",
        "| " + " | ".join(LEADERBOARD_FIELDS) + " |",
        "| " + " | ".join(["---"] * len(LEADERBOARD_FIELDS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in LEADERBOARD_FIELDS) + " |")
    return "\n".join(lines)
