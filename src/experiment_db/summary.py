"""Markdown summaries for experiment records."""

from __future__ import annotations

from pathlib import Path

from src.experiment_db.leaderboard import leaderboard_rows
from src.experiment_db.storage import ExperimentDatabase


def write_experiment_summary(
    output_path: Path | str,
    db_path: Path | str | None = None,
) -> Path:
    """Write a concise markdown summary of registered benchmark runs."""
    rows = leaderboard_rows(ExperimentDatabase(db_path).load())
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Experiment Summary", ""]
    if not rows:
        lines.append("_No experiments registered._")
    else:
        lines.append(f"Registered runs: **{len(rows)}**")
        lines.append("")
        for row in rows[:10]:
            lines.append(
                f"- #{row['rank']} `{row['model']}` on `{row['dataset']}` "
                f"accuracy={row['accuracy']} macro_f1={row['macro_f1']}"
            )
    out.write_text("\n".join(lines) + "\n")
    return out
