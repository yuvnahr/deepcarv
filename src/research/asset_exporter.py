"""Research asset exporter for paper-ready benchmark artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


FIGURE_NAMES = [
    "loss_curve",
    "accuracy_curve",
    "lr_curve",
    "confusion_matrix",
    "class_distribution",
    "roc_curve",
    "precision_recall_curve",
]


def export_research_assets(run_dir: Path | str, output_dir: Path | str | None = None) -> Path:
    """Copy standard tables, figures, CSV, and JSON files into paper_assets/."""
    run = Path(run_dir)
    root = Path(output_dir or run / "paper_assets")
    for child in ("tables", "figures", "csv", "json"):
        (root / child).mkdir(parents=True, exist_ok=True)

    for path in run.glob("*.csv"):
        shutil.copy2(path, root / "csv" / path.name)
    for path in run.glob("*.json"):
        shutil.copy2(path, root / "json" / path.name)
    for path in run.glob("*.md"):
        shutil.copy2(path, root / "tables" / path.name)
    for stem in FIGURE_NAMES:
        for suffix in (".png", ".pdf", ".svg"):
            path = run / f"{stem}{suffix}"
            if path.exists():
                shutil.copy2(path, root / "figures" / path.name)

    manifest = {
        "source_run_dir": str(run),
        "figures": sorted(path.name for path in (root / "figures").glob("*")),
        "csv": sorted(path.name for path in (root / "csv").glob("*")),
        "json": sorted(path.name for path in (root / "json").glob("*")),
        "tables": sorted(path.name for path in (root / "tables").glob("*")),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root
