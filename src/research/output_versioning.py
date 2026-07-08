"""Versioned output directory helpers for benchmark runs."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

from src.utils.paths import OUTPUTS_DIR

logger = logging.getLogger(__name__)


def make_run_dir_name(model: str, dataset: str, fragment_size: int | None = None) -> str:
    """Build a stable non-overwriting output directory name."""
    date = datetime.now().strftime("%Y-%m-%d")
    safe_dataset = dataset.lower().replace("/", "_").replace("-", "")
    suffix = f"_{fragment_size}" if fragment_size is not None else ""
    return f"{date}_{model}_{safe_dataset}{suffix}"


def create_versioned_run_dir(
    model: str,
    dataset: str,
    fragment_size: int | None = None,
    base_dir: Path | str | None = None,
) -> Path:
    """Create a versioned run directory without overwriting prior outputs."""
    base = Path(base_dir or OUTPUTS_DIR)
    base.mkdir(parents=True, exist_ok=True)
    stem = make_run_dir_name(model, dataset, fragment_size)
    candidate = base / stem
    index = 1
    while candidate.exists():
        index += 1
        candidate = base / f"{stem}_{index:02d}"
    candidate.mkdir(parents=True)
    update_output_pointers(base, candidate)
    return candidate


def update_output_pointers(base_dir: Path | str, run_dir: Path | str, is_best: bool = False) -> None:
    """Update latest/history metadata and symlinks where supported."""
    base = Path(base_dir)
    run_path = Path(run_dir)
    history_path = base / "history.json"
    history = []
    if history_path.exists():
        history = json.loads(history_path.read_text())
    history.append(str(run_path))
    history_path.write_text(json.dumps(history, indent=2) + "\n")
    _replace_pointer(base / "latest", run_path)
    if is_best:
        _replace_pointer(base / "best", run_path)


def _replace_pointer(pointer: Path, target: Path) -> None:
    if pointer.exists() or pointer.is_symlink():
        if pointer.is_dir() and not pointer.is_symlink():
            shutil.rmtree(pointer)
        else:
            pointer.unlink()
    try:
        pointer.symlink_to(target, target_is_directory=True)
    except OSError:
        pointer.write_text(str(target) + "\n")
        logger.debug("Symlink unavailable; wrote pointer file %s", pointer)
