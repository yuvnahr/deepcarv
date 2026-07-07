"""
src/training/checkpointing.py
--------------------------------
Model-agnostic checkpoint save/load/resume. Contains no ByteRCNN-specific
(or any model-specific) logic — it only ever touches state_dicts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a full checkpoint (model + optimizer + metadata)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load a checkpoint into `model` (and optionally `optimizer`) in place.

    Returns the raw checkpoint dict so callers can read epoch/metrics/extras.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def resume_epoch(ckpt: dict[str, Any]) -> int:
    """Return the epoch to resume FROM (i.e. next epoch to run)."""
    return int(ckpt.get("epoch", 0)) + 1
