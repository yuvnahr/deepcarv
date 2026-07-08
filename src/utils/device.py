"""
src/utils/device.py
--------------------
Centralized device selection so no other module hardcodes "cuda"/"cpu".
"""

from __future__ import annotations

import torch


def get_device(preferred: str | None = None) -> torch.device:
    """Resolve the torch device to use.

    Parameters
    ----------
    preferred : str, optional
        "cuda", "cpu", or "auto"/None. If "cuda" is requested but
        unavailable, falls back to CPU with no error (framework should
        never crash simply because a GPU isn't present).
    """
    if preferred in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preferred == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preferred)


def gpu_name(device: torch.device) -> str | None:
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(device.index or 0)
    return None


def peak_memory_mb(device: torch.device) -> float | None:
    """Peak allocated GPU memory in MB since the last reset, or None on CPU."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
