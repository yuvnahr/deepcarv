"""
src/models/adapters/bytercnn_adapter.py
------------------------------------------
Thin adapter around the existing ByteRCNN PyTorch implementation
(src/models/bytercnn_wrapper.py), exposing it through the framework's
`FragmentClassifier` interface.

This adapter does NOT reimplement ByteRCNN and does NOT touch anything
under benchmarks/ByteRCNN/ (the frozen reference submodule). It imports
the existing `build_bytercnn` factory directly and delegates to it —
the benchmark architecture and hyperparameters are preserved exactly.

If a future benchmark run needs the *original* frozen submodule
implementation directly (e.g. for cross-checking), do that import in a
separate, explicitly-named adapter (e.g. bytercnn_reference_adapter.py)
rather than changing this one, so the default registry entry keeps
using the maintained PyTorch model.
"""

from __future__ import annotations

from typing import Any

import torch

from src.core.interfaces import FragmentClassifier
from src.models.bytercnn_wrapper import build_bytercnn


class ByteRCNNAdapter(FragmentClassifier):
    """Framework-facing wrapper around `build_bytercnn`.

    Parameters
    ----------
    num_classes : int
        Number of output classes (75 for the frozen FFT-75 S1 benchmark).
    **model_kwargs :
        Forwarded to `build_bytercnn`. In benchmark runs these should be
        left at defaults — the frozen architecture is defined in
        bytercnn_wrapper.py, not here.
    """

    name = "bytercnn"

    def __init__(self, num_classes: int, **model_kwargs: Any) -> None:
        super().__init__(num_classes=num_classes)
        self._model = build_bytercnn(num_classes=num_classes, **model_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # build_bytercnn's forward already returns log-softmax probabilities,
        # matching FragmentClassifier's contract exactly.
        return self._model(x)

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self._model.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)


def build_bytercnn_adapter(num_classes: int, **kwargs: Any) -> ByteRCNNAdapter:
    """Factory used by the model registry."""
    return ByteRCNNAdapter(num_classes=num_classes, **kwargs)
