"""
benchmarks/DepthwiseCNN/src/adapter.py
-----------------------------------------
Benchmark-local adapter for DepthwiseCNN.

This module bridges the DepthwiseCNN model (model.py) and the DeepCarv
benchmark framework's FragmentClassifier interface.

The framework-facing adapter lives in:
    src/models/adapters/depthwisecnn_adapter.py
That file imports and delegates to this one, keeping all DepthwiseCNN model
code isolated inside the benchmarks/DepthwiseCNN/ directory.

Public API
----------
DepthwiseCNNBenchmarkAdapter  — FragmentClassifier wrapping DepthwiseCNN
build_depthwisecnn_benchmark_adapter  — factory called by the registry adapter
"""

from __future__ import annotations

from typing import Any

import torch

from src.core.interfaces import FragmentClassifier
from benchmarks.DepthwiseCNN.src.model import DepthwiseCNN, build_depthwisecnn


class DepthwiseCNNBenchmarkAdapter(FragmentClassifier):
    """Framework-facing wrapper around DepthwiseCNN.

    Satisfies the FragmentClassifier contract:
        forward(x: Tensor[B, fragment_size]) -> log_probs: Tensor[B, num_classes]
        loss(log_probs, targets) -> scalar  (inherited NLLLoss default)

    Parameters
    ----------
    num_classes : int
        Number of output classes (75 for FFT-75).
    variant : str
        'dsc', 'dsc_se', or 'm_dsc'.  Selects the architectural variant.
    fragment_size : int
        Input fragment length in bytes (512 or 4096).
    embed_dim : int
        Embedding dimensionality.  ASSUMPTION: 64.
    channels : list[int] | None
        Channel widths for each block stage.  ASSUMPTION: [64, 128, 256, 256].
    kernel_size : int
        Depthwise kernel size (DSC/DSC-SE only).  ASSUMPTION: 3.
    se_reduction : int
        SE block reduction ratio (DSC-SE only).  ASSUMPTION: 16.
    p_dropout : float
        Dropout probability before the classifier.  ASSUMPTION: 0.5.
    """

    name: str = "depthwisecnn"

    def __init__(
        self,
        num_classes: int,
        variant: str = "dsc",
        fragment_size: int = 512,
        embed_dim: int = 64,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        se_reduction: int = 16,
        p_dropout: float = 0.5,
        **_ignored: Any,
    ) -> None:
        super().__init__(num_classes=num_classes)
        self.fragment_size = fragment_size
        self._model: DepthwiseCNN = build_depthwisecnn(
            num_classes=num_classes,
            fragment_size=fragment_size,
            variant=variant,
            embed_dim=embed_dim,
            channels=list(channels) if channels is not None else None,
            kernel_size=kernel_size,
            se_reduction=se_reduction,
            p_dropout=p_dropout,
        )
        # Expose variant for logging
        self.variant = variant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities [B, num_classes]."""
        return self._model(x)

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self._model.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)


def build_depthwisecnn_benchmark_adapter(
    num_classes: int,
    variant: str = "dsc",
    fragment_size: int = 512,
    **model_kwargs: Any,
) -> DepthwiseCNNBenchmarkAdapter:
    """Factory used by src/models/adapters/depthwisecnn_adapter.py."""
    return DepthwiseCNNBenchmarkAdapter(
        num_classes=num_classes,
        variant=variant,
        fragment_size=fragment_size,
        **model_kwargs,
    )
