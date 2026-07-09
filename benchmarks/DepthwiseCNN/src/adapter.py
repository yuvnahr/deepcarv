"""
benchmarks/DepthwiseCNN/src/adapter.py
---------------------------------------
Adapter for the DepthwiseCNN model family to conform to the
`FragmentClassifier` interface required by the generic benchmark framework.
"""

from __future__ import annotations

from typing import Any
import torch

from src.core.interfaces import FragmentClassifier
from benchmarks.DepthwiseCNN.src.model import DepthwiseCNNModel

class DepthwiseCNNAdapter(FragmentClassifier):
    def __init__(self, num_classes: int, variant: str = 'dsc', **model_kwargs: Any) -> None:
        super().__init__(num_classes=num_classes)
        self.variant = variant
        self.name = f"depthwisecnn_{variant}"
        self._model = DepthwiseCNNModel(num_classes=num_classes, variant=variant, **model_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DepthwiseCNNModel returns log-softmax probabilities directly
        return self._model(x)

def build_depthwisecnn_adapter(num_classes: int, variant: str = 'dsc', **kwargs: Any) -> DepthwiseCNNAdapter:
    """Factory used by the model registry or dynamically in train.py."""
    return DepthwiseCNNAdapter(num_classes=num_classes, variant=variant, **kwargs)
