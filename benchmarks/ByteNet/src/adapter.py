"""
benchmarks/ByteNet/src/adapter.py
-----------------------------------
Thin benchmark-local adapter for ByteNet.

This module is the bridge between the ByteNet model (model.py) and the
DeepCarv benchmark framework's FragmentClassifier interface.

The framework-facing adapter lives in:
    src/models/adapters/bytenet_adapter.py
That file imports and delegates to this one, keeping model code isolated
inside the benchmarks/ByteNet/ directory.

Public API
----------
ByteNetBenchmarkAdapter  — FragmentClassifier wrapping ByteNet
build_bytenet_benchmark_adapter  — factory called by the registry adapter
"""

from __future__ import annotations

from typing import Any

import torch

from src.core.interfaces import FragmentClassifier
from benchmarks.ByteNet.src.model import (
    ByteNet,
    build_bytenet_resnet,
    build_bytenet_former,
)


class ByteNetBenchmarkAdapter(FragmentClassifier):
    """Framework-facing wrapper around ByteNet.

    Satisfies the FragmentClassifier contract:
        forward(x: Tensor[B, fragment_size]) -> log_probs: Tensor[B, num_classes]
        loss(log_probs, targets) -> scalar  (inherited NLLLoss default)

    Parameters
    ----------
    num_classes : int
    variant : str
        'bytenet_resnet' or 'bytenet_former'
    fragment_size : int
        512 or 4096
    **model_kwargs :
        Forwarded to build_bytenet_resnet / build_bytenet_former.
    """

    name: str = "bytenet"

    def __init__(
        self,
        num_classes: int,
        variant: str = "bytenet_resnet",
        fragment_size: int = 512,
        **model_kwargs: Any,
    ) -> None:
        super().__init__(num_classes=num_classes)
        self.fragment_size = fragment_size

        if variant == "bytenet_resnet":
            self._model: ByteNet = build_bytenet_resnet(
                num_classes=num_classes,
                fragment_size=fragment_size,
                **model_kwargs,
            )
        elif variant == "bytenet_former":
            self._model = build_bytenet_former(
                num_classes=num_classes,
                fragment_size=fragment_size,
                **model_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown ByteNet variant '{variant}'. "
                "Choose 'bytenet_resnet' or 'bytenet_former'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities [B, num_classes]."""
        return self._model(x)

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self._model.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)


def build_bytenet_benchmark_adapter(
    num_classes: int,
    variant: str = "bytenet_resnet",
    fragment_size: int = 512,
    **model_kwargs: Any,
) -> ByteNetBenchmarkAdapter:
    """Factory used by src/models/adapters/bytenet_adapter.py."""
    return ByteNetBenchmarkAdapter(
        num_classes=num_classes,
        variant=variant,
        fragment_size=fragment_size,
        **model_kwargs,
    )
