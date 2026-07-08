"""
src/core/interfaces.py
-----------------------
Abstract interfaces that decouple the generic trainer / evaluator from any
specific model implementation (ByteRCNN, CarveFormer, ByteNet, DeepCarv, ...).

Every adapter in src/models/adapters/ must produce an object that satisfies
`FragmentClassifier`. Every dataset used by the framework must satisfy
`FragmentDatasetProtocol`. This is the *only* contract the trainer and
evaluator are allowed to depend on — no model-specific branching anywhere
else in the framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class FragmentDatasetProtocol(Protocol):
    """Minimal contract a dataset must satisfy to be used by the framework."""

    num_classes: int
    fragment_size: int

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]: ...


class FragmentClassifier(nn.Module, ABC):
    """Base class every model adapter must inherit from.

    The generic trainer/evaluator only ever call:
        - forward(x) -> log-probabilities [B, num_classes]
        - loss(log_probs, y) -> scalar loss tensor
        - num_classes (attribute)
        - name (attribute, for logging/reporting)

    Concrete adapters are responsible for translating this contract into
    whatever the wrapped model actually expects.
    """

    #: Human-readable model name used in logs, reports, and registry lookups.
    name: str = "unnamed_model"

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities of shape [batch, num_classes]."""
        raise NotImplementedError

    def loss(self, log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Default loss: NLLLoss over log-probabilities.

        Adapters may override this if the wrapped model produces raw logits
        instead of log-probabilities (in which case also override forward
        to keep the contract, or override both consistently).
        """
        return nn.functional.nll_loss(log_probs, targets)

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices [batch]."""
        with torch.no_grad():
            return self.forward(x).argmax(dim=1)


@dataclass
class BatchResult:
    """Standardized result of a single train/eval batch step."""

    loss: float
    logits: torch.Tensor | None = None
    predictions: torch.Tensor | None = None
    targets: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpochResult:
    """Standardized result of a full epoch (train or val)."""

    loss: float
    accuracy: float
    extra: dict[str, float] = field(default_factory=dict)
