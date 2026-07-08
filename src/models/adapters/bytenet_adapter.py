"""
src/models/adapters/bytenet_adapter.py
--------------------------------------
Template/stub adapter for ByteNet.

This model is not implemented yet. The class exists so the model
registry API stays stable (src.models.registry.MODEL_REGISTRY["bytenet"]
resolves to something) while the real implementation is pending.

To implement this model later:
  1. Write/port the ByteNet architecture (as its own module, or as a
     submodule under benchmarks/ if it ships as a frozen reference impl).
  2. Replace the class body below with a real nn.Module wrapped to satisfy
     src.core.interfaces.FragmentClassifier (forward() -> log-probs
     [batch, num_classes]).
  3. Add configs/models/bytenet.yaml with real hyperparameters.
  4. Leave the registry key ("bytenet") and constructor signature stable
     so existing experiment configs referencing "bytenet" keep working.
"""

from __future__ import annotations

from typing import Any

from src.models.base import NotAvailableModel


class ByteNetAdapter(NotAvailableModel):
    """Stub adapter for ByteNet. Raises NotImplementedError on instantiation."""

    name = "bytenet"


def build_bytenet_adapter(num_classes: int, **kwargs: Any) -> "ByteNetAdapter":
    """Factory used by the model registry. Raises until ByteNet is implemented."""
    return ByteNetAdapter(num_classes=num_classes, **kwargs)
