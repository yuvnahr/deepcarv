"""
src/models/adapters/deepcarv_adapter.py
---------------------------------------
Template/stub adapter for DeepCarv.

This model is not implemented yet. The class exists so the model
registry API stays stable (src.models.registry.MODEL_REGISTRY["deepcarv"]
resolves to something) while the real implementation is pending.

To implement this model later:
  1. Write/port the DeepCarv architecture (as its own module, or as a
     submodule under benchmarks/ if it ships as a frozen reference impl).
  2. Replace the class body below with a real nn.Module wrapped to satisfy
     src.core.interfaces.FragmentClassifier (forward() -> log-probs
     [batch, num_classes]).
  3. Add configs/models/deepcarv.yaml with real hyperparameters.
  4. Leave the registry key ("deepcarv") and constructor signature stable
     so existing experiment configs referencing "deepcarv" keep working.
"""

from __future__ import annotations

from typing import Any

from src.models.base import NotAvailableModel


class DeepCarvAdapter(NotAvailableModel):
    """Stub adapter for DeepCarv. Raises NotImplementedError on instantiation."""

    name = "deepcarv"


def build_deepcarv_adapter(num_classes: int, **kwargs: Any) -> "DeepCarvAdapter":
    """Factory used by the model registry. Raises until DeepCarv is implemented."""
    return DeepCarvAdapter(num_classes=num_classes, **kwargs)
