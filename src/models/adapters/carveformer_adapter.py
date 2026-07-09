"""
src/models/adapters/carveformer_adapter.py
------------------------------------------
Template/stub adapter for CarveFormer.

This model is not implemented yet. The class exists so the model
registry API stays stable (src.models.registry.MODEL_REGISTRY["carveformer"]
resolves to something) while the real implementation is pending.

To implement this model later:
  1. Write/port the CarveFormer architecture (as its own module, or as a
     submodule under benchmarks/ if it ships as a frozen reference impl).
  2. Replace the class body below with a real nn.Module wrapped to satisfy
     src.core.interfaces.FragmentClassifier (forward() -> log-probs
     [batch, num_classes]).
  3. Add configs/models/carveformer.yaml with real hyperparameters.
  4. Leave the registry key ("carveformer") and constructor signature stable
     so existing experiment configs referencing "carveformer" keep working.
"""

from __future__ import annotations

from typing import Any

from src.models.base import NotAvailableModel


class CarveFormerAdapter(NotAvailableModel):
    """Stub adapter for CarveFormer. Raises NotImplementedError on instantiation."""

    name = "carveformer"


def build_carveformer_adapter(num_classes: int, **kwargs: Any) -> "CarveFormerAdapter":
    """Factory used by the model registry. Raises until CarveFormer is implemented."""
    return CarveFormerAdapter(num_classes=num_classes, **kwargs)
