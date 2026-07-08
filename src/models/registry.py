"""
src/models/registry.py
------------------------
Central model registry. This is the ONLY place the trainer/evaluator use
to instantiate a model. Adding a new model to the framework means:

    1. Implement its adapter in src/models/adapters/<name>_adapter.py
    2. Add one line here
    3. Add configs/models/<name>.yaml

No other framework code should ever import a specific adapter directly.
"""

from __future__ import annotations

from typing import Any, Callable

from src.core.interfaces import FragmentClassifier
from src.models.adapters.bytenet_adapter import build_bytenet_adapter
from src.models.adapters.bytercnn_adapter import build_bytercnn_adapter
from src.models.adapters.carveformer_adapter import build_carveformer_adapter
from src.models.adapters.deepcarv_adapter import build_deepcarv_adapter

ModelFactory = Callable[..., FragmentClassifier]


class ModelRegistryError(RuntimeError):
    """Raised for unknown model names or registry misuse."""


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
# Maps model name (as used in configs/models/<name>.yaml and experiment
# configs) -> a zero-argument-friendly factory function.
#
# Implemented:   bytercnn
# Stub/template: carveformer, bytenet, deepcarv (raise NotImplementedError
#                 on instantiation, but the key resolves — see
#                 src/models/base.py:NotAvailableModel)
MODEL_REGISTRY: dict[str, ModelFactory] = {
    "bytercnn": build_bytercnn_adapter,
    "carveformer": build_carveformer_adapter,
    "bytenet": build_bytenet_adapter,
    "deepcarv": build_deepcarv_adapter,
}


def register_model(name: str, factory: ModelFactory, overwrite: bool = False) -> None:
    """Register a new model factory at runtime (e.g. from a plugin/notebook)."""
    if name in MODEL_REGISTRY and not overwrite:
        raise ModelRegistryError(
            f"Model '{name}' is already registered. Pass overwrite=True to replace it."
        )
    MODEL_REGISTRY[name] = factory


def is_available(name: str) -> bool:
    """Whether a model name is registered (does not guarantee it's implemented —
    stub models are registered but raise on instantiation)."""
    return name in MODEL_REGISTRY


def list_models() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())


def build_model(name: str, num_classes: int, **model_kwargs: Any) -> FragmentClassifier:
    """Instantiate a registered model by name.

    Raises
    ------
    ModelRegistryError
        If `name` is not a registered key at all.
    NotImplementedError
        If `name` is registered but only as a stub/template (propagated
        from NotAvailableModel).
    """
    if name not in MODEL_REGISTRY:
        raise ModelRegistryError(
            f"Unknown model '{name}'. Registered models: {list_models()}"
        )
    factory = MODEL_REGISTRY[name]
    return factory(num_classes=num_classes, **model_kwargs)
