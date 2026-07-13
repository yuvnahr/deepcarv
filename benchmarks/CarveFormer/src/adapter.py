"""
benchmarks/CarveFormer/src/adapter.py
=====================================
Framework adapter for CarveFormer.

This is the thin layer that plugs the framework-independent model in
``benchmarks/CarveFormer/src/model.py`` into the DeepCarv benchmark framework's
``FragmentClassifier`` contract (``src.core.interfaces``). The generic
trainer/evaluator only ever call ``forward()``, ``loss()``, ``.num_classes``
and ``.name`` — this adapter satisfies exactly that and nothing more.

Design mirrors ``src/models/adapters/bytercnn_adapter.py``: the adapter wraps
the model factory, keeps the wrapped model as a submodule, and forwards the
call. The model already returns ``log_softmax`` so the framework's default
``nll_loss`` applies unchanged.

Fragment size, class count, backbone, and pretraining are all configuration
driven and passed straight through to :func:`build_carveformer`.
"""

from __future__ import annotations

from typing import Any

import torch

from src.core.interfaces import FragmentClassifier

from .model import CarveFormer, build_carveformer


class CarveFormerAdapter(FragmentClassifier):
    """Framework-facing wrapper around :class:`CarveFormer`.

    Parameters
    ----------
    num_classes:
        Number of output classes for the target FFT-75 scenario.
    fragment_size:
        Byte-fragment length (512 or 4096). Passed through to the model so the
        512/4096 switch is configuration-only.
    **model_kwargs:
        Additional keyword arguments forwarded to :func:`build_carveformer`
        (e.g. ``pretrained``, ``backbone``, ``embed_dim``, ``grid_hw``,
        ``drop_rate``).
    """

    name = "carveformer"

    def __init__(
        self,
        num_classes: int,
        fragment_size: int = 512,
        **model_kwargs: Any,
    ) -> None:
        super().__init__(num_classes=num_classes)
        self.fragment_size = fragment_size
        self._model: CarveFormer = build_carveformer(
            num_classes=num_classes,
            fragment_size=fragment_size,
            **model_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities ``[B, num_classes]``.

        The wrapped model already applies ``log_softmax``, matching the
        framework contract (default loss is ``nll_loss``).
        """
        return self._model(x)

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return the wrapped model's parameter count."""
        return self._model.num_parameters(trainable_only=trainable_only)


def build_carveformer_adapter(
    num_classes: int,
    fragment_size: int = 512,
    **kwargs: Any,
) -> CarveFormerAdapter:
    """Factory used by the DeepCarv model registry.

    Registered under the key ``"carveformer"`` (see
    ``src/models/adapters/carveformer_adapter.py`` in the parent framework,
    which now delegates here).
    """
    return CarveFormerAdapter(
        num_classes=num_classes,
        fragment_size=fragment_size,
        **kwargs,
    )
