"""
benchmarks/DepthwiseCNN/src/adapter.py
---------------------------------------
Thin adapter that wraps :class:`DepthwiseCNNModel` and exposes it through
the generic DeepCarv :class:`FragmentClassifier` interface.

This module is the **only** file the framework ever imports from this
benchmark package.  The model registry maps the key ``"depthwisecnn"`` to
:func:`build_depthwisecnn_adapter`, the sole public factory here.

Framework contract surface
--------------------------
The generic :class:`~src.training.trainer.Trainer` and
:class:`~src.evaluation.evaluator.Evaluator` call **exactly** the following
on any :class:`~src.core.interfaces.FragmentClassifier`:

    - ``model(x)``                → log-probabilities ``[B, num_classes]``
    - ``model.loss(lp, y)``       → scalar NLLLoss  (inherited default)
    - ``model.num_classes``       → int attribute   (set by base class)
    - ``model.name``              → str attribute   (set below)
    - ``model.num_parameters()``  → int             (overridden below)
    - ``model.predict(x)``        → ``[B]`` argmax  (inherited default)
    - ``model.train(bool)``       → from nn.Module
    - ``model.to(device)``        → from nn.Module
    - ``model.state_dict()``      → from nn.Module  (for checkpointing)
    - ``model.load_state_dict()`` → from nn.Module  (for resume)
    - ``model.parameters()``      → from nn.Module  (for optimizer)

No other framework code imports from this benchmark package; all coupling
goes through the registry key ``"depthwisecnn"``.

Design constraints
------------------
- **No architecture logic here.** Every structural decision lives in
  ``model.py``.
- **No hidden side effects.** The adapter constructor only builds the inner
  model and sets metadata attributes; it does not write files, start
  threads, or modify global state.
- **Config-driven variant.** The ``variant`` kwarg is forwarded from the
  benchmark YAML via ``build_model(..., variant="dsc-se")``; the adapter
  does not hard-code any variant.

Usage (via registry — preferred)
---------------------------------
    from src.models.registry import build_model

    model = build_model("depthwisecnn", num_classes=75, variant="dsc-se")
    log_probs = model(x)   # x: [B, L] int64 in [0, 255]

Usage (direct — notebooks / standalone scripts)
------------------------------------------------
    from benchmarks.DepthwiseCNN.src.adapter import build_depthwisecnn_adapter

    model = build_depthwisecnn_adapter(num_classes=75, variant="m-dsc")
"""

from __future__ import annotations

from typing import Any

import torch

from src.core.interfaces import FragmentClassifier
from benchmarks.DepthwiseCNN.src.model import (
    DepthwiseCNNModel,
    Variant,
    build_depthwisecnn,
)


class DepthwiseCNNAdapter(FragmentClassifier):
    """Framework-facing wrapper around :class:`DepthwiseCNNModel`.

    Inherits :class:`~src.core.interfaces.FragmentClassifier` which itself
    inherits ``nn.Module``.  The adapter adds **no** learnable parameters
    of its own — all parameters belong to ``self._model``.

    Satisfying the framework contract
    ----------------------------------
    * ``forward`` — delegates to ``self._model``, which returns
      log-softmax probabilities compatible with NLLLoss.
    * ``loss``    — **not** overridden; the base-class default
      (``F.nll_loss``) is correct because ``forward`` already returns
      log-probabilities.
    * ``predict`` — **not** overridden; the base-class ``argmax`` over
      log-probabilities is correct.
    * ``num_parameters`` — overridden to count the inner model's parameters
      (excluding the zero-parameter adapter wrapper overhead).

    Parameters
    ----------
    num_classes:
        Number of output classes.  Stored as ``self.num_classes`` by the
        base-class ``__init__`` and used by the evaluator.
    variant:
        Architecture variant — ``"dsc"``, ``"dsc-se"``, or ``"m-dsc"``.
        Determines normalisation, activation, first-conv style, SE gates,
        and head dropout.  See :class:`DepthwiseCNNModel` for details.
    **model_kwargs:
        Passed through to :func:`build_depthwisecnn`.  The only recognised
        extra key is ``dropout_p`` (default ``0.2``, M-DSC head only).
    """

    def __init__(
        self,
        num_classes: int,
        variant: Variant = "dsc",
        **model_kwargs: Any,
    ) -> None:
        # FragmentClassifier.__init__ sets self.num_classes = num_classes
        super().__init__(num_classes=num_classes)

        # Human-readable identifier for logging and registry lookups.
        # The trainer logs this as ``model.name`` at the start of each run.
        self.name: str = f"depthwisecnn_{variant}"

        # Store variant for repr / checkpoint metadata; not used by the
        # framework itself but useful for introspection and serialisation.
        self.variant: Variant = variant

        # Build the underlying architecture — no side effects beyond this.
        self._model: DepthwiseCNNModel = build_depthwisecnn(
            num_classes=num_classes,
            variant=variant,
            **model_kwargs,
        )

    # ------------------------------------------------------------------
    # FragmentClassifier contract — mandatory override
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: raw bytes → log-probabilities.

        The trainer calls ``model(x)`` inside ``torch.autocast`` during
        training and inside ``torch.no_grad()`` during validation.  Both
        paths work without any change here because the inner model is a
        standard ``nn.Module``.

        Parameters
        ----------
        x:
            Integer tensor of shape ``[B, L]`` with values in ``[0, 255]``.
            ``L`` is the fragment size in bytes (typically 512 or 4 096).
            Dtype must be ``torch.long`` (int64).

        Returns
        -------
        torch.Tensor
            Log-probabilities of shape ``[B, num_classes]``.  Values are
            ≤ 0 and each row sums to 0 in log-space (i.e. ``exp(row).sum()
            ≈ 1``).  Compatible with ``nn.NLLLoss`` and the base-class
            ``loss()`` method.
        """
        return self._model(x)

    # ------------------------------------------------------------------
    # FragmentClassifier contract — optional overrides
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters of the underlying model.

        Overrides the base-class implementation so that the count refers
        to ``self._model.parameters()`` rather than ``self.parameters()``
        — they are identical in practice (the adapter adds no parameters)
        but this makes the intent explicit and guards against future
        accidental additions to the wrapper.

        Parameters
        ----------
        trainable_only:
            If ``True`` (default), count only parameters with
            ``requires_grad=True``.

        Returns
        -------
        int
            Number of (trainable) scalar parameters.
        """
        params = self._model.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    # ------------------------------------------------------------------
    # nn.Module helpers
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        """Append variant and num_classes to the module repr string."""
        return f"variant={self.variant!r}, num_classes={self.num_classes}"


# ---------------------------------------------------------------------------
# Public factory — registered in src/models/registry.py
# ---------------------------------------------------------------------------


def build_depthwisecnn_adapter(
    num_classes: int,
    variant: Variant = "dsc",
    **kwargs: Any,
) -> DepthwiseCNNAdapter:
    """Construct a :class:`DepthwiseCNNAdapter`.

    This function is the **sole entry point** registered in the model
    registry under the key ``"depthwisecnn"``.  The registry calls it as::

        factory(num_classes=num_classes, **model_kwargs)

    where ``model_kwargs`` originates from the benchmark YAML (e.g.
    ``model.kwargs.variant: dsc-se``).

    Parameters
    ----------
    num_classes:
        Number of output classes.
    variant:
        Architecture variant — ``"dsc"``, ``"dsc-se"``, or ``"m-dsc"``.
    **kwargs:
        Forwarded to :class:`DepthwiseCNNAdapter`.  Recognised keys:
        ``dropout_p`` (float, default ``0.2``, M-DSC head only).

    Returns
    -------
    DepthwiseCNNAdapter
        Fully constructed, untrained adapter ready for the trainer.
    """
    return DepthwiseCNNAdapter(num_classes=num_classes, variant=variant, **kwargs)
