"""
src/models/base.py
--------------------
Shared helpers for model adapters. Adapters should inherit from
`src.core.interfaces.FragmentClassifier` directly; this module holds
small utilities so adapters don't duplicate boilerplate.
"""

from __future__ import annotations

from src.core.interfaces import FragmentClassifier


class NotAvailableModel(FragmentClassifier):
    """Placeholder returned by the registry for models that are registered
    but not yet implemented (CarveFormer, ByteNet, DeepCarv stubs).

    Instantiating this raises immediately with a clear message rather than
    silently pretending to be a working model — the registry API stays
    stable (the name resolves) but the model is not runnable yet.
    """

    name = "not_available"

    def __init__(self, num_classes: int = 0, **kwargs) -> None:  # noqa: D401
        raise NotImplementedError(
            f"Model '{self.name}' is registered as a template/stub only and "
            "has no implementation yet. Implement its adapter in "
            "src/models/adapters/ and update the registry entry to point "
            "at the real class."
        )

    def forward(self, x):  # pragma: no cover - unreachable, __init__ raises
        raise NotImplementedError
