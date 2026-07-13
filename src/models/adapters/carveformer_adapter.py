"""
src/models/adapters/carveformer_adapter.py
------------------------------------------
Registry hook for CarveFormer.

The real CarveFormer implementation lives in the benchmark branch under
``benchmarks/CarveFormer/src/`` (model.py + adapter.py), keeping the paper
reproduction self-contained in its own directory. This module is the stable
registry entry point: it attempts to load that real adapter and delegates to
it. If the benchmark code cannot be imported (e.g. optional ``timm`` dependency
missing, or the benchmark directory absent), it falls back to the
``NotAvailableModel`` stub so the registry key still resolves with a clear
error instead of an import crash.

The registry key ("carveformer") and the factory signature
``build_carveformer_adapter(num_classes, **kwargs)`` are unchanged from the
original stub, so existing experiment configs keep working.
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

from src.models.base import NotAvailableModel
from src.utils.paths import BENCHMARKS_DIR

_CARVEFORMER_SRC = BENCHMARKS_DIR / "CarveFormer" / "src"


def _load_real_adapter_factory() -> Any:
    """Import the CarveFormer benchmark adapter factory, or return None.

    Loads ``benchmarks/CarveFormer/src/adapter.py`` as a package so its
    relative import of ``.model`` resolves. Returns the
    ``build_carveformer_adapter`` callable, or ``None`` if unavailable.
    """
    pkg_init = _CARVEFORMER_SRC / "__init__.py"
    adapter_file = _CARVEFORMER_SRC / "adapter.py"
    if not adapter_file.exists() or not pkg_init.exists():
        return None

    pkg_name = "carveformer_benchmark_src"
    try:
        if pkg_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                pkg_name,
                pkg_init,
                submodule_search_locations=[str(_CARVEFORMER_SRC)],
            )
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            sys.modules[pkg_name] = module
            spec.loader.exec_module(module)

        adapter_mod = importlib.import_module(f"{pkg_name}.adapter")
        return adapter_mod.build_carveformer_adapter
    except Exception:  # noqa: BLE001 - any import failure -> graceful fallback
        return None


class CarveFormerAdapter(NotAvailableModel):
    """Fallback stub used only if the benchmark implementation can't be loaded."""

    name = "carveformer"


def build_carveformer_adapter(num_classes: int, **kwargs: Any) -> Any:
    """Factory used by the model registry.

    Delegates to the real CarveFormer benchmark adapter when available;
    otherwise raises via the ``NotAvailableModel`` stub with a clear message.
    """
    factory = _load_real_adapter_factory()
    if factory is not None:
        return factory(num_classes=num_classes, **kwargs)
    return CarveFormerAdapter(num_classes=num_classes, **kwargs)
