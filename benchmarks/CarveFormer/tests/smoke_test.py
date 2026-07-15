"""
benchmarks/CarveFormer/tests/smoke_test.py
==========================================
Cheap, offline smoke tests for the CarveFormer benchmark.

These run on CPU with ``pretrained=False`` so they need no network access and
complete quickly. They verify the Phase 1/2 success criteria:

- the model builds and forwards for both fragment sizes,
- output shape matches ``num_classes`` and rows are valid log-probabilities,
- the framework adapter builds and forwards,
- the model registry can construct CarveFormer by name.

Run with either:
    pytest benchmarks/CarveFormer/tests/smoke_test.py
    python  benchmarks/CarveFormer/tests/smoke_test.py   (falls back to manual)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Make the benchmark ``src`` importable as a package for direct model/adapter
# access, and the repo root importable for the framework registry.
_BENCH_SRC = Path(__file__).resolve().parents[1] / "src"
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_REPO_ROOT), str(_BENCH_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _assert_log_probs(out: torch.Tensor, batch: int, num_classes: int) -> None:
    assert out.shape == (batch, num_classes), f"bad shape {tuple(out.shape)}"
    sums = out.exp().sum(dim=1)
    assert torch.allclose(sums, torch.ones(batch), atol=1e-4), f"not log-probs: {sums}"


def test_model_forward_512() -> None:
    from model import build_carveformer

    model = build_carveformer(num_classes=75, fragment_size=512, pretrained=False)
    model.eval()
    x = torch.randint(0, 256, (2, 512))
    with torch.no_grad():
        out = model(x)
    _assert_log_probs(out, 2, 75)


def test_model_forward_4096() -> None:
    from model import build_carveformer

    model = build_carveformer(num_classes=25, fragment_size=4096, pretrained=False)
    model.eval()
    x = torch.randint(0, 256, (2, 4096))
    with torch.no_grad():
        out = model(x)
    _assert_log_probs(out, 2, 25)


def test_adapter_forward() -> None:
    # Load the benchmark ``src`` as a package so the adapter's ``.model``
    # relative import resolves — this mirrors how the framework registry
    # loads it.
    import importlib.util

    pkg_name = "carveformer_benchmark_src"
    if pkg_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            pkg_name,
            _BENCH_SRC / "__init__.py",
            submodule_search_locations=[str(_BENCH_SRC)],
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[pkg_name] = module
        spec.loader.exec_module(module)
    adapter_mod = importlib.import_module(f"{pkg_name}.adapter")
    build_carveformer_adapter = adapter_mod.build_carveformer_adapter

    adapter = build_carveformer_adapter(num_classes=5, fragment_size=512, pretrained=False)
    adapter.eval()
    assert adapter.name == "carveformer"
    assert adapter.num_classes == 5
    assert adapter.num_parameters() > 0
    x = torch.randint(0, 256, (3, 512))
    with torch.no_grad():
        out = adapter(x)
    _assert_log_probs(out, 3, 5)


def test_registry_builds_carveformer() -> None:
    from src.models.registry import build_model, is_available

    assert is_available("carveformer")
    model = build_model("carveformer", num_classes=5, fragment_size=512, pretrained=False)
    model.eval()
    x = torch.randint(0, 256, (2, 512))
    with torch.no_grad():
        out = model(x)
    _assert_log_probs(out, 2, 5)


if __name__ == "__main__":
    test_model_forward_512()
    test_model_forward_4096()
    test_adapter_forward()
    test_registry_builds_carveformer()
    print("All CarveFormer smoke tests passed.")


# ---------------------------------------------------------------------------
# Phase 4: config loading + instantiation tests
# ---------------------------------------------------------------------------
def test_config_loads_and_has_paper_hyperparams() -> None:
    """The committed benchmark config loads and encodes the paper settings."""
    import yaml

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "benchmark.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    assert cfg["model"]["name"] == "carveformer"
    assert cfg["model"]["embed_dim"] == 96
    assert cfg["dataset"]["fragment_size"] in (512, 4096)
    # Paper hyperparameters
    assert abs(float(cfg["training"]["weight_decay"]) - 0.05) < 1e-9
    assert abs(float(cfg["training"]["lr"]) - 3.75e-4) < 1e-12
    assert cfg["training"]["optimizer"] == "adamw"
    assert cfg["training"]["epochs"] == 50


def test_default_grid_shapes() -> None:
    """The documented reshape grids are correct and consistent."""
    from model import default_grid_for_length

    h512, w512 = default_grid_for_length(512)
    h4096, w4096 = default_grid_for_length(4096)
    assert h512 * w512 == 512
    assert h4096 * w4096 == 4096
    assert (h4096, w4096) == (64, 64)  # matches SwinV2-Tiny native grid


def test_config_num_classes_variants() -> None:
    """The model builds for every FFT-75 scenario class count."""
    from model import build_carveformer

    for ncls in (75, 11, 25, 5, 2):
        m = build_carveformer(num_classes=ncls, fragment_size=512, pretrained=False)
        assert m.config.num_classes == ncls
