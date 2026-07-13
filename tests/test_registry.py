import pytest

from src.models.adapters.bytercnn_adapter import ByteRCNNAdapter
from src.models.registry import ModelRegistryError, build_model, list_models


def test_registry_lists_all_expected_models():
    models = list_models()
    for name in ("bytercnn", "carveformer", "bytenet", "deepcarv"):
        assert name in models


def test_registry_builds_bytercnn():
    model = build_model("bytercnn", num_classes=5)
    assert isinstance(model, ByteRCNNAdapter)
    assert model.num_classes == 5


def test_registry_implemented_models_build():
    """CarveFormer / ByteNet / DepthwiseCNN are implemented and must build.

    (This test previously asserted they raised NotImplementedError, which was
    correct while they were stubs. They are now real models.)
    """
    for name in ("carveformer", "bytenet", "depthwisecnn"):
        model = build_model(name, num_classes=5)
        assert model.num_classes == 5


def test_registry_unimplemented_stub_still_raises():
    """Any model still registered only as a stub must fail loudly, not silently."""
    with pytest.raises(NotImplementedError):
        build_model("deepcarv", num_classes=5)


def test_registry_unknown_model_raises_registry_error():
    with pytest.raises(ModelRegistryError):
        build_model("does_not_exist", num_classes=5)
