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


def test_registry_stub_models_raise_not_implemented():
    with pytest.raises(NotImplementedError):
        build_model("carveformer", num_classes=5)


def test_registry_unknown_model_raises_registry_error():
    with pytest.raises(ModelRegistryError):
        build_model("does_not_exist", num_classes=5)
