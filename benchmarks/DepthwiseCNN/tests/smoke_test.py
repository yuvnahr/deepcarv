"""
benchmarks/DepthwiseCNN/tests/smoke_test.py
-------------------------------------------
Smoke and integration tests for the DepthwiseCNN benchmark.

Covers two layers:
- **Phase 1 (model layer)**: all three variants instantiate, produce correct
  output shapes, valid log-probabilities, gradients, and deterministic eval.
- **Phase 2 (adapter + registry layer)**: adapter satisfies the framework's
  ``FragmentClassifier`` contract as exercised by the generic trainer and
  evaluator (``loss``, ``predict``, ``num_classes``, ``name``,
  ``num_parameters``, ``state_dict`` round-trip, train/eval mode switching).

Run with::

    PYTHONPATH=. pytest benchmarks/DepthwiseCNN/tests/smoke_test.py -v
"""

from __future__ import annotations

import copy

import pytest
import torch

from benchmarks.DepthwiseCNN.src.adapter import (
    DepthwiseCNNAdapter,
    build_depthwisecnn_adapter,
)
from benchmarks.DepthwiseCNN.src.model import (
    DepthwiseCNNModel,
    build_depthwisecnn,
)
from src.core.interfaces import FragmentClassifier
from src.models.registry import build_model, is_available, list_models

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

NUM_CLASSES = 75
BATCH_SIZE = 4
ALL_VARIANTS = ["dsc", "dsc-se", "m-dsc"]
FRAGMENT_SIZES = [512, 4096]


# ===========================================================================
# Phase 1 — model layer
# ===========================================================================


@pytest.mark.parametrize("variant", ALL_VARIANTS)
@pytest.mark.parametrize("fragment_size", FRAGMENT_SIZES)
def test_model_forward_shape(variant: str, fragment_size: int) -> None:
    """Raw model forward pass produces the correct output shape."""
    model = build_depthwisecnn(num_classes=NUM_CLASSES, variant=variant)
    x = torch.randint(0, 256, (BATCH_SIZE, fragment_size), dtype=torch.long)
    out = model(x)
    assert out.shape == (BATCH_SIZE, NUM_CLASSES)


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_model_factory_variant_attribute(variant: str) -> None:
    """build_depthwisecnn sets the variant attribute on the model."""
    model = build_depthwisecnn(num_classes=NUM_CLASSES, variant=variant)
    assert isinstance(model, DepthwiseCNNModel)
    assert model.variant == variant


def test_model_invalid_variant_raises() -> None:
    """An unknown variant string raises ValueError at construction time."""
    with pytest.raises(ValueError, match="Unknown variant"):
        build_depthwisecnn(num_classes=NUM_CLASSES, variant="bad")  # type: ignore[arg-type]


def test_model_invalid_num_classes_raises() -> None:
    """num_classes < 1 raises ValueError from the factory guard."""
    with pytest.raises(ValueError, match="num_classes must be"):
        build_depthwisecnn(num_classes=0, variant="dsc")


# ===========================================================================
# Phase 2 — adapter layer (framework contract surface)
# ===========================================================================


# ---------------------------------------------------------------------------
# 2a. Instantiation and identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_adapter_instantiates(variant: str) -> None:
    """Adapter constructs without error for every variant."""
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    assert model is not None


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_adapter_is_fragment_classifier(variant: str) -> None:
    """Adapter satisfies the ``FragmentClassifier`` runtime-checkable protocol."""
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    assert isinstance(model, FragmentClassifier)
    assert isinstance(model, DepthwiseCNNAdapter)


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_adapter_name_attribute(variant: str) -> None:
    """``name`` attribute matches the expected pattern for log/registry look-ups."""
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    assert model.name == f"depthwisecnn_{variant}"


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_adapter_variant_attribute(variant: str) -> None:
    """``variant`` attribute is set on the adapter for introspection."""
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    assert model.variant == variant


@pytest.mark.parametrize("num_classes", [10, 75, 100])
def test_adapter_num_classes_attribute(num_classes: int) -> None:
    """``num_classes`` attribute matches the constructor argument."""
    model = build_depthwisecnn_adapter(num_classes=num_classes, variant="dsc")
    assert model.num_classes == num_classes


# ---------------------------------------------------------------------------
# 2b. forward() — shape and log-probability correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ALL_VARIANTS)
@pytest.mark.parametrize("fragment_size", FRAGMENT_SIZES)
def test_adapter_forward_shape(variant: str, fragment_size: int) -> None:
    """Adapter forward produces shape ``[B, num_classes]``."""
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    x = torch.randint(0, 256, (BATCH_SIZE, fragment_size), dtype=torch.long)
    out = model(x)
    assert out.shape == (BATCH_SIZE, NUM_CLASSES)


@pytest.mark.parametrize("variant", ALL_VARIANTS)
@pytest.mark.parametrize("fragment_size", FRAGMENT_SIZES)
def test_adapter_output_is_log_probs(variant: str, fragment_size: int) -> None:
    """Adapter output sums to ≈ 1 after exp — valid log-probability distribution."""
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    model.eval()
    x = torch.randint(0, 256, (BATCH_SIZE, fragment_size), dtype=torch.long)
    with torch.no_grad():
        log_probs = model(x)
    probs_sum = log_probs.exp().sum(dim=1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-4), (
        f"[{variant}, L={fragment_size}] probs_sum={probs_sum}"
    )


# ---------------------------------------------------------------------------
# 2c. loss() — NLLLoss via inherited default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_adapter_loss_returns_scalar(variant: str) -> None:
    """``model.loss(log_probs, y)`` returns a 0-d scalar tensor."""
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    x = torch.randint(0, 256, (BATCH_SIZE, 512), dtype=torch.long)
    y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), dtype=torch.long)
    log_probs = model(x)
    loss = model.loss(log_probs, y)
    assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"
    assert loss.item() > 0.0, "Loss should be positive"


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_adapter_loss_is_differentiable(variant: str) -> None:
    """Loss from ``model.loss`` is differentiable (backward does not raise)."""
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    model.train()
    x = torch.randint(0, 256, (BATCH_SIZE, 512), dtype=torch.long)
    y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), dtype=torch.long)
    loss = model.loss(model(x), y)
    loss.backward()
    any_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )
    assert any_grad, f"[{variant}] No non-zero gradient after backward"


# ---------------------------------------------------------------------------
# 2d. predict() — inherited argmax
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_adapter_predict_shape_and_dtype(variant: str) -> None:
    """``model.predict(x)`` returns int64 class indices of shape ``[B]``."""
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    x = torch.randint(0, 256, (BATCH_SIZE, 512), dtype=torch.long)
    preds = model.predict(x)
    assert preds.shape == (BATCH_SIZE,), f"Expected ({BATCH_SIZE},), got {preds.shape}"
    assert preds.dtype == torch.int64
    assert preds.min() >= 0 and preds.max() < NUM_CLASSES


# ---------------------------------------------------------------------------
# 2e. num_parameters()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_adapter_num_parameters_reasonable(variant: str) -> None:
    """Parameter count is in the paper's expected range (~100 K ± 100 K)."""
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    n = model.num_parameters(trainable_only=True)
    assert 50_000 < n < 300_000, f"[{variant}] Unexpected param count: {n:,}"


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_adapter_num_parameters_total_ge_trainable(variant: str) -> None:
    """Total parameter count ≥ trainable parameter count."""
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    assert model.num_parameters(trainable_only=False) >= model.num_parameters(trainable_only=True)


# ---------------------------------------------------------------------------
# 2f. train/eval mode toggling — mirrors the trainer's call pattern
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_adapter_train_eval_toggle(variant: str) -> None:
    """model.train(True/False) switches training mode without raising."""
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    model.train(True)
    assert model.training is True
    model.train(False)
    assert model.training is False


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_eval_mode_deterministic(variant: str) -> None:
    """Two consecutive forward passes in eval mode return identical outputs."""
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    model.eval()
    x = torch.randint(0, 256, (BATCH_SIZE, 512), dtype=torch.long)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2), f"[{variant}] Eval outputs are not deterministic"


# ---------------------------------------------------------------------------
# 2g. state_dict round-trip — mirrors checkpoint save/load
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_adapter_state_dict_round_trip(variant: str) -> None:
    """state_dict can be saved and loaded back; loaded model produces identical output."""
    model_a = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    model_a.eval()
    x = torch.randint(0, 256, (BATCH_SIZE, 512), dtype=torch.long)

    with torch.no_grad():
        out_a = model_a(x)

    # Save and restore state dict into a fresh instance
    sd = copy.deepcopy(model_a.state_dict())
    model_b = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    model_b.load_state_dict(sd)
    model_b.eval()

    with torch.no_grad():
        out_b = model_b(x)

    assert torch.allclose(out_a, out_b), (
        f"[{variant}] Outputs differ after state_dict round-trip"
    )


# ---------------------------------------------------------------------------
# 2h. device transfer — mirrors trainer.model.to(device)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_adapter_to_cpu(variant: str) -> None:
    """model.to('cpu') does not raise and forward still works."""
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    model.to("cpu")
    x = torch.randint(0, 256, (BATCH_SIZE, 512), dtype=torch.long)
    out = model(x)
    assert out.shape == (BATCH_SIZE, NUM_CLASSES)


# ===========================================================================
# Phase 2 — registry integration
# ===========================================================================


def test_registry_contains_depthwisecnn() -> None:
    """``'depthwisecnn'`` key is present in the model registry."""
    assert is_available("depthwisecnn"), (
        "'depthwisecnn' is not registered.  Check src/models/registry.py."
    )
    assert "depthwisecnn" in list_models()


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_registry_build_model(variant: str) -> None:
    """``build_model('depthwisecnn', ...)`` instantiates without error."""
    model = build_model("depthwisecnn", num_classes=NUM_CLASSES, variant=variant)
    assert isinstance(model, DepthwiseCNNAdapter)
    assert isinstance(model, FragmentClassifier)


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_registry_forward_produces_correct_shape(variant: str) -> None:
    """Model built via registry forward pass returns ``[B, num_classes]``."""
    model = build_model("depthwisecnn", num_classes=NUM_CLASSES, variant=variant)
    x = torch.randint(0, 256, (BATCH_SIZE, 512), dtype=torch.long)
    out = model(x)
    assert out.shape == (BATCH_SIZE, NUM_CLASSES)


def test_registry_default_variant_is_dsc() -> None:
    """Default variant when none is specified is ``'dsc'``."""
    model = build_model("depthwisecnn", num_classes=NUM_CLASSES)
    assert model.variant == "dsc"  # type: ignore[attr-defined]


def test_registry_num_classes_alignment() -> None:
    """num_classes requested via registry matches the adapter's attribute."""
    model = build_model("depthwisecnn", num_classes=42, variant="dsc-se")
    assert model.num_classes == 42
    # Verify output dimension also matches
    x = torch.randint(0, 256, (2, 512), dtype=torch.long)
    assert model(x).shape == (2, 42)


def test_registry_does_not_break_existing_models() -> None:
    """Adding depthwisecnn to the registry does not remove other models."""
    registered = list_models()
    for expected in ("bytercnn",):
        assert expected in registered, f"'{expected}' missing from registry after depthwisecnn added"


# ===========================================================================
# Phase 4 — config loading tests
# ===========================================================================

from pathlib import Path
import yaml

# Locate benchmark.yaml relative to this test file (3 levels up = repo root).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _REPO_ROOT / "benchmarks" / "DepthwiseCNN" / "configs" / "benchmark.yaml"


def test_config_file_exists() -> None:
    """benchmark.yaml exists at the expected path."""
    assert _CONFIG_PATH.exists(), f"Config not found: {_CONFIG_PATH}"


def test_config_loads_as_valid_yaml() -> None:
    """benchmark.yaml is valid YAML and parses to a dict."""
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict), "Config must be a YAML mapping."


def test_config_has_required_top_level_sections() -> None:
    """benchmark.yaml has all required top-level sections."""
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    for section in ("dataset", "model", "training", "evaluation", "paths"):
        assert section in cfg, f"Missing top-level section: '{section}'"


def test_config_dataset_section() -> None:
    """dataset section has root_dir, fragment_size in {512, 4096}, bool cache."""
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    ds = cfg["dataset"]
    assert "root_dir" in ds
    assert ds["fragment_size"] in (512, 4096)
    assert isinstance(ds.get("cache"), bool)


def test_config_model_section() -> None:
    """model.name is depthwisecnn and model.kwargs.variant is a valid variant."""
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    mdl = cfg["model"]
    assert mdl["name"] == "depthwisecnn"
    assert mdl["kwargs"]["variant"] in ("dsc", "dsc-se", "m-dsc")


def test_config_training_section_keys() -> None:
    """training section contains every key consumed by TrainerConfig.from_dict()."""
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    train = cfg["training"]
    for key in ("epochs", "lr", "weight_decay", "optimizer",
                "scheduler", "grad_clip", "patience",
                "monitor", "monitor_mode", "amp"):
        assert key in train, f"training.{key} missing from config"


def test_config_paths_section() -> None:
    """paths section has run_outputs, best/last checkpoint, eval_outputs."""
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    for key in ("run_outputs", "best_checkpoint", "last_checkpoint", "eval_outputs"):
        assert key in paths, f"paths.{key} missing from config"


def test_config_model_builds_from_config_kwargs() -> None:
    """Model instantiates using name + kwargs directly from the config."""
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    model = build_model(
        cfg["model"]["name"],
        num_classes=75,
        **cfg["model"].get("kwargs", {}),
    )
    assert isinstance(model, FragmentClassifier)
    assert model.num_classes == 75


# ===========================================================================
# Phase 4 — model internals and block structure
# ===========================================================================


def test_model_has_three_inception_blocks() -> None:
    """Model contains exactly 3 InceptionBlock modules."""
    from benchmarks.DepthwiseCNN.src.model import InceptionBlock
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant="dsc")
    blocks = [m for m in model._model.modules() if isinstance(m, InceptionBlock)]
    assert len(blocks) == 3, f"Expected 3 InceptionBlocks, found {len(blocks)}"


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_model_embedding_vocab_and_dim(variant: str) -> None:
    """Embedding has vocab=256 and dim=32 (paper spec)."""
    import torch.nn as nn
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    emb = model._model.embedding
    assert isinstance(emb, nn.Embedding)
    assert emb.num_embeddings == 256
    assert emb.embedding_dim == 32


def test_dsc_se_has_three_se_blocks() -> None:
    """DSC-SE contains exactly 3 SEBlock modules."""
    from benchmarks.DepthwiseCNN.src.model import SEBlock
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant="dsc-se")
    se_blocks = [m for m in model._model.modules() if isinstance(m, SEBlock)]
    assert len(se_blocks) == 3, f"Expected 3 SEBlocks, found {len(se_blocks)}"


@pytest.mark.parametrize("variant", ["dsc", "m-dsc"])
def test_non_se_variants_have_no_se_blocks(variant: str) -> None:
    """DSC and M-DSC contain zero SEBlock modules."""
    from benchmarks.DepthwiseCNN.src.model import SEBlock
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    se_blocks = [m for m in model._model.modules() if isinstance(m, SEBlock)]
    assert len(se_blocks) == 0, f"[{variant}] Expected 0 SEBlocks, found {len(se_blocks)}"


def test_m_dsc_first_conv_is_depthwise() -> None:
    """M-DSC first Conv1d has groups == in_channels (depthwise)."""
    import torch.nn as nn
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant="m-dsc")
    conv1 = model._model.conv1
    assert isinstance(conv1, nn.Conv1d)
    assert conv1.groups == conv1.in_channels


def test_dsc_first_conv_is_not_depthwise() -> None:
    """DSC first Conv1d is a full (groups=1) convolution."""
    import torch.nn as nn
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant="dsc")
    conv1 = model._model.conv1
    assert isinstance(conv1, nn.Conv1d)
    assert conv1.groups == 1, f"DSC first conv should be groups=1, got {conv1.groups}"


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_first_conv_kernel_size_is_19(variant: str) -> None:
    """First Conv1d kernel size is 19 (paper spec)."""
    import torch.nn as nn
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    conv1 = model._model.conv1
    assert isinstance(conv1, nn.Conv1d)
    assert conv1.kernel_size == (19,), f"[{variant}] got {conv1.kernel_size}"


@pytest.mark.parametrize("nc", [10, 75, 100])
def test_classifier_out_channels_matches_num_classes(nc: int) -> None:
    """Final classifier Conv1d has out_channels == num_classes."""
    import torch.nn as nn
    model = build_depthwisecnn_adapter(num_classes=nc, variant="dsc")
    clf = model._model.classifier
    assert isinstance(clf, nn.Conv1d)
    assert clf.out_channels == nc


# ===========================================================================
# Phase 4 — framework output contract completeness
# ===========================================================================


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_output_values_are_non_positive(variant: str) -> None:
    """All log-probability outputs are <= 0."""
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    model.eval()
    x = torch.randint(0, 256, (BATCH_SIZE, 512), dtype=torch.long)
    with torch.no_grad():
        out = model(x)
    assert (out <= 0).all(), f"[{variant}] max log-prob={out.max().item():.4f}"


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_predict_consistent_with_argmax(variant: str) -> None:
    """model(x).argmax(1) equals model.predict(x)."""
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    model.eval()
    x = torch.randint(0, 256, (BATCH_SIZE, 512), dtype=torch.long)
    with torch.no_grad():
        log_probs = model(x)
    assert torch.equal(log_probs.argmax(dim=1), model.predict(x))


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_nll_loss_on_output_is_finite(variant: str) -> None:
    """NLLLoss applied to model output is finite and positive."""
    import torch.nn.functional as F
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    model.eval()
    x = torch.randint(0, 256, (BATCH_SIZE, 512), dtype=torch.long)
    y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), dtype=torch.long)
    with torch.no_grad():
        loss = F.nll_loss(model(x), y)
    assert torch.isfinite(loss) and loss.item() > 0


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_all_parameters_require_grad(variant: str) -> None:
    """All parameters have requires_grad=True after construction."""
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert frozen == [], f"[{variant}] Unexpectedly frozen: {frozen}"


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_no_nan_for_zero_input(variant: str) -> None:
    """Forward produces no NaN for an all-zero byte input."""
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    model.eval()
    x = torch.zeros(BATCH_SIZE, 512, dtype=torch.long)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any(), f"[{variant}] NaN on zero input"


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_no_nan_for_max_input(variant: str) -> None:
    """Forward produces no NaN for an all-255 byte input."""
    model = build_depthwisecnn_adapter(num_classes=NUM_CLASSES, variant=variant)
    model.eval()
    x = torch.full((BATCH_SIZE, 512), 255, dtype=torch.long)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any(), f"[{variant}] NaN on max-byte input"
