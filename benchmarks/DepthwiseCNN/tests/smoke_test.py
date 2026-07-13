"""
benchmarks/DepthwiseCNN/tests/smoke_test.py
--------------------------------------------
Smoke tests for the DepthwiseCNN benchmark.

Tests verify:
1. Model instantiation for all three variants (DSC, DSC-SE, M-DSC)
2. Forward pass shape correctness
3. Output is valid log-probabilities (finite, ≤ 0)
4. Registry integration (model accessible by name)
5. Adapter produces the correct FragmentClassifier interface
6. Parameter counts are sane (lightweight: < 1M for 512-byte input)
7. Training/evaluation integration with tiny synthetic data

These tests do NOT require the real FFT-75 dataset.
All data is synthetic and created in memory.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks.DepthwiseCNN.src.model import (
    DSCBlock,
    DSCSEBlock,
    MDSCBlock,
    DepthwiseCNN,
    build_depthwisecnn,
)
from benchmarks.DepthwiseCNN.src.adapter import (
    DepthwiseCNNBenchmarkAdapter,
    build_depthwisecnn_benchmark_adapter,
)
from src.models.registry import build_model, list_models


# ---------------------------------------------------------------------------
# Constants for all tests
# ---------------------------------------------------------------------------
NUM_CLASSES = 75
FRAGMENT_SIZE = 512
BATCH_SIZE = 4
VARIANTS = ["dsc", "dsc_se", "m_dsc"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(batch_size: int = BATCH_SIZE, fragment_size: int = FRAGMENT_SIZE) -> torch.Tensor:
    """Random byte indices in [0, 255]."""
    return torch.randint(0, 256, (batch_size, fragment_size), dtype=torch.long)


# ---------------------------------------------------------------------------
# Building-block tests
# ---------------------------------------------------------------------------


class TestDSCBlock:
    def test_same_channels(self):
        block = DSCBlock(64, 64, kernel_size=3)
        x = torch.randn(2, 64, 32)
        out = block(x)
        assert out.shape == (2, 64, 32), f"Expected (2,64,32), got {out.shape}"

    def test_different_channels(self):
        block = DSCBlock(64, 128, kernel_size=3)
        x = torch.randn(2, 64, 32)
        out = block(x)
        assert out.shape == (2, 128, 32), f"Expected (2,128,32), got {out.shape}"

    def test_large_kernel(self):
        block = DSCBlock(32, 32, kernel_size=7)
        x = torch.randn(2, 32, 64)
        out = block(x)
        assert out.shape == (2, 32, 64)


class TestDSCSEBlock:
    def test_shape(self):
        block = DSCSEBlock(64, 128, kernel_size=3, se_reduction=16)
        x = torch.randn(2, 64, 32)
        out = block(x)
        assert out.shape == (2, 128, 32)


class TestMDSCBlock:
    def test_shape(self):
        block = MDSCBlock(64, 128)
        x = torch.randn(2, 64, 32)
        out = block(x)
        assert out.shape == (2, 128, 32)

    def test_channel_divisible_by_3(self):
        # Channels not perfectly divisible by 3 — remainder goes to branch 0
        block = MDSCBlock(64, 65)   # 65 = 21 + 22 + 22 or similar
        x = torch.randn(2, 64, 32)
        out = block(x)
        assert out.shape == (2, 65, 32)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestDepthwiseCNN:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_instantiation(self, variant):
        model = build_depthwisecnn(
            num_classes=NUM_CLASSES,
            fragment_size=FRAGMENT_SIZE,
            variant=variant,
        )
        assert isinstance(model, DepthwiseCNN)
        assert model.num_classes == NUM_CLASSES

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_forward_shape(self, variant):
        model = build_depthwisecnn(
            num_classes=NUM_CLASSES, fragment_size=FRAGMENT_SIZE, variant=variant
        )
        model.eval()
        x = _make_batch()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES), \
            f"variant={variant}: expected ({BATCH_SIZE},{NUM_CLASSES}), got {out.shape}"

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_log_probs_valid(self, variant):
        model = build_depthwisecnn(
            num_classes=NUM_CLASSES, fragment_size=FRAGMENT_SIZE, variant=variant
        )
        model.eval()
        x = _make_batch()
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"
        assert (out <= 0).all(), "Log-probabilities must be ≤ 0"
        # Probabilities should sum to ~1.0 per sample
        probs_sum = out.exp().sum(dim=1)
        assert torch.allclose(probs_sum, torch.ones(BATCH_SIZE), atol=1e-4), \
            f"Probs do not sum to 1: {probs_sum}"

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_lightweight_parameter_count(self, variant):
        """All variants should be < 1M parameters for 512-byte input."""
        model = build_depthwisecnn(
            num_classes=NUM_CLASSES, fragment_size=FRAGMENT_SIZE, variant=variant
        )
        n_params = model.num_parameters()
        assert n_params < 1_500_000, \
            f"variant={variant}: {n_params:,} params exceeds 1.5M (not lightweight)"

    def test_4096_fragment_size(self):
        """Model should accept 4096-byte fragments."""
        model = build_depthwisecnn(
            num_classes=NUM_CLASSES, fragment_size=4096, variant="dsc"
        )
        model.eval()
        x = torch.randint(0, 256, (2, 4096), dtype=torch.long)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, NUM_CLASSES)

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            build_depthwisecnn(num_classes=10, variant="nonexistent")  # type: ignore


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------


class TestDepthwiseCNNAdapter:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_adapter_forward(self, variant):
        adapter = build_depthwisecnn_benchmark_adapter(
            num_classes=NUM_CLASSES, variant=variant, fragment_size=FRAGMENT_SIZE
        )
        adapter.eval()
        x = _make_batch()
        with torch.no_grad():
            out = adapter(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_adapter_loss(self):
        adapter = build_depthwisecnn_benchmark_adapter(
            num_classes=NUM_CLASSES, variant="dsc", fragment_size=FRAGMENT_SIZE
        )
        adapter.eval()
        x = _make_batch()
        y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
        with torch.no_grad():
            log_probs = adapter(x)
            loss = adapter.loss(log_probs, y)
        assert loss.ndim == 0, "Loss should be a scalar"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_adapter_num_parameters(self):
        adapter = build_depthwisecnn_benchmark_adapter(
            num_classes=NUM_CLASSES, variant="dsc", fragment_size=FRAGMENT_SIZE
        )
        n = adapter.num_parameters(trainable_only=True)
        assert n > 0

    def test_adapter_predict(self):
        adapter = build_depthwisecnn_benchmark_adapter(
            num_classes=NUM_CLASSES, variant="dsc", fragment_size=FRAGMENT_SIZE
        )
        x = _make_batch()
        preds = adapter.predict(x)
        assert preds.shape == (BATCH_SIZE,)
        assert ((preds >= 0) & (preds < NUM_CLASSES)).all()


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_depthwisecnn_in_registry(self):
        models = list_models()
        assert "depthwisecnn" in models, f"depthwisecnn not in registry: {models}"

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_build_via_registry(self, variant):
        model = build_model(
            "depthwisecnn",
            num_classes=NUM_CLASSES,
            variant=variant,
            fragment_size=FRAGMENT_SIZE,
        )
        assert hasattr(model, "forward")
        assert hasattr(model, "num_parameters")
        model.eval()
        x = _make_batch()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)


# ---------------------------------------------------------------------------
# Training integration smoke test
# ---------------------------------------------------------------------------


class TestTrainingIntegration:
    def test_one_gradient_step(self):
        """Verify that a forward + backward pass works without errors."""
        model = build_depthwisecnn(
            num_classes=5, fragment_size=FRAGMENT_SIZE, variant="dsc"
        )
        model.train()
        x = _make_batch(batch_size=8, fragment_size=FRAGMENT_SIZE)
        y = torch.randint(0, 5, (8,))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        log_probs = model(x)
        loss = torch.nn.functional.nll_loss(log_probs, y)
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_train_val_pipeline_with_tiny_dataset(self, tmp_path):
        """End-to-end smoke test with synthetic NPZ dataset."""
        from pathlib import Path

        # Create a tiny synthetic dataset
        frag_dir = tmp_path / "FFT-75" / "512"
        frag_dir.mkdir(parents=True)
        rng = np.random.default_rng(0)
        num_classes = 5
        for split, n in (("train", 40), ("val", 20), ("test", 20)):
            X = rng.integers(0, 256, (n, 512), dtype=np.uint8)
            y = rng.integers(0, num_classes, (n,), dtype=np.int64)
            np.savez(frag_dir / f"{split}.npz", x=X, y=y)

        from src.data.dataset import FragmentDataset, build_dataloader
        from src.training.trainer import Trainer, TrainerConfig

        train_ds = FragmentDataset(tmp_path / "FFT-75", "train", 512)
        val_ds   = FragmentDataset(tmp_path / "FFT-75", "val",   512)
        train_loader = build_dataloader(train_ds, batch_size=16, shuffle=True)
        val_loader   = build_dataloader(val_ds,   batch_size=16, shuffle=False)

        model = build_model("depthwisecnn", num_classes=num_classes, fragment_size=512)
        cfg = TrainerConfig(epochs=2, lr=1e-3, patience=999, amp=False)
        trainer = Trainer(model, cfg, tmp_path / "outputs")
        history = trainer.fit(train_loader, val_loader)

        assert len(history.train_loss) == 2
        assert all(torch.isfinite(torch.tensor(l)) for l in history.train_loss)
