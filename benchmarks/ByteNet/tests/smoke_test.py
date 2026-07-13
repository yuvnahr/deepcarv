"""
benchmarks/ByteNet/tests/smoke_test.py
----------------------------------------
CPU-only smoke tests for the ByteNet benchmark.

Tests verify:
1. Registry resolves 'bytenet' without error.
2. Byte2Image produces the correct output shape.
3. ByteResNet forward pass (tiny batch, no dataset needed).
4. ByteFormer forward pass (tiny batch, no dataset needed).
5. Adapter satisfies the FragmentClassifier contract (forward, loss, num_parameters).
6. 4096-byte mode (8-chunk) forward pass.
7. NGramEmbedding and PatchEmbedding output shapes.

These tests run on CPU with tiny inputs and require no FFT-75 dataset.
They are designed to catch shape bugs, import errors, and registry
misconfigurations quickly.

Run with:
    pytest benchmarks/ByteNet/tests/smoke_test.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Ensure repo root is on sys.path when running from benchmarks/ dir
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH = 4
FS_512 = 512
FS_4096 = 4096
NUM_CLASSES = 75
NGRAM_N = 16

# Expected image shape for a 512-byte sector with n=16
_H = FS_512 - NGRAM_N + 1   # 497
_W = 8 * NGRAM_N             # 128


def _make_byte_tensor(fragment_size: int, batch: int = BATCH) -> torch.Tensor:
    """Create a random byte tensor matching FragmentDataset output dtype."""
    return torch.randint(0, 256, (batch, fragment_size), dtype=torch.int64)


# ---------------------------------------------------------------------------
# Test: Byte2Image shape
# ---------------------------------------------------------------------------

def test_byte2image_512_shape():
    from benchmarks.ByteNet.src.model import Byte2Image
    b2i = Byte2Image(ngram_n=NGRAM_N, fragment_size=FS_512)
    x = _make_byte_tensor(FS_512)
    img = b2i(x)
    assert img.shape == (BATCH, 1, _H, _W), f"Expected [4,1,497,128], got {img.shape}"


def test_byte2image_values_normalised():
    from benchmarks.ByteNet.src.model import Byte2Image
    b2i = Byte2Image(ngram_n=NGRAM_N, fragment_size=FS_512)
    x = torch.zeros(2, FS_512, dtype=torch.int64)
    img = b2i(x)
    # All bytes are 0, so after normalisation the value should be constant
    assert img.dtype == torch.float32
    assert img.isfinite().all(), "Image contains NaN/Inf"


# ---------------------------------------------------------------------------
# Test: NGramEmbedding and PatchEmbedding shapes
# ---------------------------------------------------------------------------

def test_ngram_embedding_shape():
    from benchmarks.ByteNet.src.model import NGramEmbedding
    emb = NGramEmbedding(embed_dim=96, in_h=_H, in_w=_W, out_channels=64)
    img = torch.randn(BATCH, 1, _H, _W)
    out = emb(img)
    assert out.shape[0] == BATCH
    assert out.shape[1] == 64, f"Expected 64 channels, got {out.shape[1]}"


def test_patch_embedding_shape():
    from benchmarks.ByteNet.src.model import PatchEmbedding
    emb = PatchEmbedding(embed_dim=64, patch_size=8, in_channels=1,
                         in_h=_H, in_w=_W)
    img = torch.randn(BATCH, 1, _H, _W)
    out = emb(img)
    assert out.shape[0] == BATCH
    assert out.shape[1] == 64, f"Expected embed_dim=64, got {out.shape[1]}"


# ---------------------------------------------------------------------------
# Test: ByteResNet forward pass (512B)
# ---------------------------------------------------------------------------

def test_bytenet_resnet_512_forward():
    from benchmarks.ByteNet.src.model import build_bytenet_resnet
    model = build_bytenet_resnet(
        num_classes=NUM_CLASSES,
        fragment_size=FS_512,
        embed_dim=96,
        stage_layers=(2, 2, 2, 2),
        channels=(64, 128, 256, 512),
    )
    model.eval()
    x = _make_byte_tensor(FS_512)
    with torch.no_grad():
        log_probs = model(x)
    assert log_probs.shape == (BATCH, NUM_CLASSES), \
        f"Expected [{BATCH}, {NUM_CLASSES}], got {log_probs.shape}"
    # Log-probabilities should sum to ~1 in probability space
    probs = log_probs.exp()
    assert torch.allclose(probs.sum(dim=1), torch.ones(BATCH), atol=1e-4), \
        "Probabilities do not sum to 1"


# ---------------------------------------------------------------------------
# Test: ByteFormer forward pass (512B)
# ---------------------------------------------------------------------------

def test_bytenet_former_512_forward():
    from benchmarks.ByteNet.src.model import build_bytenet_former
    model = build_bytenet_former(
        num_classes=NUM_CLASSES,
        fragment_size=FS_512,
        embed_dim=64,
        patch_size=8,
        stage_layers=(2, 2, 4, 2),   # reduced for speed in test
        channels=(32, 64, 128, 256),  # reduced for speed in test
    )
    model.eval()
    x = _make_byte_tensor(FS_512)
    with torch.no_grad():
        log_probs = model(x)
    assert log_probs.shape == (BATCH, NUM_CLASSES)
    probs = log_probs.exp()
    assert torch.allclose(probs.sum(dim=1), torch.ones(BATCH), atol=1e-4)


# ---------------------------------------------------------------------------
# Test: 4096B mode (8×512 chunking)
# ---------------------------------------------------------------------------

def test_bytenet_resnet_4096_forward():
    from benchmarks.ByteNet.src.model import build_bytenet_resnet
    model = build_bytenet_resnet(
        num_classes=NUM_CLASSES,
        fragment_size=FS_4096,
        embed_dim=96,
        stage_layers=(1, 1, 1, 1),   # reduced for speed
        channels=(32, 64, 128, 256),  # reduced for speed
    )
    model.eval()
    x = _make_byte_tensor(FS_4096, batch=2)
    with torch.no_grad():
        log_probs = model(x)
    assert log_probs.shape == (2, NUM_CLASSES)


# ---------------------------------------------------------------------------
# Test: Registry resolution
# ---------------------------------------------------------------------------

def test_registry_resolves_bytenet():
    from src.models.registry import build_model, is_available
    assert is_available("bytenet"), "'bytenet' not in registry"
    model = build_model(
        "bytenet",
        num_classes=NUM_CLASSES,
        variant="bytenet_resnet",
        fragment_size=FS_512,
        stage_layers=(1, 1, 1, 1),
        channels=(32, 64, 128, 256),
    )
    assert model is not None


# ---------------------------------------------------------------------------
# Test: Adapter contract (FragmentClassifier interface)
# ---------------------------------------------------------------------------

def test_adapter_contract():
    from src.models.registry import build_model
    from src.core.interfaces import FragmentClassifier

    model = build_model(
        "bytenet",
        num_classes=NUM_CLASSES,
        variant="bytenet_resnet",
        fragment_size=FS_512,
        stage_layers=(1, 1, 1, 1),
        channels=(32, 64, 128, 256),
    )
    # Must be a FragmentClassifier
    assert isinstance(model, FragmentClassifier), \
        "ByteNetAdapter does not inherit FragmentClassifier"

    # forward must return log-probs of correct shape
    x = _make_byte_tensor(FS_512)
    log_probs = model(x)
    assert log_probs.shape == (BATCH, NUM_CLASSES)

    # loss must return a scalar
    targets = torch.randint(0, NUM_CLASSES, (BATCH,))
    loss = model.loss(log_probs, targets)
    assert loss.ndim == 0, f"loss should be scalar, got shape {loss.shape}"

    # num_parameters must be a positive int
    nparams = model.num_parameters()
    assert isinstance(nparams, int) and nparams > 0


# ---------------------------------------------------------------------------
# Test: ByteBranch output shape
# ---------------------------------------------------------------------------

def test_byte_branch_shape():
    from benchmarks.ByteNet.src.model import ByteBranch
    bb = ByteBranch(fragment_size=FS_512, out_dim=512)
    x = _make_byte_tensor(FS_512)
    out = bb(x)
    assert out.shape == (BATCH, 512)


# ---------------------------------------------------------------------------
# Test: ResNetBlock shape preservation
# ---------------------------------------------------------------------------

def test_resnet_block_shape():
    from benchmarks.ByteNet.src.model import ResNetBlock
    block = ResNetBlock(64, 64)
    x = torch.randn(BATCH, 64, 16, 16)
    out = block(x)
    assert out.shape == (BATCH, 64, 16, 16)


def test_resnet_block_channel_change():
    from benchmarks.ByteNet.src.model import ResNetBlock
    block = ResNetBlock(64, 128, stride=2)
    x = torch.randn(BATCH, 64, 16, 16)
    out = block(x)
    assert out.shape == (BATCH, 128, 8, 8)


# ---------------------------------------------------------------------------
# Test: PoolFormerBlock shape preservation
# ---------------------------------------------------------------------------

def test_poolformer_block_shape():
    from benchmarks.ByteNet.src.model import PoolFormerBlock
    block = PoolFormerBlock(64)
    x = torch.randn(BATCH, 64, 32, 16)
    out = block(x)
    assert out.shape == (BATCH, 64, 32, 16)
