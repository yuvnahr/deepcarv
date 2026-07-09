"""
benchmarks/DepthwiseCNN/tests/smoke_test.py
-------------------------------------------
Smoke tests for the DepthwiseCNN benchmark variants to ensure they load
and can perform a forward pass without shape mismatch errors.
"""

import pytest
import torch

from benchmarks.DepthwiseCNN.src.adapter import build_depthwisecnn_adapter
from src.core.interfaces import FragmentClassifier

@pytest.mark.parametrize("variant", ["dsc", "dsc-se", "m-dsc"])
@pytest.mark.parametrize("fragment_size", [512, 4096])
def test_forward_pass(variant, fragment_size):
    num_classes = 75
    batch_size = 4
    
    # Instantiate adapter
    model = build_depthwisecnn_adapter(num_classes=num_classes, variant=variant)
    
    # Assert model conforms to FragmentClassifier
    assert isinstance(model, FragmentClassifier)
    
    # Create random byte input (integers 0-255)
    x = torch.randint(0, 256, (batch_size, fragment_size), dtype=torch.long)
    
    # Forward pass
    log_probs = model(x)
    
    # Check output shape
    assert log_probs.shape == (batch_size, num_classes)
    
    # Check log_probs are actually log probabilities (sum of exp across classes approx 1)
    probs_sum = torch.exp(log_probs).sum(dim=1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-4)

def test_parameter_count():
    model_dsc = build_depthwisecnn_adapter(num_classes=75, variant="dsc")
    params = model_dsc.num_parameters(trainable_only=True)
    # The paper says around 100K parameters
    assert 50_000 < params < 150_000, f"Expected around 100K parameters, got {params}"
