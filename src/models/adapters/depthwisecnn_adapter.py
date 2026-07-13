"""
src/models/adapters/depthwisecnn_adapter.py
--------------------------------------------
Framework registry adapter for DepthwiseCNN.

Delegates to benchmarks/DepthwiseCNN/src/adapter.py, keeping all
DepthwiseCNN model code isolated in the benchmarks/ directory.

Registry key: "depthwisecnn"
Factory:      build_depthwisecnn_adapter(num_classes, **kwargs)

Supported kwargs (forwarded to DepthwiseCNNBenchmarkAdapter):
    variant       : 'dsc' | 'dsc_se' | 'm_dsc'  (default: 'dsc')
    fragment_size : 512 | 4096                   (default: 512)
    embed_dim     : int   (byte embedding dim; default: 64)
    channels      : list[int]  (channel dims per block; default: [64,128,256,256])
    kernel_size   : int  (depthwise kernel for DSC/DSC-SE; default: 3)
    se_reduction  : int  (SE block reduction ratio; default: 16)
    p_dropout     : float  (head dropout probability; default: 0.5)
"""

from __future__ import annotations

from typing import Any

from benchmarks.DepthwiseCNN.src.adapter import (
    DepthwiseCNNAdapter as BenchmarkDepthwiseCNNAdapter,
    build_depthwisecnn_adapter as build_depthwisecnn_benchmark_adapter,
)
from src.core.interfaces import FragmentClassifier

# Re-export so the registry import chain is transparent
DepthwiseCNNAdapter = BenchmarkDepthwiseCNNAdapter


def build_depthwisecnn_adapter(num_classes: int, **kwargs: Any) -> FragmentClassifier:
    """Factory used by src.models.registry.MODEL_REGISTRY['depthwisecnn']."""
    return build_depthwisecnn_benchmark_adapter(num_classes=num_classes, **kwargs)
