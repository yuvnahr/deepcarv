"""
src/models/adapters/bytenet_adapter.py
---------------------------------------
Framework registry adapter for ByteNet.

Delegates to benchmarks/ByteNet/src/adapter.py, keeping all
ByteNet model code isolated in the benchmarks/ directory.

Registry key: "bytenet"
Factory:      build_bytenet_adapter(num_classes, **kwargs)

Supported kwargs (forwarded to ByteNetBenchmarkAdapter):
    variant       : 'bytenet_resnet' | 'bytenet_former'  (default: 'bytenet_resnet')
    fragment_size : 512 | 4096                           (default: 512)
    embed_dim     : int   (embedding channel dim; auto from paper if not set)
    stage_layers  : list[int]  (blocks per stage)
    channels      : list[int]  (channel dims per stage)
    byte_branch_dim : int  (BBFE output dim, default 512)
    ngram_n       : int  (n-gram size, default 16)
    patch_size    : int  (ByteFormer only, default 8)
"""

from __future__ import annotations

from typing import Any

from benchmarks.ByteNet.src.adapter import (
    ByteNetBenchmarkAdapter,
    build_bytenet_benchmark_adapter,
)
from src.core.interfaces import FragmentClassifier


# Re-export so the registry import chain is transparent
ByteNetAdapter = ByteNetBenchmarkAdapter


def build_bytenet_adapter(num_classes: int, **kwargs: Any) -> FragmentClassifier:
    """Factory used by src.models.registry.MODEL_REGISTRY['bytenet']."""
    return build_bytenet_benchmark_adapter(num_classes=num_classes, **kwargs)
