"""DepthwiseCNN benchmark package.

Exposes the benchmark-local adapter for import by the framework-level
adapter (src/models/adapters/depthwisecnn_adapter.py).
"""

from benchmarks.DepthwiseCNN.src.adapter import (
    DepthwiseCNNBenchmarkAdapter,
    build_depthwisecnn_benchmark_adapter,
)
from benchmarks.DepthwiseCNN.src.model import (
    DepthwiseCNN,
    DSCBlock,
    DSCSEBlock,
    MDSCBlock,
    build_depthwisecnn,
)

__all__ = [
    "DepthwiseCNN",
    "DSCBlock",
    "DSCSEBlock",
    "MDSCBlock",
    "build_depthwisecnn",
    "DepthwiseCNNBenchmarkAdapter",
    "build_depthwisecnn_benchmark_adapter",
]
