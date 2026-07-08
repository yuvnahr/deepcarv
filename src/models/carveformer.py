"""CarveFormer template.

Architecture summary
--------------------
Transformer-style byte-fragment classifier for FFT-75 fragments. Intended
components include byte/token embedding, positional encoding, lightweight
self-attention blocks, pooling, and a classifier head.

Expected input
--------------
Tensor shaped ``[batch, fragment_size]`` containing byte values or normalized
byte features compatible with the benchmark dataset loader.

Expected output
---------------
Log-probabilities shaped ``[batch, num_classes]`` for use with the generic
``FragmentClassifier`` training interface.

Registry interface
------------------
Implement ``src.models.adapters.carveformer_adapter.build_carveformer_adapter``
so it returns a concrete ``FragmentClassifier``.

TODO
----
- Define the production architecture and parameter budget.
- Add compatibility methods from ``src.compatibility.BenchmarkCompatible``.
- Add model config fields under ``configs/models/carveformer.yaml``.
- Add smoke tests that instantiate the adapter.

No executable model code lives here yet; this file is a research template only.
"""
