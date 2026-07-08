"""DeepCarv template.

Architecture summary
--------------------
Reserved for the eventual flagship DeepCarv architecture. It should plug into
the same benchmark interfaces as ByteRCNN, CarveFormer, ByteNet, and CNN
families.

Expected input
--------------
Tensor shaped ``[batch, fragment_size]`` for FFT-75 and future supported
datasets.

Expected output
---------------
Log-probabilities shaped ``[batch, num_classes]``.

Registry interface
------------------
Implement through an adapter that returns ``FragmentClassifier`` and exposes
compatibility/profiling estimates.

TODO
----
- Define the architecture after baseline comparisons are locked.
- Document any DeepCarv-specific modules under ``src/research/extensions.py``.
- Add configs, tests, and reproducibility notes before enabling training.

No executable model code lives here yet; this file is a research template only.
"""
