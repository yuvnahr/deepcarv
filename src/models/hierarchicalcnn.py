"""Hierarchical CNN template.

Architecture summary
--------------------
Multi-scale convolutional classifier that aggregates local byte motifs into
larger fragment-level representations.

Expected input
--------------
Tensor shaped ``[batch, fragment_size]``.

Expected output
---------------
Log-probabilities shaped ``[batch, num_classes]``.

Registry interface
------------------
Add an adapter under ``src/models/adapters/`` and register it in
``src.models.registry`` when the model is implemented.

TODO
----
- Define local, mid-level, and global aggregation stages.
- Decide whether 512 and 4096 fragments share weights.
- Add benchmark compatibility metadata.
- Add config and lightweight smoke tests.

No executable model code lives here yet; this file is a research template only.
"""
