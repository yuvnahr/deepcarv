# Adding Models

To add a model:

1. Implement a `FragmentClassifier` adapter under `src/models/adapters/`.
2. Register the adapter in `src/models/registry.py`.
3. Add `configs/models/<name>.yaml`.
4. Add compatibility methods where useful:
   - `supports_fragment_size`
   - `supports_num_classes`
   - `supports_dataset`
   - `supports_dtype`
   - `supports_gpu`
   - `estimated_memory`
   - `estimated_flops`
5. Add smoke tests that instantiate the model through the registry.

Template files exist for CarveFormer, ByteNet, Hierarchical CNN, Depthwise CNN,
and DeepCarv under `src/models/`. They are research templates only and do not
implement executable models.
