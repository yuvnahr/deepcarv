# DepthwiseCNN Implementation Checklist

This checklist tracks the integration of the paper "File Fragment Type Classification Using Light-Weight Convolutional Neural Networks" into the DeepCarv framework.

## Phase 0: Scaffold & Plan (Completed)
- [x] Analyze codebase structure.
- [x] Design integration plan that respects existing contracts (e.g. `FragmentClassifier`).
- [x] Preserve FFT-75 NPZ loader without resorting to CSV generation.

## Phase 1: Model Architecture (Completed)
- [x] Implement standard Depthwise Separable Convolutions.
- [x] Implement 3-branch Inception Block with 11, 19, 27 kernel branches and max-pool/residual connections.
- [x] Implement Squeeze-and-Excitation (SE) blocks.
- [x] Implement DSC base architecture.
- [x] Implement DSC-SE variant.
- [x] Implement M-DSC variant with GroupNorm, ReLU, Depthwise initial conv, and Dropout.

## Phase 2: Integration & Infrastructure (Completed)
- [x] Create PyTorch wrapper `DepthwiseCNNAdapter`.
- [x] Add `configs/benchmark.yaml`.
- [x] Register model natively in `src/models/registry.py`.
- [x] Write `train.py` script.
- [x] Write `evaluate.py` script.
- [x] Write `smoke_test.py`.
- [x] Document architecture mapping.
- [x] Create Kaggle execution notebook.

## Pending Further Optimization
- [ ] Profiling model inference time on CPU vs GPU.
- [ ] Validating parameter counts tightly match paper claims (approx 105K).