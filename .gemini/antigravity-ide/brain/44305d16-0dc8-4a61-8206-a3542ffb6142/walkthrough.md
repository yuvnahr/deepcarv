# Phase 1: DepthwiseCNN Implementation — Walkthrough

## What Was Built

Phase 1 implements the complete DepthwiseCNN benchmark from the paper:
> "File Fragment Type Classification Using Light-Weight Convolutional Neural Networks"

All three paper variants are implemented and fully integrated into the DeepCarv framework.

---

## Files Created / Modified

### New benchmark code

| File | Purpose |
|------|---------|
| [model.py](file:///Users/apekshaashok/Documents/isfcr/deepcarv/benchmarks/DepthwiseCNN/src/model.py) | DSC, DSC-SE, M-DSC model implementations + building blocks |
| [adapter.py](file:///Users/apekshaashok/Documents/isfcr/deepcarv/benchmarks/DepthwiseCNN/src/adapter.py) | Benchmark-local FragmentClassifier adapter |
| [__init__.py](file:///Users/apekshaashok/Documents/isfcr/deepcarv/benchmarks/DepthwiseCNN/src/__init__.py) | Package exports |
| [smoke_test.py](file:///Users/apekshaashok/Documents/isfcr/deepcarv/benchmarks/DepthwiseCNN/tests/smoke_test.py) | 32 smoke tests (building blocks, model, adapter, registry, e2e) |
| [train.py](file:///Users/apekshaashok/Documents/isfcr/deepcarv/benchmarks/DepthwiseCNN/scripts/train.py) | Training entry point |
| [evaluate.py](file:///Users/apekshaashok/Documents/isfcr/deepcarv/benchmarks/DepthwiseCNN/scripts/evaluate.py) | Evaluation entry point |
| [architecture.md](file:///Users/apekshaashok/Documents/isfcr/deepcarv/benchmarks/DepthwiseCNN/docs/architecture.md) | Architecture documentation |

### New framework code

| File | Purpose |
|------|---------|
| [depthwisecnn_adapter.py](file:///Users/apekshaashok/Documents/isfcr/deepcarv/src/models/adapters/depthwisecnn_adapter.py) | Framework-level adapter (registry → benchmark adapter) |
| [dataset_factory.py](file:///Users/apekshaashok/Documents/isfcr/deepcarv/src/data/dataset_factory.py) | Missing framework module (`build_datasets`, `build_dataloaders`) |
| [conftest.py](file:///Users/apekshaashok/Documents/isfcr/deepcarv/conftest.py) | Root conftest (sys.path for benchmark tests) |

### New configs

| File | Purpose |
|------|---------|
| [depthwisecnn.yaml](file:///Users/apekshaashok/Documents/isfcr/deepcarv/configs/models/depthwisecnn.yaml) | Model hyperparameters |
| [depthwisecnn.yaml](file:///Users/apekshaashok/Documents/isfcr/deepcarv/configs/training/depthwisecnn.yaml) | Training hyperparameters |
| [depthwisecnn_fft75_512.yaml](file:///Users/apekshaashok/Documents/isfcr/deepcarv/configs/experiments/depthwisecnn_fft75_512.yaml) | Experiment config (512-byte) |
| [depthwisecnn_fft75_4096.yaml](file:///Users/apekshaashok/Documents/isfcr/deepcarv/configs/experiments/depthwisecnn_fft75_4096.yaml) | Experiment config (4096-byte) |
| [fft75_512.yaml](file:///Users/apekshaashok/Documents/isfcr/deepcarv/configs/datasets/fft75_512.yaml) | Dataset config (512-byte) |
| [fft75_4096.yaml](file:///Users/apekshaashok/Documents/isfcr/deepcarv/configs/datasets/fft75_4096.yaml) | Dataset config (4096-byte) |

### Modified files

| File | Change |
|------|--------|
| [registry.py](file:///Users/apekshaashok/Documents/isfcr/deepcarv/src/models/registry.py) | Added `depthwisecnn` entry |
| [tests/conftest.py](file:///Users/apekshaashok/Documents/isfcr/deepcarv/tests/conftest.py) | Fixed NPZ key case (`X` → `x`, pre-existing bug) |
| [model.yaml](file:///Users/apekshaashok/Documents/isfcr/deepcarv/benchmarks/DepthwiseCNN/model.yaml) | Updated status to `implemented` |
| [benchmark.yaml](file:///Users/apekshaashok/Documents/isfcr/deepcarv/benchmarks/DepthwiseCNN/configs/benchmark.yaml) | Updated with variant info |

---

## Test Results

```
56 passed, 1 warning in 4.87s
```

- **32 new smoke tests**: Building blocks, model variants, adapter, registry, gradient, e2e pipeline
- **24 existing framework tests**: All passing (including test_trainer, test_evaluator that were previously broken by missing `dataset_factory`)

---

## Architecture Summary

```
Input [B, L] → Embedding(256, 64) → [B, 64, L]
  ↓
Block0: DSC/DSC-SE/M-DSC (64→64) + MaxPool → [B, 64, L/2]
Block1: DSC/DSC-SE/M-DSC (64→128)           → [B, 128, L/2]
Block2: DSC/DSC-SE/M-DSC (128→256) + MaxPool→ [B, 256, L/4]
Block3: DSC/DSC-SE/M-DSC (256→256)          → [B, 256, L/4]
  ↓
AdaptiveAvgPool → [B, 256]
Dropout(0.5) → Linear(256, 75) → log_softmax
```

| Variant | ~Parameters (512B input) |
|---------|--------------------------|
| DSC     | ~270K |
| DSC-SE  | ~272K |
| M-DSC   | ~290K |

All variants are well under the 1.5M parameter budget, confirming lightweight design.

---

## How to Run

```bash
# Generic DeepCarv runner (recommended):
python -m src.core.runner --experiment depthwisecnn_fft75_512

# Benchmark training script:
python benchmarks/DepthwiseCNN/scripts/train.py \
    --config configs/experiments/depthwisecnn_fft75_512.yaml

# Sanity check (no dataset required beyond structure):
python benchmarks/DepthwiseCNN/scripts/train.py \
    --data_dir data/FFT-75 --sanity --variant dsc

# DSC-SE or M-DSC via override:
python -m src.core.runner --experiment depthwisecnn_fft75_512 \
    --override model.variant=dsc_se

# Evaluate a saved checkpoint:
python benchmarks/DepthwiseCNN/scripts/evaluate.py \
    --checkpoint checkpoints/best_depthwisecnn_dsc_fft75_512b.pt \
    --data_dir data/FFT-75 --fragment_size 512

# Run smoke tests:
pytest benchmarks/DepthwiseCNN/tests/smoke_test.py -v
```

---

## Outstanding Items for Later Phases

- Kaggle notebook generation (Phase 2)
- Full training run on actual FFT-75 dataset to verify reproduced accuracy
- Parameter count documentation vs paper's reported values
- `src/models/depthwisecnn.py` template file is now superseded by the benchmark implementation (kept for compatibility)
