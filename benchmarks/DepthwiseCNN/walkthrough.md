# Phase 5: Final Integration and Freeze — DepthwiseCNN

The DepthwiseCNN benchmark implementation is fully complete, validated, and merge-ready.

## Files Changed
The branch is entirely additive and isolated to the `benchmarks/DepthwiseCNN/` directory, with a single hook point in the core framework registry:

#### [MODIFY] [registry.py](file:///Users/apekshaashok/Documents/isfcr/deepcarv/src/models/registry.py)
Registered `"depthwisecnn"` as a valid benchmark model. No ByteRCNN or core framework logic was altered.

#### [NEW] [src/model.py](file:///Users/apekshaashok/Documents/isfcr/deepcarv/benchmarks/DepthwiseCNN/src/model.py)
Implemented the core architecture with shared blocks (`SeparableConv1d`, `InceptionBlock`, `SEBlock`) and dynamic toggles for variants.

#### [NEW] [src/adapter.py](file:///Users/apekshaashok/Documents/isfcr/deepcarv/benchmarks/DepthwiseCNN/src/adapter.py)
Satisfied the `FragmentClassifier` protocol required by DeepCarv's `Trainer` and `Evaluator` (shapes, log-probs, predict mapping).

#### [NEW] [configs/benchmark.yaml](file:///Users/apekshaashok/Documents/isfcr/deepcarv/benchmarks/DepthwiseCNN/configs/benchmark.yaml)
Single source of truth for benchmark configuration.

#### [NEW] [scripts/train.py](file:///Users/apekshaashok/Documents/isfcr/deepcarv/benchmarks/DepthwiseCNN/scripts/train.py) & [scripts/evaluate.py](file:///Users/apekshaashok/Documents/isfcr/deepcarv/benchmarks/DepthwiseCNN/scripts/evaluate.py)
End-to-end execution scripts built on top of the generic DeepCarv framework. 

#### [NEW] [tests/smoke_test.py](file:///Users/apekshaashok/Documents/isfcr/deepcarv/benchmarks/DepthwiseCNN/tests/smoke_test.py)
Comprehensive 116-test suite that verifies everything from parameter counts to config parsing and NLLLoss compatibility.

#### [NEW] [notebooks/kaggle.ipynb](file:///Users/apekshaashok/Documents/isfcr/deepcarv/benchmarks/DepthwiseCNN/notebooks/kaggle.ipynb)
Fully autonomous Kaggle notebook integrating the new benchmark.

#### [NEW] [docs/architecture.md](file:///Users/apekshaashok/Documents/isfcr/deepcarv/benchmarks/DepthwiseCNN/docs/architecture.md), [paper/implementation_checklist.md](file:///Users/apekshaashok/Documents/isfcr/deepcarv/benchmarks/DepthwiseCNN/paper/implementation_checklist.md), [README.md](file:///Users/apekshaashok/Documents/isfcr/deepcarv/benchmarks/DepthwiseCNN/README.md)
Documentation mapping exactly how the paper is reproduced, how the codebase tracks it, and quick-start instructions.

---

## Model Architecture Implemented
Three architecture variants have been fully mapped directly from *File Fragment Type Classification Using Light-Weight Convolutional Neural Networks*:

- **DSC**: The baseline Depthwise Separable Convolution block (101K params). Uses standard Conv1d initially, InceptionBlock branches with spatial pooling, `BatchNorm`, and `Hardswish`.
- **DSC-SE**: Adds MobileNetV3-style Squeeze-and-Excitation blocks (113K params) after each InceptionBlock to gate channel responses.
- **M-DSC**: Optimization-focused variant (82K params). Replaces the initial standard Conv1d with a pure Depthwise convolution, switches `BatchNorm` to `GroupNorm(8)` for small-batch robustness, uses `ReLU`, and adds a regularizing `Dropout(p=0.2)` head.

> [!NOTE]
> Parameter counts assume `num_classes=75` and all shape traces have been confirmed end-to-end to work perfectly across 512-byte and 4096-byte input lengths.

---

## Paper Deviations & Ambiguities
We implemented the closest faithful version and kept the architecture modular. Six specific decisions were made and annotated:

1. **Shortcut when channels change**: Used a `1×1 Conv1d` (no bias, no norm) for the residual shortcut when `in_channels ≠ out_channels`.
2. **MaxPool on shortcut**: The same `MaxPool1d(4, 4)` is applied to the residual shortcut when pooling is active to maintain spatial matching.
3. **SE Reduction Ratio**: Adopted ratio=4 (MobileNetV3 default) as the paper did not specify.
4. **M-DSC first conv**: The paper specified "depthwise" but omitted mention of a pointwise projection; we stuck to a pure depthwise conv (`groups=32`).
5. **GroupNorm group count**: Used `num_groups=8` (divides cleanly into the network's 32, 64, and 128 channel widths).
6. **Dropout placement (M-DSC)**: Placed after GAP and before the final `1x1 Conv1d` classifier, matching MobileNetV3 architecture norms.

---

## How to Run Local Training
Fragment size (512 vs 4096) and architecture variant are controlled exclusively via `benchmark.yaml` (or CLI overrides).

```bash
# Default (4096-byte DSC)
python -m benchmarks.DepthwiseCNN.scripts.train \
    --config benchmarks/DepthwiseCNN/configs/benchmark.yaml

# Override via CLI (e.g. 512-byte DSC-SE)
python -m benchmarks.DepthwiseCNN.scripts.train \
    --config benchmarks/DepthwiseCNN/configs/benchmark.yaml \
    --fragment-size 512 \
    --variant dsc-se

# Evaluate
python -m benchmarks.DepthwiseCNN.scripts.evaluate \
    --config benchmarks/DepthwiseCNN/configs/benchmark.yaml \
    --checkpoint outputs/DepthwiseCNN/checkpoint_best.pt
```

Outputs are automatically written to `outputs/DepthwiseCNN/`.

---

## How to Run Kaggle Training
The provided notebook `benchmarks/DepthwiseCNN/notebooks/kaggle.ipynb` handles all dependency installations, data mounting, script execution, and reporting.

1. **Configuration**: The notebook's `Cell § 0` contains the runtime config. Edit `FRAGMENT_SIZE = 512` and `VARIANT = 'dsc'` here. No other cells need modifying.
2. **Data**: The notebook automatically downloads the FFT-75 dataset layout using either an attached Kaggle dataset (`fft-75-npz`) or a Google Drive fallback download.
3. **Execution**: Simply "Run All". It will validate the NPZ structure, run a 5-epoch sanity pass on a tiny subset, launch the full end-to-end `train.py` run, evaluate, and spit out the full metrics, classification report, and confusion matrix heatmap.

---

## Known Limitations
- The architecture correctly implements the components, but actual model accuracy (`~79%`) and throughput relative to the 164M FLOPs claim have not yet been evaluated at scale on FFT-75.
- The M-DSC architecture's pure depthwise (non-pointwise) initial convolution might underfit compared to the DSC base variant. If underfitting is observed, the choice annotated as "Decision 4" in `model.py` can be updated with a single-line patch.

---

## Next Steps
1. **Merge the Branch**: The branch `eval-depthwiseCNN` is stable, has 116 passing tests, properly implemented configs, and complete architectural documentation.
2. **Run Full Evaluation**: Launch the Kaggle notebook in an actual Kaggle GPU runtime to train over 30 epochs and log the actual performance benchmarks for the DSC-SE variant. 
3. **Profilers**: Optionally, run FLOPs profiling and CPU vs GPU inference latency benchmarking on the finalized architecture using DeepCarv tools.
