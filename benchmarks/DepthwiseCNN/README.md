# DepthwiseCNN

Lightweight file-fragment type classifier for the FFT-75 benchmark, implemented in PyTorch.

**Paper:** *File Fragment Type Classification Using Light-Weight Convolutional Neural Networks*  
**Branch:** `eval-depthwiseCNN`  
**Status:** Phase 4 complete — all tests pass.

---

## Quick start

All commands are run from the **repository root** with `PYTHONPATH=.`.

### 1. Run smoke tests

```bash
PYTHONPATH=. pytest benchmarks/DepthwiseCNN/tests/smoke_test.py -v
```

Expected: **75 passed** in < 2 s (no GPU needed).

### 2. Train

```bash
# Default: DSC variant, 4096-byte fragments
python -m benchmarks.DepthwiseCNN.scripts.train \
    --config benchmarks/DepthwiseCNN/configs/benchmark.yaml

# 512-byte fragments, DSC-SE variant
python -m benchmarks.DepthwiseCNN.scripts.train \
    --config benchmarks/DepthwiseCNN/configs/benchmark.yaml \
    --fragment-size 512 \
    --variant dsc-se

# M-DSC, override epochs without editing YAML
python -m benchmarks.DepthwiseCNN.scripts.train \
    --config benchmarks/DepthwiseCNN/configs/benchmark.yaml \
    --variant m-dsc \
    --epochs 50

# Resume from last checkpoint
python -m benchmarks.DepthwiseCNN.scripts.train \
    --config benchmarks/DepthwiseCNN/configs/benchmark.yaml \
    --resume outputs/DepthwiseCNN/checkpoint_last.pt
```

### 3. Evaluate

```bash
# Test split (default)
python -m benchmarks.DepthwiseCNN.scripts.evaluate \
    --config benchmarks/DepthwiseCNN/configs/benchmark.yaml \
    --checkpoint outputs/DepthwiseCNN/checkpoint_best.pt

# Validation split, custom output directory
python -m benchmarks.DepthwiseCNN.scripts.evaluate \
    --config benchmarks/DepthwiseCNN/configs/benchmark.yaml \
    --checkpoint outputs/DepthwiseCNN/checkpoint_best.pt \
    --split val \
    --out-dir outputs/DepthwiseCNN/eval_val
```

### 4. Use via registry (Python)

```python
from src.models.registry import build_model
import torch

model = build_model("depthwisecnn", num_classes=75, variant="dsc-se")
x = torch.randint(0, 256, (4, 512))   # [B, L] byte integers
log_probs = model(x)                   # [4, 75] log-probabilities
```

---

## Fragment-size and variant switch

**Everything is driven by `configs/benchmark.yaml`.**  
The only edits needed to run a different benchmark scenario:

| Scenario | Key to change | Value |
|----------|--------------|-------|
| 512-byte fragments | `dataset.fragment_size` | `512` |
| 4096-byte fragments | `dataset.fragment_size` | `4096` |
| Baseline DSC | `model.kwargs.variant` | `"dsc"` |
| With squeeze-and-excitation | `model.kwargs.variant` | `"dsc-se"` |
| Modified / inference-optimised | `model.kwargs.variant` | `"m-dsc"` |

Alternatively, pass `--fragment-size` or `--variant` on the CLI to override the YAML without editing it.

---

## Architecture summary

Three variants share the same macro-structure:

```
Embedding(256, 32)  →  FirstConv(k=19, s=2)  →
InceptionBlock(32→64,  pool) → [SE]  →
InceptionBlock(64→64,  no pool) → [SE]  →
InceptionBlock(64→128, pool) → [SE]  →
GlobalAveragePool  →  [Dropout]  →  Classifier(128→num_classes)  →  log_softmax
```

| Variant | Norm | Activation | First conv | SE | Dropout |
|---------|------|-----------|----------|-----|---------|
| **DSC** | BatchNorm | Hardswish | Standard | — | — |
| **DSC-SE** | BatchNorm | Hardswish | Standard | ✓ | — |
| **M-DSC** | GroupNorm(8) | ReLU | Depthwise | — | ✓ (p=0.2) |

Parameter counts (75 classes):

| Variant | Parameters |
|---------|----------:|
| DSC | 101,291 |
| DSC-SE | 113,579 |
| M-DSC | 82,443 |

For the full architecture specification including exact shapes, layer-by-layer breakdown, and paper ambiguity decisions, see [`docs/architecture.md`](docs/architecture.md).

---

## Data layout

FFT-75 must be structured as:

```
data/FFT-75/
├── 512/
│   ├── train.npz
│   ├── val.npz
│   └── test.npz
└── 4096/
    ├── train.npz
    ├── val.npz
    └── test.npz
```

Each `.npz` file contains two arrays: `x` (byte fragments, dtype uint8) and `y` (integer labels).

Override the default data path via:

```bash
export DEEPCARV_DATA_ROOT=/path/to/your/data
# or set dataset.root_dir in benchmark.yaml
```

---

## Kaggle

Open `notebooks/kaggle.ipynb` in a Kaggle notebook environment.  The notebook:

1. Installs dependencies and clones the repo
2. Downloads FFT-75 (Kaggle dataset input or Google Drive)
3. Validates all three NPZ splits
4. Runs a 5-epoch sanity pass on a small subset
5. Launches full training
6. Evaluates the best checkpoint on the test split
7. Produces a benchmark card, per-class table, and confusion matrix heatmap

Only **§ 0** needs editing — set `FRAGMENT_SIZE` and `VARIANT` there.

---

## Evaluation outputs

The evaluator writes the following to `paths.eval_outputs` (default: `outputs/DepthwiseCNN/eval/`):

| File | Description |
|------|-------------|
| `metrics.json` | Full metric dictionary (accuracy, F1, …) |
| `summary.json` | Compact summary with latency and GPU stats |
| `predictions.csv` | Per-sample `y_true`, `y_pred`, `confidence` |
| `confusion_matrix.csv` | 75×75 confusion matrix |
| `per_class_metrics.csv` | Precision, recall, F1 per class |
| `classification_report.txt` | Scikit-learn classification report |

---

## File structure

```
benchmarks/DepthwiseCNN/
├── configs/
│   └── benchmark.yaml          ← single source of truth for all settings
├── docs/
│   └── architecture.md         ← exact architecture spec + decisions
├── notebooks/
│   └── kaggle.ipynb            ← end-to-end Kaggle notebook
├── paper/
│   └── implementation_checklist.md
├── scripts/
│   ├── train.py                ← training entry-point
│   └── evaluate.py             ← evaluation entry-point
├── src/
│   ├── __init__.py
│   ├── model.py                ← DSC / DSC-SE / M-DSC architectures
│   └── adapter.py              ← FragmentClassifier wrapper + registry factory
└── tests/
    └── smoke_test.py           ← 75 automated tests (model + adapter + registry)
```

---

## Implementation status

See [`paper/implementation_checklist.md`](paper/implementation_checklist.md) for the full checklist.

| Phase | Status |
|-------|--------|
| Paper review | ✅ |
| Architecture (model.py) | ✅ |
| Adapter + registry | ✅ |
| Config (benchmark.yaml) | ✅ |
| Training script | ✅ |
| Evaluation script | ✅ |
| Smoke tests (75 passed) | ✅ |
| Architecture docs | ✅ |
| Kaggle notebook | ✅ |
| Training verified | 🔲 (requires FFT-75 data) |
| Evaluation verified | 🔲 (requires FFT-75 data) |
