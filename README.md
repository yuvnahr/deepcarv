# DeepCarv — Model-Agnostic File-Fragment Classification Benchmark

Reproducible benchmark framework for file-fragment classification and
carving models. The **ByteRCNN** baseline on **FFT-75 Scenario #1** is the
first model registered in the framework; CarveFormer, ByteNet, Depthwise
CNNs, Hierarchical CNNs, and DeepCarv are intended to plug into the same
trainer/evaluator without any workflow changes.

| Setting | Value |
|---|---|
| Dataset | FFT-75 (75 file types), pre-split NPZ (`train.npz`/`val.npz`/`test.npz`) |
| Scenario | #1 (all 75 classes) |
| Fragment length | 512 or 4096 bytes (config-selectable) |
| Random seed | **42** (frozen for the benchmark) |
| Baseline model | ByteRCNN (PyTorch reimplementation) |

See **[docs/architecture.md](docs/architecture.md)** for how the framework
is put together, or jump straight to
**[docs/benchmark.md](docs/benchmark.md)** to run something.

---

## Repository Structure

```
deepcarv/
├── benchmarks/
│   └── ByteRCNN/                     ← frozen reference submodule (read-only)
├── configs/                          ← composable YAML configs
│   ├── datasets/                     ← fft75_512.yaml, fft75_4096.yaml, ...
│   ├── models/                       ← bytercnn.yaml, carveformer.yaml (stub), ...
│   ├── training/                     ← default.yaml
│   ├── evaluation/                   ← benchmark.yaml
│   └── experiments/                  ← composes the above into one run
├── src/
│   ├── core/                         ← interfaces, experiment_manager, runner (entrypoint)
│   ├── data/                         ← npz_dataset, dataset_factory, validators
│   ├── models/
│   │   ├── registry.py               ← MODEL_REGISTRY — the only place models are built
│   │   ├── bytercnn_wrapper.py       ← PyTorch ByteRCNN implementation
│   │   └── adapters/                 ← bytercnn (real), carveformer/bytenet/deepcarv (stubs)
│   ├── training/                     ← generic trainer, callbacks, checkpointing
│   ├── evaluation/                   ← evaluator, metrics, comparison, reporting
│   ├── visualization/                ← plots, confusion matrix rendering
│   └── utils/                        ← config, device, environment, logging, seed, paths, statistics
├── datasets/ logs/ models/ outputs/ results/ cache/ checkpoints/   ← runtime data (gitignored where noted)
├── docs/                             ← benchmark.md, configuration.md, training.md, evaluation.md, architecture.md
├── tests/                            ← smoke tests (config, dataset, registry, trainer, evaluator)
├── notebooks/
│   └── kaggle_bytercnn_fft75.ipynb   ← Kaggle workflow
├── requirements.txt
├── setup.sh / setup.ps1
└── README.md
```



## Prerequisites

- Python ≥ 3.10
- PyTorch ≥ 2.2 (with CUDA for GPU training)
- The FFT-75 dataset hosted on your Google Drive

---

## Local Setup

### Linux / macOS

```bash
bash setup.sh
source venv/bin/activate
```

### Windows (PowerShell)

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
.\venv\Scripts\Activate.ps1
```

---

## Data Preparation

The dataset lives on **Google Drive** and is never committed to this repo.

The framework consumes the **already pre-split** FFT-75 NPZ files directly —
it does not generate splits or read CSVs. Expected layout after download:

```
data/FFT-75/
├── 512/
│   ├── train.npz   # {"X": [N, 512] uint8, "y": [N] int}
│   ├── val.npz
│   └── test.npz
└── 4096/
    ├── train.npz
    ├── val.npz
    └── test.npz
```

```bash
pip install gdown
gdown YOUR_GDRIVE_FILE_ID -O data/FFT-75.zip
unzip data/FFT-75.zip -d data/
```

Point `configs/datasets/fft75_512.yaml` (`root_dir`) at wherever you unpack
this if it isn't `data/FFT-75`. Switching between 512- and 4096-byte
fragments is a config change (`fft75_512` → `fft75_4096`), never a code
change. See **[docs/configuration.md](docs/configuration.md)** for details.

---

## Running the Benchmark

Everything runs through the single unified entrypoint,
`src.core.runner`, which composes a config, builds the dataset, builds
the model via the registry, trains, evaluates, and writes all outputs.
There is no per-model script.

### 1. Smoke test (no GPU, no full dataset required)

```bash
pytest tests/ -q
```

### 2. Full run — ByteRCNN on FFT-75 (512-byte fragments)

```bash
python -m src.core.runner --experiment configs/experiments/bytercnn_fft75.yaml
# or, by name:
python -m src.core.runner --experiment bytercnn_fft75
```

Override any composed config value from the CLI without editing files:

```bash
python -m src.core.runner --experiment bytercnn_fft75 \
    --override training.epochs=5 --override training.batch_size=128
```

### 3. Switch to 4096-byte fragments

```bash
python -m src.core.runner --experiment bytercnn_fft75_4096
```

### 4. Outputs

Every run writes a self-contained, reproducible directory to
`outputs/<run_name>_<timestamp>/`:

| File | Contents |
|---|---|
| `config.yaml` | Composed config snapshot |
| `environment.json`, `git_commit.txt` | Reproducibility metadata |
| `train.log`, `checkpoint_best.pt`, `checkpoint_last.pt` | Training artifacts |
| `loss_curve.png`, `accuracy_curve.png`, `lr_curve.png`, `confusion_matrix.png` | Plots |
| `metrics.json`, `summary.json` | Aggregate metrics |
| `predictions.csv`, `confusion_matrix.csv`, `per_class_metrics.csv`, `classification_report.txt` | Evaluation detail |

### 5. Compare runs / generate a report

```python
from pathlib import Path
from src.evaluation.comparison import discover_runs, compare_runs
from src.evaluation.reporting import write_benchmark_results, generate_report

runs = discover_runs(Path("outputs"))
df = compare_runs(runs)
write_benchmark_results(df, Path("results"))   # benchmark_results.csv / .md
generate_report(df, Path("results/report.md"))
```

Full details: **[docs/benchmark.md](docs/benchmark.md)**,
**[docs/training.md](docs/training.md)**,
**[docs/evaluation.md](docs/evaluation.md)**.

---

## Kaggle Workflow

1. Open `notebooks/kaggle_bytercnn_fft75.ipynb` in Kaggle.
2. In **Cell 3**, set `GDRIVE_FILE_ID` to your Google Drive file/folder ID.
3. In **Cell 2**, update the `git clone` URL to your repo (or attach it as a
   Kaggle dataset).
4. Run all cells top-to-bottom.

The notebook:
- Downloads the dataset with `gdown`
- Unpacks and validates 512-byte fragments
- Builds the frozen split
- Runs sanity training (2 epochs)
- Runs full training (30 epochs, early stopping)
- Evaluates on the frozen test set
- Bundles all results into a ZIP

---

## Architecture Reference

PyTorch reimplementation of ByteRCNN (Keras original: [kristian-fer/ByteRCNN](https://github.com/kristian-fer/ByteRCNN)).

```
Input: [B, 512]   ← byte sequences (values 0–255)
   │
Embedding(256, 16)
   │
   ├──→ BiGRU (2 layers, hidden=64, bidirectional) → last hidden → [B, 128]
   │
   └──→ 4 × Conv1d branches (kernels=[9,27,40,65], filters=64) → GlobalMaxPool → [B, 256]
   │
Concat [B, 384]
   │
Linear(384→1024) → BN → ReLU → Dropout(0.5)
   │
Linear(1024→512) → BN → ReLU → Dropout(0.5)
   │
Linear(512→75) → LogSoftmax
```

Smoke-test without data:
```bash
python src/models/bytercnn_wrapper.py
```

---

## Key Implementation Constraints

- **Split never generated inside training code.**
- **Seed=42, fragment_size=512, num_classes=75 are hard-frozen.**
- **Third-party code is isolated** under `benchmarks/ByteRCNN/` (submodule).
- **Dataset is never committed** to the repo.
- **No silent re-generation** of class maps or splits.

---

## Citation

If you use this benchmark baseline, please also cite the original ByteRCNN paper
and the FFT-75 dataset:

```
@inproceedings{bytercnn,
  title     = {ByteRCNN: ...},
  author    = {Srivastava et al.},
  year      = {2021}
}
```

*(Update with the exact citation from your reference.)*
