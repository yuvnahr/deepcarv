# DeepCarv — ByteRCNN FFT-75 Baseline

Reproducible baseline implementation of **ByteRCNN** on the
**FFT-75 Scenario #1** file-fragment classification benchmark.

| Setting | Value |
|---|---|
| Dataset | FFT-75 (75 file types) |
| Scenario | #1 (all 75 classes) |
| Fragment length | 512 bytes |
| Split | 80 / 10 / 10 (train / val / test) |
| Random seed | **42** (frozen) |
| Model | ByteRCNN (PyTorch reimplementation) |

---

## Repository Structure

```
deepcarv/
├── benchmarks/
│   └── ByteRCNN/             ← reference submodule (Keras, not used directly)
├── configs/
│   └── fft75_s1_512_bytercnn.yaml   ← canonical frozen config
├── notebooks/
│   └── kaggle_bytercnn_fft75.ipynb  ← top-to-bottom Kaggle workflow
├── src/
│   ├── data/
│   │   ├── build_fft75_split.py     ← builds frozen train/val/test CSVs
│   │   └── dataset.py               ← FragmentDataset (PyTorch)
│   ├── models/
│   │   └── bytercnn_wrapper.py      ← full PyTorch reimplementation
│   ├── training/
│   │   ├── sanity_train_bytercnn.py ← 2-epoch sanity run
│   │   └── train_bytercnn.py        ← full training with early stopping
│   ├── evaluation/
│   │   └── evaluate_bytercnn.py     ← frozen test-set evaluation
│   └── utils/
│       ├── seed.py                  ← set_seed(42)
│       ├── logging.py               ← RunLogger
│       └── paths.py                 ← canonical path resolution
├── requirements.txt
├── setup.sh
├── setup.ps1
└── README.md
```

---

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

### Step 1 — Download

```bash
pip install gdown
gdown YOUR_GDRIVE_FILE_ID -O data/FFT-75.zip
# If it's a folder:
# gdown --folder YOUR_GDRIVE_FOLDER_ID -O data/raw/
```

### Step 2 — Unpack

```bash
unzip data/FFT-75.zip -d data/raw/
```

Expected layout after unpacking (class-per-subfolder):
```
data/raw/
    pdf/
        fragment_000001.bin   # exactly 512 bytes each
        ...
    exe/
    ...
```

Or flat (filename encodes class):
```
data/raw/
    pdf_000001.bin
    exe_000042.bin
    ...
```

Both layouts are auto-detected by `build_fft75_split.py`.

---

## Running the Benchmark

All scripts below read from `configs/fft75_s1_512_bytercnn.yaml` by default.

### 1. Build the Frozen Split

```bash
python -m src.data.build_fft75_split \
    --raw_dir   data/raw \
    --splits_dir data/splits
```

Writes to `data/splits/fft75_s1_512/`:  
`train.csv`, `val.csv`, `test.csv`, `class_map.json`, `manifest.json`

> **Do not re-run with a different seed.** The split is frozen at seed=42.

### 2. Sanity Check (fast, no GPU required)

```bash
python -m src.training.sanity_train_bytercnn \
    --config configs/fft75_s1_512_bytercnn.yaml
```

Runs 2 epochs on ≤2 000 samples. Checks that the loader/model/loss chain
is correctly wired. Saves `checkpoints/sanity_bytercnn_fft75.pt`.

### 3. Full Training

```bash
python -m src.training.train_bytercnn \
    --config configs/fft75_s1_512_bytercnn.yaml
```

Default settings:
- Epochs: 30 (early stopping, patience=5)
- Batch size: 256
- Optimizer: AdamW (lr=1e-3, wd=1e-4)
- Gradient clipping: 1.0

Saves:
- `checkpoints/best_bytercnn_fft75.pt` — best checkpoint
- `outputs/bytercnn_fft75/training_curves.png`
- `logs/bytercnn_fft75_<timestamp>/metrics_per_epoch.csv`

### 4. Evaluation

```bash
python -m src.evaluation.evaluate_bytercnn \
    --config configs/fft75_s1_512_bytercnn.yaml
```

Evaluates **only** on the frozen test split. Outputs to
`outputs/bytercnn_fft75/`:

| File | Contents |
|---|---|
| `metrics.json` | Accuracy, macro P/R/F1, inference time, peak GPU memory |
| `confusion_matrix.csv` | 75×75 confusion matrix |
| `per_class_metrics.csv` | Per-class precision, recall, F1, support |
| `predictions.csv` | Per-sample predictions and confidence |
| `eval_summary.txt` | Human-readable summary |

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
