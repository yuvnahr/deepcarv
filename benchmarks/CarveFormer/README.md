# CarveFormer

Faithful reproduction of **CarveFormer** (Guzhov & Wirth, ECCWS 2025) —
Swin Transformer V2 Tiny with its image front-end replaced by a 96-d byte
embedding — integrated into the DeepCarv benchmark framework.

## Paper

*Transformer-Based File Fragment Type Classification for File Carving in
Digital Forensics.* See `./paper/notes.md` and `./docs/architecture.md` for the
architecture mapping, assumptions, and documented deviations.

## Goal

Reproduce the original benchmark architecture faithfully and run it through the
shared DeepCarv trainer/evaluator on FFT-75 (512- and 4096-byte fragments),
switchable by config only.

## Layout

```
benchmarks/CarveFormer/
├── src/model.py       # framework-independent CarveFormer + build_carveformer
├── src/adapter.py     # CarveFormerAdapter (FragmentClassifier)
├── configs/benchmark.yaml
├── scripts/train.py       # thin wrapper over the shared Trainer
├── scripts/evaluate.py    # thin wrapper over the shared Evaluator
├── notebooks/carveformer_fft75_kaggle.ipynb
├── tests/smoke_test.py
├── docs/architecture.md
└── paper/{notes.md, implementation_checklist.md, phase0_recon.md}
```

## Install

```bash
pip install "timm>=1.0.0"          # Swin V2 backbone (already in requirements)
```

## Data

The FFT-75 dataset (a few GB) is **not** stored in the git repo. It lives in
Google Drive and is fetched at runtime with `gdown` (see the Kaggle notebook).
The benchmark then reads the pre-split NPZ files directly — no CSV, no split
regeneration:

```
{root_dir}/{fragment_size}/{train,val,test}.npz    # each: X [N, L] uint8, y [N] int
```

On Kaggle, set `GDRIVE_FILE_ID` (a single zip) or `GDRIVE_FOLDER_ID` (a shared
folder) in the notebook's config cell; it downloads and extracts FFT-75 into
`DATA_DIR` automatically (idempotent — skips if already present). For local
runs, point `--data-dir` at wherever you unpacked the dataset.

## Run locally

```bash
# Train (512-byte fragments; paper hyperparameters from the config)
python -m benchmarks.CarveFormer.scripts.train \
    --config benchmarks/CarveFormer/configs/benchmark.yaml \
    --data-dir data/FFT-75

# Switch to 4096-byte fragments (config-only change; or CLI override)
python -m benchmarks.CarveFormer.scripts.train \
    --config benchmarks/CarveFormer/configs/benchmark.yaml \
    --data-dir data/FFT-75 --fragment-size 4096

# Evaluate a checkpoint
python -m benchmarks.CarveFormer.scripts.evaluate \
    --config benchmarks/CarveFormer/configs/benchmark.yaml \
    --checkpoint benchmarks/CarveFormer/outputs/checkpoint_best.pt

# Quick offline smoke test (no GPU, no pretrained weights, no dataset)
pytest benchmarks/CarveFormer/tests/smoke_test.py
```

## Run on Kaggle

Open `notebooks/carveformer_fft75_kaggle.ipynb`, attach the FFT-75 dataset,
set `FRAGMENT_SIZE` in the config cell, and run top-to-bottom. It installs deps,
verifies the NPZ layout, sanity-trains, full-trains, evaluates, and saves the
standardized outputs.

## Outputs

The shared evaluator writes the standard set to `outputs/`: `metrics.json`,
`summary.json`, `predictions.csv`, `confusion_matrix.csv`,
`per_class_metrics.csv`, `classification_report.txt`, plus checkpoints and
training-curve plots.

## Status

- [x] Paper reviewed
- [x] Architecture implemented (SwinV2-Tiny + 96-d byte embedding)
- [x] Adapter integrated (registry key `carveformer`)
- [x] Config added (paper hyperparameters)
- [x] Smoke test passing
- [x] Train / evaluate scripts wired to the shared framework
- [x] Kaggle notebook complete
- [ ] Full FFT-75 training run (needs GPU + real dataset)

## Branch

`eval-CarveFormer`
