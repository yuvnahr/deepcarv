# Running a Benchmark

Quick-start for someone who hasn't read any other doc.

## 1. Install

```bash
bash setup.sh
source venv/bin/activate
```

## 2. Get the data

Download and unpack the FFT-75 NPZ splits so you have:

```
data/FFT-75/512/{train,val,test}.npz
data/FFT-75/4096/{train,val,test}.npz
```

(See the README's "Data Preparation" section for the Google Drive
download commands.) Nothing under `data/` is committed to the repo.

## 3. Verify everything is wired correctly (no full dataset needed)

```bash
pytest tests/ -q
```

This uses a tiny synthetic NPZ fixture (`tests/conftest.py`) to check
config composition, dataset loading + validation, the model registry,
one training epoch, and one evaluation pass — all in a few seconds, on
CPU, without touching the real dataset.

## 4. Run the frozen ByteRCNN baseline

```bash
python -m src.core.runner --experiment bytercnn_fft75
```

This is the **only** command needed to train + evaluate the baseline. It:

1. Composes `configs/experiments/bytercnn_fft75.yaml`
   (→ `datasets/fft75_512.yaml` + `models/bytercnn.yaml` +
   `training/default.yaml` + `evaluation/benchmark.yaml`)
2. Loads the official `train.npz` / `val.npz` / `test.npz` (no CSVs, no
   split regeneration)
3. Builds ByteRCNN via the model registry
4. Trains with early stopping, checkpointing, and AMP (if CUDA is
   available)
5. Evaluates the best checkpoint on the test split
6. Writes everything to `outputs/bytercnn_fft75_<timestamp>/`

## 5. Try the other fragment size

```bash
python -m src.core.runner --experiment bytercnn_fft75_4096
```

## 6. Compare runs

```python
from pathlib import Path
from src.evaluation.comparison import discover_runs, compare_runs
from src.evaluation.reporting import write_benchmark_results, generate_report

runs = discover_runs(Path("outputs"))
df = compare_runs(runs)
write_benchmark_results(df, Path("results"))
generate_report(df, Path("results/report.md"))
```

## What's next (not implemented yet)

CarveFormer, ByteNet, and DeepCarv are registered as stubs
(`src/models/registry.py`) — trying to run them today raises
`NotImplementedError` with instructions on what to fill in. See
`docs/architecture.md` → "Extending the framework" for the exact steps.
