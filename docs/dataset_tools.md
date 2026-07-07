# FFT-75 Dataset Audit Toolkit

The dataset audit toolkit validates and reports on pre-split FFT-75 NPZ data before benchmark training runs. It is independent of ByteRCNN, the trainer, evaluator, registry, and experiment manager.

Expected dataset layout:

```text
FFT-75/
├── 512/
│   ├── train.npz
│   ├── val.npz
│   └── test.npz
└── 4096/
    ├── train.npz
    ├── val.npz
    └── test.npz
```

Each split file must contain exactly the arrays used by the benchmark: `X` fragments and `y` labels.

## Commands

Inspect one NPZ file:

```bash
python -m src.dataset_tools.cli inspect /path/to/FFT-75/512/train.npz
```

Validate a full fragment-size folder:

```bash
python -m src.dataset_tools.cli validate --root /path/to/FFT-75 --fragment-size 512
```

Analyze a validated dataset:

```bash
python -m src.dataset_tools.cli analyze --root /path/to/FFT-75 --fragment-size 512
```

Generate the complete audit report:

```bash
python -m src.dataset_tools.cli report \
  --root /path/to/FFT-75 \
  --fragment-size 512 \
  --output outputs/dataset_audit_fft75_512
```

Add `--json` to `inspect`, `validate`, `analyze`, or `report` for machine-readable output. `validate` and `report` return a non-zero exit code when validation fails, so they can be used as a quality gate in local scripts or Kaggle workflows.

## What Validation Checks

The validator checks that the selected fragment folder exists, all three split files are present, each NPZ contains `X` and `y`, sample counts match, arrays are numeric, labels are integer-valued, fragments match the requested size, byte values stay within `[0, 255]`, splits are non-empty, and label coverage is consistent across splits.

Errors are blocking. Warnings are issues worth reviewing, such as label coverage differences or detected duplicates.

## Report Outputs

The report command writes:

- `report.md`: human-readable audit summary.
- `summary.json`: validation, statistics, leakage, and plot metadata.
- `validation.json`: structured validation result.
- `class_distribution.png` and `.pdf`.
- `overall_class_distribution.png` and `.pdf`.
- `byte_histogram.png` and `.pdf`.
- `entropy_histogram.png` and `.pdf`.
- `split_sample_counts.png` and `.pdf`.
- `duplicate_summary.png` and `.pdf`.

If validation fails, the toolkit still writes `report.md`, `summary.json`, and `validation.json`, and clearly marks the report as failed. Plots and deeper statistics are generated only after the dataset passes structural validation.

## Interpreting Results

Use the validation status first. A failed validation means the dataset should not be trusted for benchmark training until the listed errors are fixed.

Review leakage and duplicate warnings before running model comparisons. Any train/val/test overlap can inflate evaluation scores and should be explained or corrected before final freezes.

Review class imbalance and label coverage notes when interpreting benchmark metrics. High imbalance does not always invalidate a dataset, but it means per-class metrics should be reported alongside aggregate scores.
