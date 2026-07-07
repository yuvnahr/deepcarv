# Evaluation

`src/evaluation/evaluator.py:Evaluator` runs inference over a DataLoader
and computes the full standard metric suite via
`src/evaluation/metrics.py` (the only module in the framework that calls
`sklearn.metrics`).

## Metrics computed

- Accuracy, top-1 (and top-k if `y_proba` is available and k < num_classes)
- Macro precision / recall / F1
- Weighted F1
- Per-class precision / recall / F1 / support
- Confusion matrix
- Inference latency per sample (ms)
- Peak GPU memory (MB, `None` on CPU)

## Standard outputs

`Evaluator.evaluate_and_save(loader, out_dir, run_name)` always writes:

| File | Contents |
|---|---|
| `metrics.json` | Aggregate metrics (accuracy, macro/weighted F1, top-k) |
| `summary.json` | Compact summary incl. latency, GPU memory, GPU name |
| `predictions.csv` | Per-sample `y_true`, `y_pred`, `confidence` |
| `confusion_matrix.csv` | Raw confusion matrix |
| `per_class_metrics.csv` | Per-class precision/recall/F1/support |
| `classification_report.txt` | `sklearn.metrics.classification_report` text |

When run through `src.core.runner`, this directory also gets
`confusion_matrix.png` (via `src/visualization/confusion.py`) alongside
the training curves.

## Running evaluation on its own

Normally `src.core.runner` calls this automatically after training
(loading `checkpoint_best.pt` first). To evaluate a specific checkpoint
standalone:

```python
from src.training.checkpointing import load_checkpoint
from src.evaluation.evaluator import Evaluator
from src.models.registry import build_model
from src.data.dataset_factory import build_datasets, build_dataloaders

model = build_model("bytercnn", num_classes=75)
load_checkpoint(Path("outputs/some_run/checkpoint_best.pt"), model)

datasets = build_datasets({"format": "npz", "root_dir": "data/FFT-75", "fragment_size": 512})
loaders = build_dataloaders(datasets, training_cfg={"batch_size": 256})

evaluator = Evaluator(model)
result = evaluator.evaluate_and_save(loaders["test"], Path("outputs/some_run"), run_name="some_run")
```

## Comparing runs / generating a report

`src/evaluation/comparison.py` reads every run's `summary.json`,
`metrics.json`, `config.yaml`, and `environment.json` into one comparison
`DataFrame`; `src/evaluation/reporting.py` turns that into
`benchmark_results.csv`, `benchmark_results.md`, and a narrative
`report.md` (best run, per-run table, and a note on cross-hardware latency
caveats). See the README's "Compare runs" section for the exact snippet.
