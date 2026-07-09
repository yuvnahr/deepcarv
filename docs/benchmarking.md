# Benchmarking

Run a benchmark through the unified runner:

```bash
python -m src.core.runner --experiment bytercnn_fft75
```

The benchmark flow is:

1. Load and validate the composed YAML config.
2. Build datasets and dataloaders.
3. Build a model through `src.models.registry`.
4. Train with the generic trainer.
5. Evaluate with the generic evaluator.
6. Save metadata, fingerprints, checkpoints, plots, summaries, and reports.
7. Register the run in the experiment database and refresh the leaderboard.

The benchmark protocol, optimizer, scheduler, checkpoint payload, and dataset
loading are not changed by research-infrastructure modules. New infrastructure
writes sidecar files and additional reports.
