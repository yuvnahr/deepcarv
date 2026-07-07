# Experiments

Every benchmark run should produce:

- `config.yaml`
- `environment.json`
- `benchmark_metadata.json`
- `dataset_fingerprint.json`
- `checkpoint_best.pt`
- `best_metadata.json`
- `metrics.json`
- `summary.json`
- `performance.json`
- `benchmark_report.md`

Registered experiments are stored in `outputs/experiment_db/experiments.jsonl`.
Leaderboards are written to:

- `outputs/leaderboard.md`
- `outputs/leaderboard.csv`
- `outputs/leaderboard.json`

The leaderboard sorts by accuracy, macro F1, and inference latency.
