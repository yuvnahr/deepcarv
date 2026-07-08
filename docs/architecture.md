# DeepCarv Architecture

DeepCarv is organized as a model-agnostic benchmark suite. Model code,
dataset tooling, training, evaluation, experiment tracking, and reporting are
separate so new architectures can be compared without rewriting infrastructure.

## Main Areas

- `src/core/`: benchmark runner and experiment manager.
- `src/training/`: generic trainer and checkpoint helpers.
- `src/evaluation/`: metrics, evaluator, and benchmark result summaries.
- `src/models/`: registry, adapters, and future model templates.
- `src/dataset_tools/`: FFT-75 validation, statistics, leakage, plots, reports, and fingerprints.
- `src/experiment_db/`: file-backed run registry and leaderboard generation.
- `src/research/`: reproducibility metadata, output versioning, paper assets, reports, and extension hooks.

## Design Rule

Training and evaluation code should depend on interfaces and registry keys, not
on concrete model implementations. ByteRCNN remains the frozen baseline; future
models should be added through adapters, configs, and tests.
