# Reproducibility

DeepCarv records the environment and data used for each benchmark run.

`benchmark_metadata.json` captures:

- dataset hashes and version
- git commit
- Python, PyTorch, and CUDA versions
- GPU, CPU, RAM, hostname, OS
- command line
- runtime

`dataset_fingerprint.json` captures exact split file hashes. Checkpoint sidecar
metadata records epoch, metrics, optimizer name, scheduler name, training config,
dataset fingerprint, parameter count, and best validation score.

Use `python scripts/repo_check.py` before contributing or running experiments.
