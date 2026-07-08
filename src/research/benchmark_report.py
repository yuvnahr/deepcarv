"""Publication-oriented benchmark report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_benchmark_report(
    run_dir: Path | str,
    output_path: Path | str | None = None,
) -> Path:
    """Generate benchmark_report.md from standard run artifacts."""
    run = Path(run_dir)
    out = Path(output_path or run / "benchmark_report.md")
    summary = _read_json(run / "summary.json")
    metadata = _read_json(run / "benchmark_metadata.json")
    performance = _read_json(run / "performance.json")

    lines = [
        "# DeepCarv Benchmark Report",
        "",
        "## Dataset",
        "",
        f"- Dataset: `{summary.get('dataset', metadata.get('dataset_version', 'unknown'))}`",
        f"- Fragment size: `{summary.get('fragment_size', 'unknown')}`",
        "",
        "## Model",
        "",
        f"- Model: `{summary.get('model', 'unknown')}`",
        f"- Parameter count: `{summary.get('n_parameters', performance.get('parameter_count'))}`",
        "",
        "## Training",
        "",
        f"- Best epoch: `{summary.get('best_epoch', 'unknown')}`",
        f"- Best validation score: `{summary.get('best_monitor_value', 'unknown')}`",
        "",
        "## Evaluation",
        "",
        f"- Accuracy: `{summary.get('accuracy', 'unknown')}`",
        f"- Macro F1: `{summary.get('macro_f1', 'unknown')}`",
        f"- Weighted F1: `{summary.get('weighted_f1', 'unknown')}`",
        "",
        "## Hardware",
        "",
        f"- GPU: `{metadata.get('gpu', 'unknown')}`",
        f"- CPU: `{metadata.get('cpu', 'unknown')}`",
        f"- RAM GB: `{metadata.get('ram_gb', 'unknown')}`",
        "",
        "## Environment",
        "",
        f"- Python: `{metadata.get('python_version', 'unknown')}`",
        f"- Torch: `{metadata.get('torch_version', 'unknown')}`",
        f"- CUDA: `{metadata.get('cuda_version', 'unknown')}`",
        f"- Git: `{metadata.get('git_hash', 'unknown')}`",
        "",
        "## Performance",
        "",
        f"- GPU memory MB: `{performance.get('peak_gpu_memory_mb', 'unknown')}`",
        f"- Latency ms/sample: `{summary.get('latency_ms_per_sample', 'unknown')}`",
        f"- Samples/sec: `{performance.get('samples_per_sec', 'unknown')}`",
        "",
        "## Plots",
        "",
        "- Loss: `loss_curve.png`",
        "- Accuracy: `accuracy_curve.png`",
        "- Learning rate: `lr_curve.png`",
        "- Confusion matrix: `confusion_matrix.png`",
        "",
        "## Recommendations",
        "",
        "- Review per-class metrics before comparing models on imbalanced data.",
        "- Compare latency only on matching hardware.",
        "",
        "## Warnings",
        "",
        "- This report is assembled from available artifacts; missing fields are marked unknown.",
        "",
        "## Future Work",
        "",
        "- Add model-specific analysis notes once additional architectures are implemented.",
    ]
    out.write_text("\n".join(lines) + "\n")
    return out


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())
