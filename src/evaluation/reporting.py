"""
src/evaluation/reporting.py
--------------------------------
Turns a run comparison DataFrame (src.evaluation.comparison) into
benchmark_results.csv, benchmark_results.md, and a narrative report.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_benchmark_results(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "benchmark_results.csv", index=False)
    (out_dir / "benchmark_results.md").write_text(_to_markdown_table(df))


def _to_markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        # tabulate not installed — fall back to a manual, still-valid table
        headers = list(df.columns)
        lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
        for _, row in df.iterrows():
            cells = []
            for v in row:
                cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)


def generate_report(df: pd.DataFrame, out_path: Path, title: str = "DeepCarv Benchmark Report") -> None:
    """Write a narrative markdown report summarizing all compared runs."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [f"# {title}", ""]

    if df.empty:
        lines.append("_No runs found to report on._")
        out_path.write_text("\n".join(lines) + "\n")
        return

    best = df.iloc[0]
    lines += [
        f"Compared **{len(df)}** run(s) across model(s): "
        f"{', '.join(sorted(df['model'].dropna().unique()))}.",
        "",
        f"**Best by accuracy:** `{best['run_name']}` "
        f"({best['model']}, fragment_size={best.get('fragment_size')}) "
        f"— accuracy={best['accuracy']:.4f}, macro_f1={best.get('macro_f1', float('nan')):.4f}",
        "",
        "## Results",
        "",
        _to_markdown_table(df),
        "",
        "## Notes",
        "",
        "- `latency_ms_per_sample` and `peak_gpu_memory_mb` are measured on the "
        "hardware recorded in each run's `environment.json`; cross-hardware "
        "comparisons of these two columns should be read with that in mind.",
        "- Per-class results and confusion matrices are available in each "
        "run's own output directory (`per_class_metrics.csv`, "
        "`confusion_matrix.csv`).",
    ]

    out_path.write_text("\n".join(lines) + "\n")
