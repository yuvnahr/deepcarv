"""Markdown report generation for FFT-75 dataset audits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.dataset_tools.io import ensure_output_dir, write_json
from src.dataset_tools.plots import generate_all_plots
from src.dataset_tools.statistics import DatasetStatistics, compute_dataset_statistics
from src.dataset_tools.validator import DatasetValidationResult, validate_dataset


@dataclass(frozen=True)
class DatasetAuditReport:
    """Artifacts produced by one dataset audit run."""

    output_dir: str
    report_path: str
    summary_path: str
    validation_path: str
    plot_paths: dict[str, list[str]]
    validation: DatasetValidationResult
    statistics: DatasetStatistics | None


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(item) for item in row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def _recommendations(
    validation: DatasetValidationResult,
    stats: DatasetStatistics | None,
) -> list[str]:
    notes: list[str] = []
    if validation.errors:
        notes.extend(validation.errors)
    notes.extend(validation.warnings)
    if stats is not None:
        notes.extend(stats.leakage_report.warnings)
        if stats.imbalance_severity == "high":
            notes.append("Class imbalance is high; report per-class metrics in model comparisons.")
        if not stats.label_coverage_consistency:
            notes.append("One or more splits do not cover the full label space.")
    if not notes:
        notes.append("No blocking dataset quality issues were detected.")
    return notes


def _render_markdown(
    root: Path,
    fragment_size: int,
    validation: DatasetValidationResult,
    stats: DatasetStatistics | None,
    plot_paths: dict[str, list[str]],
) -> str:
    lines = [
        f"# FFT-75 Dataset Audit ({fragment_size})",
        "",
        f"- Dataset path: `{root}`",
        f"- Fragment size: `{fragment_size}`",
        f"- Validation status: `{'PASS' if validation.ok else 'FAIL'}`",
        "",
    ]

    if stats is not None:
        split_rows = [
            [
                split,
                split_stats.sample_count,
                split_stats.class_count,
                f"{split_stats.class_imbalance_ratio:.3f}",
                f"{split_stats.mean_fragment_entropy:.3f}",
                f"{split_stats.duplicate_rate:.4f}",
            ]
            for split, split_stats in stats.splits.items()
        ]
        lines.extend(
            [
                "## Summary",
                "",
                f"- Total samples: `{stats.total_sample_count}`",
                f"- Shared label space size: `{len(stats.shared_label_space)}`",
                "- Label coverage consistent: "
                f"`{'yes' if stats.label_coverage_consistency else 'no'}`",
                f"- Imbalance severity: `{stats.imbalance_severity}`",
                f"- Cross-split duplicate rate: `{stats.duplicate_rate_across_splits:.6f}`",
                "",
                "## Split Statistics",
                "",
                _markdown_table(
                    [
                        "Split",
                        "Samples",
                        "Classes",
                        "Imbalance",
                        "Mean entropy",
                        "Duplicate rate",
                    ],
                    split_rows,
                ),
                "",
                "## Byte Statistics",
                "",
            ]
        )
        byte_rows = [
            [
                split,
                f"{split_stats.mean_fragment_value:.3f}",
                f"{split_stats.std_fragment_value:.3f}",
                split_stats.min_byte_value,
                split_stats.max_byte_value,
            ]
            for split, split_stats in stats.splits.items()
        ]
        lines.extend(
            [
                _markdown_table(["Split", "Mean", "Std", "Min", "Max"], byte_rows),
                "",
                "## Leakage Summary",
                "",
            ]
        )
        overlap_rows = [
            [
                f"{item.left_split}/{item.right_split}",
                item.overlapping_hashes,
                item.overlapping_samples_left,
                item.overlapping_samples_right,
            ]
            for item in stats.leakage_report.overlaps
        ]
        pair_overlap_rows = [
            [
                f"{item.left_split}/{item.right_split}",
                item.overlapping_hashes,
                item.overlapping_samples_left,
                item.overlapping_samples_right,
            ]
            for item in stats.leakage_report.pair_overlaps
        ]
        lines.extend(
            [
                "Exact fragment overlap:",
                "",
                _markdown_table(
                    ["Split pair", "Overlap hashes", "Left samples", "Right samples"],
                    overlap_rows,
                ),
                "",
                "Exact `(fragment, label)` pair overlap:",
                "",
                _markdown_table(
                    ["Split pair", "Overlap hashes", "Left samples", "Right samples"],
                    pair_overlap_rows,
                ),
                "",
                "## Class Counts",
                "",
                _markdown_table(
                    ["Class", "Samples"],
                    [[label, count] for label, count in stats.overall_class_distribution.items()],
                ),
                "",
            ]
        )

    if validation.errors:
        lines.extend(["## Validation Errors", ""])
        lines.extend(f"- {error}" for error in validation.errors)
        lines.append("")
    if validation.warnings:
        lines.extend(["## Validation Warnings", ""])
        lines.extend(f"- {warning}" for warning in validation.warnings)
        lines.append("")

    lines.extend(["## Recommendations", ""])
    lines.extend(f"- {note}" for note in _recommendations(validation, stats))
    lines.append("")

    if plot_paths:
        lines.extend(["## Figures", ""])
        for name, paths in plot_paths.items():
            png_paths = [Path(path).name for path in paths if path.endswith(".png")]
            if png_paths:
                title = name.replace("_", " ").title()
                lines.extend([f"### {title}", "", f"![{title}]({png_paths[0]})", ""])

    return "\n".join(lines).rstrip() + "\n"


def generate_report(
    root: Path | str,
    fragment_size: int,
    output_dir: Path | str,
) -> DatasetAuditReport:
    """Run validation, analysis, plotting, JSON export, and markdown report generation."""
    dataset_root = Path(root)
    out = ensure_output_dir(output_dir)
    validation = validate_dataset(dataset_root, fragment_size)
    validation_path = out / "validation.json"
    write_json(validation_path, validation)

    stats: DatasetStatistics | None = None
    plot_paths: dict[str, list[str]] = {}
    if validation.ok:
        stats = compute_dataset_statistics(dataset_root, fragment_size)
        plot_paths = generate_all_plots(stats, out)

    summary_path = out / "summary.json"
    write_json(
        summary_path,
        {
            "validation": validation,
            "statistics": stats,
            "plots": plot_paths,
        },
    )

    report_path = out / "report.md"
    report_path.write_text(
        _render_markdown(dataset_root, fragment_size, validation, stats, plot_paths)
    )

    return DatasetAuditReport(
        output_dir=str(out),
        report_path=str(report_path),
        summary_path=str(summary_path),
        validation_path=str(validation_path),
        plot_paths=plot_paths,
        validation=validation,
        statistics=stats,
    )
