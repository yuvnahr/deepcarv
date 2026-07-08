"""Command-line interface for FFT-75 dataset audit tools."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.dataset_tools.io import to_jsonable
from src.dataset_tools.fingerprint import write_dataset_fingerprint
from src.dataset_tools.npz_inspector import inspect_npz
from src.dataset_tools.report import generate_report
from src.dataset_tools.statistics import compute_dataset_statistics
from src.dataset_tools.validator import validate_dataset

LOGGER = logging.getLogger("dataset_tools")


def build_parser() -> argparse.ArgumentParser:
    """Build the dataset tools argument parser."""
    parser = argparse.ArgumentParser(description="FFT-75 dataset audit toolkit")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect one NPZ split file")
    inspect_parser.add_argument("path", type=Path)
    inspect_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    validate_parser = subparsers.add_parser("validate", help="Validate an FFT-75 dataset folder")
    validate_parser.add_argument("--root", type=Path, required=True)
    validate_parser.add_argument("--fragment-size", type=int, required=True)
    validate_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a validated dataset")
    analyze_parser.add_argument("--root", type=Path, required=True)
    analyze_parser.add_argument("--fragment-size", type=int, required=True)
    analyze_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    report_parser = subparsers.add_parser("report", help="Generate a full audit report")
    report_parser.add_argument("--root", type=Path, required=True)
    report_parser.add_argument("--fragment-size", type=int, required=True)
    report_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory, defaults to outputs/dataset_audit_fft75_<fragment-size>",
    )
    report_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    fingerprint_parser = subparsers.add_parser(
        "fingerprint",
        help="Generate dataset_fingerprint.json for an FFT-75 dataset",
    )
    fingerprint_parser.add_argument("--root", type=Path, required=True)
    fingerprint_parser.add_argument("--fragment-size", type=int, required=True)
    fingerprint_parser.add_argument("--output", type=Path, required=True)
    fingerprint_parser.add_argument("--fft75-version", default="unknown")
    fingerprint_parser.add_argument("--json", action="store_true")
    return parser


def _emit(data: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(to_jsonable(data), indent=2, sort_keys=True))


def _print_inspection(result: Any) -> None:
    status = "OK" if result.ok else "FAILED"
    LOGGER.info("Inspection %s: %s", status, result.path)
    LOGGER.info("Keys: %s", ", ".join(result.keys) if result.keys else "(none)")
    LOGGER.info("X shape/dtype: %s / %s", result.x_shape, result.x_dtype)
    LOGGER.info("y shape/dtype: %s / %s", result.y_shape, result.y_dtype)
    LOGGER.info("Samples: %s  Fragment length: %s", result.num_samples, result.fragment_length)
    LOGGER.info("Byte range: %s..%s", result.value_min, result.value_max)
    if result.errors:
        for error in result.errors:
            LOGGER.error(error)


def main(argv: list[str] | None = None) -> int:
    """Run the dataset tools CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    if args.command == "inspect":
        inspection_result = inspect_npz(args.path)
        if args.json:
            _emit(inspection_result, True)
        else:
            _print_inspection(inspection_result)
        return 0 if inspection_result.ok else 1

    if args.command == "validate":
        validation_result = validate_dataset(args.root, args.fragment_size)
        if args.json:
            _emit(validation_result, True)
        else:
            LOGGER.info("Validation %s", "PASSED" if validation_result.ok else "FAILED")
            for warning in validation_result.warnings:
                LOGGER.warning(warning)
            for error in validation_result.errors:
                LOGGER.error(error)
        return 0 if validation_result.ok else 1

    if args.command == "analyze":
        validation = validate_dataset(args.root, args.fragment_size)
        if not validation.ok:
            for error in validation.errors:
                LOGGER.error(error)
            return 1
        stats = compute_dataset_statistics(args.root, args.fragment_size)
        if args.json:
            _emit(stats, True)
        else:
            LOGGER.info("Total samples: %s", stats.total_sample_count)
            LOGGER.info("Imbalance severity: %s", stats.imbalance_severity)
            LOGGER.info("Leakage detected: %s", "yes" if stats.leakage_report.has_leakage else "no")
        return 0

    if args.command == "report":
        output = args.output or Path("outputs") / f"dataset_audit_fft75_{args.fragment_size}"
        audit = generate_report(args.root, args.fragment_size, output)
        if args.json:
            _emit(audit, True)
        else:
            LOGGER.info("Report written to %s", audit.report_path)
            LOGGER.info("Validation %s", "PASSED" if audit.validation.ok else "FAILED")
        return 0 if audit.validation.ok else 1

    if args.command == "fingerprint":
        fingerprint = write_dataset_fingerprint(
            args.root,
            args.fragment_size,
            args.output,
            fft75_version=args.fft75_version,
        )
        if args.json:
            _emit(fingerprint.to_dict(), True)
        else:
            LOGGER.info("Dataset fingerprint written to %s", args.output)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
