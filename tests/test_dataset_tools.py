from __future__ import annotations

from pathlib import Path

import numpy as np

from src.dataset_tools.cli import build_parser, main
from src.dataset_tools.leakage import detect_leakage
from src.dataset_tools.npz_inspector import inspect_npz
from src.dataset_tools.report import generate_report
from src.dataset_tools.validator import validate_dataset


def _write_fft75(root: Path, fragment_size: int = 512) -> Path:
    frag_dir = root / "FFT-75" / str(fragment_size)
    frag_dir.mkdir(parents=True)
    base = np.arange(fragment_size, dtype=np.uint8)
    for split, offset in (("train", 0), ("val", 10), ("test", 20)):
        x = np.stack([(base + offset + index).astype(np.uint8) for index in range(6)])
        y = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
        np.savez(frag_dir / f"{split}.npz", X=x, y=y)
    return root / "FFT-75"


def test_inspect_tiny_npz(tmp_path: Path) -> None:
    path = tmp_path / "train.npz"
    x = np.arange(24, dtype=np.uint8).reshape(3, 8)
    y = np.array([0, 1, 1], dtype=np.int64)
    np.savez(path, X=x, y=y)

    result = inspect_npz(path)

    assert result.ok
    assert result.x_shape == (3, 8)
    assert result.y_shape == (3,)
    assert result.num_samples == 3
    assert result.fragment_length == 8
    assert result.class_counts == {0: 1, 1: 2}
    assert result.values_in_byte_range


def test_validate_correct_synthetic_fft75(tmp_path: Path) -> None:
    root = _write_fft75(tmp_path)

    result = validate_dataset(root, 512)

    assert result.ok
    assert not result.errors
    assert set(result.summary["splits"]) == {"train", "val", "test"}


def test_validate_failure_on_missing_x_or_y(tmp_path: Path) -> None:
    frag_dir = tmp_path / "FFT-75" / "512"
    frag_dir.mkdir(parents=True)
    x = np.zeros((2, 512), dtype=np.uint8)
    y = np.array([0, 1], dtype=np.int64)
    np.savez(frag_dir / "train.npz", X=x, y=y)
    np.savez(frag_dir / "val.npz", X=x)
    np.savez(frag_dir / "test.npz", X=x, y=y)

    result = validate_dataset(tmp_path / "FFT-75", 512)

    assert not result.ok
    assert any("val: Missing required key" in error for error in result.errors)


def test_validate_failure_on_fragment_size_mismatch(tmp_path: Path) -> None:
    root = _write_fft75(tmp_path, fragment_size=512)

    result = validate_dataset(root, 4096)

    assert not result.ok
    assert any("Fragment directory does not exist" in error for error in result.errors)


def test_validate_failure_on_extra_npz_key(tmp_path: Path) -> None:
    frag_dir = tmp_path / "FFT-75" / "512"
    frag_dir.mkdir(parents=True)
    for split in ("train", "val", "test"):
        np.savez(
            frag_dir / f"{split}.npz",
            X=np.zeros((3, 512), dtype=np.uint8),
            y=np.array([0, 1, 2], dtype=np.int64),
            metadata=np.array([1]),
        )

    result = validate_dataset(tmp_path / "FFT-75", 512)

    assert not result.ok
    assert any("extra key" in error for error in result.errors)


def test_validate_failure_when_x_shape_does_not_match_fragment_size(tmp_path: Path) -> None:
    frag_dir = tmp_path / "FFT-75" / "512"
    frag_dir.mkdir(parents=True)
    for split in ("train", "val", "test"):
        np.savez(
            frag_dir / f"{split}.npz",
            X=np.zeros((3, 16), dtype=np.uint8),
            y=np.array([0, 1, 2], dtype=np.int64),
        )

    result = validate_dataset(tmp_path / "FFT-75", 512)

    assert not result.ok
    assert any("fragment length mismatch" in error for error in result.errors)


def test_detect_duplicate_overlaps_across_splits(tmp_path: Path) -> None:
    root = _write_fft75(tmp_path)
    train = np.load(root / "512" / "train.npz", allow_pickle=False)
    val = np.load(root / "512" / "val.npz", allow_pickle=False)
    val_x = val["X"].copy()
    val_y = val["y"].copy()
    val_x[0] = train["X"][0]
    val_y[0] = train["y"][0]
    train.close()
    val.close()
    np.savez(root / "512" / "val.npz", X=val_x, y=val_y)

    report = detect_leakage(root, 512)

    assert report.has_leakage
    assert any(overlap.overlapping_hashes >= 1 for overlap in report.pair_overlaps)
    assert any(
        overlap.left_split == "train"
        and overlap.right_split == "val"
        and overlap.overlapping_hashes >= 1
        for overlap in report.overlaps
    )


def test_generate_report_in_temp_directory(tmp_path: Path) -> None:
    root = _write_fft75(tmp_path)
    output = tmp_path / "audit"

    report = generate_report(root, 512, output)

    assert report.validation.ok
    assert (output / "report.md").exists()
    assert (output / "summary.json").exists()
    assert (output / "validation.json").exists()
    assert (output / "class_distribution.png").exists()


def test_cli_entrypoints_import_and_parse_arguments(tmp_path: Path) -> None:
    root = _write_fft75(tmp_path)
    parser = build_parser()

    args = parser.parse_args(["validate", "--root", str(root), "--fragment-size", "512"])

    assert args.command == "validate"
    assert main(["validate", "--root", str(root), "--fragment-size", "512", "--json"]) == 0
