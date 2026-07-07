"""
src/data/verify_dataset.py
---------------------------
Verification-only script for the FFT-75 benchmark dataset.

This script verifies that the downloaded FFT-75 dataset is intact and
correctly structured.  It does NOT modify, split, or write anything.

Expected layout after downloading and unzipping:
    FFT-75/
    ├── 512/
    │   ├── train.npz
    │   ├── val.npz
    │   └── test.npz
    └── 4096/
        ├── train.npz
        ├── val.npz
        └── test.npz

Each .npz must contain exactly two arrays:
    X : byte fragments  shape [N, fragment_size]  dtype uint8 or similar
    y : integer labels  shape [N]

Usage
-----
    python -m src.data.verify_dataset \\
        --data_dir data/FFT-75 \\
        --fragment_size 512

    # Or via config:
    python -m src.data.verify_dataset \\
        --config configs/fft75_s1_512_bytercnn.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

SPLITS = ("train", "val", "test")
REQUIRED_KEYS = {"x", "y"}


# ---------------------------------------------------------------------------
# Core verification logic
# ---------------------------------------------------------------------------


def _verify_npz(npz_path: Path, fragment_size: int, split_name: str) -> dict:
    """Load and validate a single .npz file. Returns summary dict."""
    print(f"\n  [{split_name}] Checking {npz_path.name} …")

    if not npz_path.exists():
        raise FileNotFoundError(f"    ✗ Missing: {npz_path}")

    data = np.load(str(npz_path))
    actual_keys = set(data.files)

    # Key check
    if actual_keys != REQUIRED_KEYS:
        raise ValueError(
            f"    ✗ {npz_path.name} has keys {actual_keys}, expected {REQUIRED_KEYS}"
        )

    X = data["x"]
    y = data["y"]

    # Shape checks
    if X.ndim != 2:
        raise ValueError(
            f"    ✗ X must be 2-D (N, fragment_size), got shape {X.shape}"
        )
    if y.ndim != 1:
        raise ValueError(
            f"    ✗ y must be 1-D (N,), got shape {y.shape}"
        )
    if len(X) != len(y):
        raise ValueError(
            f"    ✗ len(X)={len(X)} != len(y)={len(y)}"
        )

    # Fragment length check
    if X.shape[1] != fragment_size:
        raise ValueError(
            f"    ✗ Fragment size mismatch: X.shape[1]={X.shape[1]}, "
            f"expected {fragment_size}"
        )

    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    class_counts = {int(c): int((y == c).sum()) for c in unique_classes}

    print(f"    ✓ shape X={X.shape}  dtype_X={X.dtype}")
    print(f"    ✓ shape y={y.shape}  dtype_y={y.dtype}")
    print(f"    ✓ {num_classes} unique classes  |  min_label={int(y.min())}  max_label={int(y.max())}")

    # Class distribution summary (min/max count)
    counts = list(class_counts.values())
    print(f"    ✓ samples/class — min={min(counts)}  max={max(counts)}  "
          f"mean={sum(counts)/len(counts):.1f}")

    return {
        "split": split_name,
        "n_samples": int(len(X)),
        "fragment_size": int(X.shape[1]),
        "num_classes": num_classes,
        "dtype_X": str(X.dtype),
        "dtype_y": str(y.dtype),
        "class_counts": class_counts,
    }


def verify_dataset(
    data_dir: Path,
    fragment_size: int,
) -> None:
    """Run full verification for all three splits.

    Parameters
    ----------
    data_dir : Path
        Root FFT-75 directory (contains 512/ and/or 4096/ subdirs).
    fragment_size : int
        Which fragment size to verify (512 or 4096).
    """
    print("=" * 65)
    print(f"  FFT-75 Dataset Verification")
    print(f"  data_dir      : {data_dir.resolve()}")
    print(f"  fragment_size : {fragment_size}")
    print("=" * 65)

    # ---- Top-level dir check -------------------------------------------
    if not data_dir.exists():
        raise FileNotFoundError(
            f"✗ FFT-75 root not found: {data_dir}\n"
            f"  Download and unzip the dataset first."
        )
    print(f"\n✓ FFT-75 root exists: {data_dir}")

    # ---- Fragment-size subdir ------------------------------------------
    frag_dir = data_dir / str(fragment_size)
    if not frag_dir.exists():
        available = [d.name for d in data_dir.iterdir() if d.is_dir()]
        raise FileNotFoundError(
            f"✗ Fragment directory not found: {frag_dir}\n"
            f"  Available subdirectories: {available}"
        )
    print(f"✓ Fragment directory exists: {frag_dir}")

    # ---- Verify each split ---------------------------------------------
    summaries = []
    for split in SPLITS:
        npz_path = frag_dir / f"{split}.npz"
        summary = _verify_npz(npz_path, fragment_size, split)
        summaries.append(summary)

    # ---- Cross-split consistency ---------------------------------------
    print("\n" + "-" * 65)
    print("  Cross-split consistency checks …")

    num_classes_per_split = [s["num_classes"] for s in summaries]
    if len(set(num_classes_per_split)) != 1:
        raise ValueError(
            f"  ✗ Inconsistent class counts across splits: "
            f"{dict(zip(SPLITS, num_classes_per_split))}"
        )

    dtypes_X = [s["dtype_X"] for s in summaries]
    if len(set(dtypes_X)) != 1:
        print(f"  ⚠ Warning: X dtypes differ across splits: {dict(zip(SPLITS, dtypes_X))}")
    else:
        print(f"  ✓ Consistent X dtype across splits: {dtypes_X[0]}")

    num_classes = num_classes_per_split[0]
    print(f"  ✓ Consistent class count across splits: {num_classes}")

    # ---- Summary table -------------------------------------------------
    total_samples = sum(s["n_samples"] for s in summaries)
    print("\n" + "=" * 65)
    print("  DATASET SUMMARY")
    print("=" * 65)
    print(f"  {'Split':<10} {'Samples':>10}  {'Classes':>8}")
    print(f"  {'-'*10:<10} {'-'*10:>10}  {'-'*8:>8}")
    for s in summaries:
        print(f"  {s['split']:<10} {s['n_samples']:>10,}  {s['num_classes']:>8}")
    print(f"  {'TOTAL':<10} {total_samples:>10,}")
    print(f"\n  Fragment size : {fragment_size} bytes")
    print(f"  Num classes   : {num_classes}")
    print(f"  X dtype       : {summaries[0]['dtype_X']}")
    print(f"  y dtype       : {summaries[0]['dtype_y']}")
    print("=" * 65)
    print("\n✓ All verification checks PASSED. Dataset is ready.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify the FFT-75 NPZ benchmark dataset."
    )
    p.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="Path to the FFT-75/ root directory.",
    )
    p.add_argument(
        "--fragment_size",
        type=int,
        default=None,
        help="Fragment size to verify (512 or 4096).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (reads dataset.root_dir and dataset.fragment_size).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    data_dir: Path | None = args.data_dir
    fragment_size: int | None = args.fragment_size

    # Load from config if provided
    if args.config and args.config.exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        ds_cfg = cfg.get("dataset", {})
        if data_dir is None and "root_dir" in ds_cfg:
            data_dir = Path(ds_cfg["root_dir"])
        if fragment_size is None and "fragment_size" in ds_cfg:
            fragment_size = int(ds_cfg["fragment_size"])

    # Final fallbacks
    if data_dir is None:
        data_dir = Path("data/FFT-75")
    if fragment_size is None:
        fragment_size = 512

    try:
        verify_dataset(data_dir=data_dir, fragment_size=fragment_size)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n✗ Verification FAILED:\n  {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
