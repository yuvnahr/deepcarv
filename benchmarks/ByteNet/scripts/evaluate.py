"""
benchmarks/ByteNet/scripts/evaluate.py
----------------------------------------
Evaluation entry point for ByteNet on FFT-75.

Loads a trained ByteNet checkpoint and runs the full evaluation pipeline
using the framework's generic Evaluator, producing the standard output set:
    metrics.json, summary.json, predictions.csv,
    confusion_matrix.csv, per_class_metrics.csv, classification_report.txt

Usage
-----
    python benchmarks/ByteNet/scripts/evaluate.py \\
        --checkpoint checkpoints/best_bytenet_bytenet_resnet_fft75_512b.pt \\
        --data_dir  data/FFT-75 \\
        --fragment_size 512 \\
        --variant   bytenet_resnet \\
        --out_dir   outputs/bytenet_eval

Constraints:
  - Loads the official pre-split test.npz — does NOT regenerate splits.
  - Does NOT hardcode dataset paths.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from src.data.dataset import FragmentDataset, build_dataloader
from src.evaluation.evaluator import Evaluator
from src.models.registry import build_model
from src.utils.paths import FFT75_DATA_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR, ensure_dirs

logger = logging.getLogger("bytenet_eval")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="ByteNet FFT-75 evaluation.")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to trained ByteNet checkpoint (.pt).")
    p.add_argument("--data_dir", type=Path, default=None,
                   help="FFT-75 root directory.")
    p.add_argument("--fragment_size", type=int, default=512, choices=[512, 4096])
    p.add_argument("--variant", type=str, default="bytenet_resnet",
                   choices=["bytenet_resnet", "bytenet_former"])
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Output directory for evaluation artefacts.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    )
    args = _parse_args(argv)

    from src.utils.seed import set_seed
    set_seed(args.seed)

    data_dir = args.data_dir or FFT75_DATA_DIR
    variant = args.variant
    fragment_size = args.fragment_size
    out_dir = args.out_dir or (
        OUTPUTS_DIR / f"bytenet_{variant}_fft75_{fragment_size}b_eval"
    )
    ensure_dirs(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=== ByteNet Evaluation: %s | fragment_size=%d | device=%s ===",
                variant, fragment_size, device)

    # ---- Load test dataset -----------------------------------------------
    test_ds = FragmentDataset(data_dir, "test", fragment_size, cache=True)
    logger.info("Test : %s", test_ds)
    test_loader = build_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)

    # ---- Build model (via registry) with checkpoint ----------------------
    ckpt = torch.load(args.checkpoint, map_location=device)
    num_classes = ckpt.get("num_classes", test_ds.num_classes)

    model = build_model(
        "bytenet",
        num_classes=num_classes,
        variant=variant,
        fragment_size=fragment_size,
    )
    # Load weights (checkpoint stores model._model state dict)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model._model.load_state_dict(state_dict)
    logger.info("Loaded checkpoint from epoch %d (val_acc=%.4f)",
                ckpt.get("epoch", 0), ckpt.get("val_acc", float("nan")))

    # ---- Evaluate -------------------------------------------------------
    evaluator = Evaluator(model, device=str(device))
    run_name = f"bytenet_{variant}_{fragment_size}b"
    result = evaluator.evaluate_and_save(test_loader, out_dir, run_name=run_name)

    # ---- Print summary --------------------------------------------------
    m = result["metrics"]
    print("\n=== ByteNet Evaluation Results ===")
    print(f"  Variant       : {variant}")
    print(f"  Fragment size : {fragment_size}")
    print(f"  Test samples  : {result['n_samples']:,}")
    print(f"  Accuracy      : {m.accuracy:.4f}")
    print(f"  Macro F1      : {m.macro_f1:.4f}")
    print(f"  Weighted F1   : {m.weighted_f1:.4f}")
    print(f"  Latency (ms)  : {result['latency_ms_per_sample']:.3f} per sample")
    print(f"  GPU memory    : {result['peak_gpu_memory_mb']:.1f} MB")
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
