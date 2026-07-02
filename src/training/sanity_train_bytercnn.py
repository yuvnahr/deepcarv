"""
src/training/sanity_train_bytercnn.py
--------------------------------------
Sanity-check training run for the ByteRCNN / FFT-75 benchmark.

Goals:
  - Verify that data loading, model forward pass, and loss computation
    are all correctly wired together.
  - Run FAST: ≤ 2 epochs, ≤ 2 000 training samples.
  - Save a temporary checkpoint to checkpoints/sanity_bytercnn_fft75.pt.
  - Print per-batch loss and end-of-epoch accuracy.
  - Exit non-zero if anything is broken.

This script is NOT about accuracy — it is about pipeline health.

Usage
-----
    # From repo root:
    python -m src.training.sanity_train_bytercnn \\
        --config configs/fft75_s1_512_bytercnn.yaml

    # Override paths manually:
    python -m src.training.sanity_train_bytercnn \\
        --train_csv data/splits/fft75_s1_512/train.csv \\
        --class_map data/splits/fft75_s1_512/class_map.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.data.dataset import FragmentDataset, build_dataloader
from src.models.bytercnn_wrapper import build_bytercnn
from src.utils.logging import get_simple_logger
from src.utils.paths import (
    BYTERCNN_SANITY_CKPT,
    CHECKPOINTS_DIR,
    FFT75_CLASS_MAP,
    FFT75_TRAIN_CSV,
    ensure_dirs,
)
from src.utils.seed import set_seed

# ---------------------------------------------------------------------------
# Sanity run constants
# ---------------------------------------------------------------------------
SANITY_SUBSET: int = 2_000
SANITY_EPOCHS: int = 2
SANITY_BATCH_SIZE: int = 64
SANITY_LR: float = 1e-3
SANITY_LOG_EVERY: int = 5  # log every N batches

logger = get_simple_logger("sanity_train")


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    n_samples = 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        log_probs = model(x)

        # NLLLoss expects log-probabilities
        loss = criterion(log_probs, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        preds = log_probs.argmax(dim=1)
        correct += (preds == y).sum().item()
        n_samples += bs

        if (batch_idx + 1) % SANITY_LOG_EVERY == 0:
            running_acc = correct / n_samples
            logger.info(
                "Epoch %d | Batch %d/%d | Loss %.4f | Running Acc %.4f",
                epoch,
                batch_idx + 1,
                len(loader),
                loss.item(),
                running_acc,
            )

    avg_loss = total_loss / n_samples
    accuracy = correct / n_samples
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ByteRCNN sanity training run.")
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config. If omitted, uses code defaults.",
    )
    p.add_argument("--train_csv", type=Path, default=None)
    p.add_argument("--class_map", type=Path, default=None)
    p.add_argument("--checkpoint_path", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subset", type=int, default=SANITY_SUBSET)
    p.add_argument("--epochs", type=int, default=SANITY_EPOCHS)
    p.add_argument("--batch_size", type=int, default=SANITY_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=SANITY_LR)
    return p.parse_args(argv)


def _load_config(config_path: Path | None) -> dict:
    if config_path is None or not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = _load_config(args.config)

    # Merge config → args (args take priority)
    seed: int = args.seed
    subset: int = args.subset
    epochs: int = args.epochs
    batch_size: int = args.batch_size
    lr: float = args.lr

    if args.config:
        sanity_cfg = cfg.get("sanity", {})
        subset = args.subset if "--subset" in sys.argv else sanity_cfg.get("subset_size", subset)
        epochs = args.epochs if "--epochs" in sys.argv else sanity_cfg.get("epochs", epochs)
        batch_size = args.batch_size if "--batch_size" in sys.argv else sanity_cfg.get("batch_size", batch_size)
        lr = args.lr if "--lr" in sys.argv else sanity_cfg.get("lr", lr)

    train_csv: Path = args.train_csv or FFT75_TRAIN_CSV
    class_map: Path = args.class_map or FFT75_CLASS_MAP
    ckpt_path: Path = args.checkpoint_path or BYTERCNN_SANITY_CKPT

    # ---- Setup -----------------------------------------------------------
    set_seed(seed)
    ensure_dirs(CHECKPOINTS_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=== ByteRCNN Sanity Training Run ===")
    logger.info("Device        : %s", device)
    logger.info("Seed          : %d", seed)
    logger.info("Subset size   : %d", subset)
    logger.info("Epochs        : %d", epochs)
    logger.info("Batch size    : %d", batch_size)
    logger.info("Learning rate : %.5f", lr)
    logger.info("Train CSV     : %s", train_csv)

    # ---- Data ------------------------------------------------------------
    logger.info("Loading dataset …")
    dataset = FragmentDataset(
        csv_path=train_csv,
        class_map_path=class_map,
        cache=False,
        tiny_subset=subset,
    )
    logger.info("Dataset: %s", dataset)

    loader = build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    logger.info("DataLoader: %d batches", len(loader))

    # ---- Model -----------------------------------------------------------
    model = build_bytercnn().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model loaded. Trainable parameters: %d", num_params)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- Sanity check: single batch forward/backward --------------------
    logger.info("--- Single-batch sanity check ---")
    sample_x, sample_y = next(iter(loader))
    sample_x = sample_x.to(device)
    sample_y = sample_y.to(device)

    with torch.no_grad():
        out = model(sample_x)
    assert out.shape == (sample_x.size(0), dataset.num_classes), (
        f"Unexpected output shape: {out.shape}"
    )
    loss_check = criterion(out, sample_y)
    logger.info(
        "Forward OK — output shape: %s, initial loss: %.4f",
        tuple(out.shape),
        loss_check.item(),
    )

    # ---- Training loop ---------------------------------------------------
    logger.info("--- Starting sanity training ---")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        avg_loss, acc = _one_epoch(
            model, loader, optimizer, criterion, device, epoch
        )
        elapsed = time.time() - epoch_start
        logger.info(
            "Epoch %d/%d DONE | Avg Loss %.4f | Accuracy %.4f | %.1fs",
            epoch,
            epochs,
            avg_loss,
            acc,
            elapsed,
        )

    total_time = time.time() - t0
    logger.info("Sanity training complete in %.1fs.", total_time)

    # ---- Save checkpoint -------------------------------------------------
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epochs,
            "sanity": True,
            "seed": seed,
        },
        ckpt_path,
    )
    logger.info("Sanity checkpoint saved → %s", ckpt_path)
    logger.info("=== Sanity run PASSED ===")


if __name__ == "__main__":
    main()
