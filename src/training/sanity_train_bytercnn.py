"""
src/training/sanity_train_bytercnn.py
--------------------------------------
Sanity-check training run for the ByteRCNN / FFT-75 benchmark.

Goals:
  - Verify that NPZ loading, model forward pass, and loss are wired correctly.
  - Run FAST: ≤ 2 epochs, ≤ 2 000 training samples (tiny_subset mode).
  - Print per-batch loss and end-of-epoch accuracy.
  - Save a temporary checkpoint to checkpoints/sanity_bytercnn_fft75.pt.
  - Exit non-zero if anything breaks.

This script is NOT about accuracy — it is about pipeline health.

Usage
-----
    python -m src.training.sanity_train_bytercnn \\
        --config configs/fft75_s1_512_bytercnn.yaml

    # Manual override:
    python -m src.training.sanity_train_bytercnn \\
        --data_dir  data/FFT-75 \\
        --fragment_size 512 \\
        --epochs 2 --subset 2000
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
    CHECKPOINTS_DIR,
    FFT75_DATA_DIR,
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
SANITY_LOG_EVERY: int = 5

logger = get_simple_logger("sanity_train")


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------


def _one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    n_samples = 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        log_probs = model(x)
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
            logger.info(
                "Epoch %d | Batch %d/%d | Loss %.4f | Running Acc %.4f",
                epoch, batch_idx + 1, len(loader),
                loss.item(), correct / n_samples,
            )

    return total_loss / n_samples, correct / n_samples


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ByteRCNN sanity training run.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--data_dir", type=Path, default=None,
                   help="Path to FFT-75/ root directory.")
    p.add_argument("--fragment_size", type=int, default=None)
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

    sanity_cfg = cfg.get("sanity", {})
    ds_cfg = cfg.get("dataset", {})
    path_cfg = cfg.get("paths", {})

    seed: int = args.seed
    subset: int = args.subset if args.subset != SANITY_SUBSET else sanity_cfg.get("subset_size", SANITY_SUBSET)
    epochs: int = args.epochs if args.epochs != SANITY_EPOCHS else sanity_cfg.get("epochs", SANITY_EPOCHS)
    batch_size: int = args.batch_size if args.batch_size != SANITY_BATCH_SIZE else sanity_cfg.get("batch_size", SANITY_BATCH_SIZE)
    lr: float = args.lr if args.lr != SANITY_LR else sanity_cfg.get("lr", SANITY_LR)

    # Paths
    data_dir: Path = (
        args.data_dir
        or Path(ds_cfg.get("root_dir", str(FFT75_DATA_DIR)))
    )
    fragment_size: int = (
        args.fragment_size
        or int(ds_cfg.get("fragment_size", 512))
    )
    ckpt_path: Path = (
        args.checkpoint_path
        or Path(path_cfg.get("sanity_checkpoint", str(CHECKPOINTS_DIR / "sanity_bytercnn_fft75.pt")))
    )

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
    logger.info("Data dir      : %s", data_dir)
    logger.info("Fragment size : %d", fragment_size)

    # ---- Dataset ---------------------------------------------------------
    logger.info("Loading tiny_subset dataset …")
    dataset = FragmentDataset(
        root_dir=data_dir,
        split="train",
        fragment_size=fragment_size,
        cache=True,
        tiny_subset=subset,
    )
    logger.info("%s", dataset)

    loader = build_dataloader(dataset, batch_size=batch_size, shuffle=True)
    logger.info("DataLoader: %d batches", len(loader))

    # ---- Model -----------------------------------------------------------
    model = build_bytercnn(num_classes=dataset.num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model loaded. Trainable parameters: %d", n_params)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- Single-batch forward check ------------------------------------
    logger.info("--- Single-batch forward check ---")
    sample_x, sample_y = next(iter(loader))
    sample_x, sample_y = sample_x.to(device), sample_y.to(device)

    with torch.no_grad():
        out = model(sample_x)

    assert out.shape == (sample_x.size(0), dataset.num_classes), (
        f"Unexpected output shape: {out.shape}"
    )
    initial_loss = criterion(out, sample_y)
    logger.info(
        "Forward OK — output shape: %s, initial loss: %.4f",
        tuple(out.shape), initial_loss.item(),
    )

    # ---- Training loop ---------------------------------------------------
    logger.info("--- Sanity training (%d epochs) ---", epochs)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        avg_loss, acc = _one_epoch(model, loader, optimizer, criterion, device, epoch)
        logger.info(
            "Epoch %d/%d | Avg Loss %.4f | Accuracy %.4f | %.1fs",
            epoch, epochs, avg_loss, acc, time.time() - t_ep,
        )

    logger.info("Sanity training complete in %.1fs.", time.time() - t0)

    # ---- Save checkpoint -------------------------------------------------
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epochs,
            "sanity": True,
            "seed": seed,
            "fragment_size": fragment_size,
            "num_classes": dataset.num_classes,
        },
        ckpt_path,
    )
    logger.info("Sanity checkpoint → %s", ckpt_path)
    logger.info("=== Sanity run PASSED ===")


if __name__ == "__main__":
    main()
