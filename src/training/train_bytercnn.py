"""
src/training/train_bytercnn.py
-------------------------------
Full training script for the ByteRCNN FFT-75 Scenario #1 baseline.

Loads the official pre-split NPZ files — no CSV generation, no split logic.

Usage
-----
    python -m src.training.train_bytercnn \\
        --config configs/fft75_s1_512_bytercnn.yaml

    # Manual override:
    python -m src.training.train_bytercnn \\
        --data_dir  data/FFT-75 \\
        --fragment_size 512 \\
        --epochs 30 --batch_size 256 --lr 1e-3 --patience 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import yaml

from src.data.dataset import FragmentDataset, build_dataloader
from src.models.bytercnn_wrapper import build_bytercnn
from src.utils.logging import RunLogger
from src.utils.paths import (
    BYTERCNN_BEST_CKPT,
    BYTERCNN_RUN_DIR,
    CHECKPOINTS_DIR,
    FFT75_DATA_DIR,
    LOGS_DIR,
    ensure_dirs,
)
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "seed": 42,
    "epochs": 30,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 5,
    "grad_clip": 1.0,
    "dropout": 0.5,
    "fragment_size": 512,
    "cache": True,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_LOG_EVERY = 500   # print a progress line every N batches


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: float = 0.0,
) -> tuple[float, float]:
    """One training or evaluation epoch. Returns (avg_loss, accuracy)."""
    import logging
    import time as _time
    _logger = logging.getLogger("train_epoch")

    is_train = optimizer is not None
    model.train(is_train)
    context = torch.enable_grad() if is_train else torch.no_grad()

    total_loss = 0.0
    correct = 0
    n = 0
    n_batches = len(loader)
    t0 = _time.time()

    scaler = GradScaler(enabled=torch.cuda.is_available())
    with context:
        for batch_idx, (x, y) in enumerate(loader, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(enabled=torch.cuda.is_available()):
                log_probs = model(x)
                loss = criterion(log_probs, y)

            if is_train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

            bs = x.size(0)
            total_loss += loss.item() * bs
            correct += (log_probs.argmax(1) == y).sum().item()
            n += bs

            if batch_idx % _LOG_EVERY == 0 or batch_idx == n_batches:
                elapsed = _time.time() - t0
                phase = "train" if is_train else "val"
                _logger.info(
                    "  [%s] batch %5d/%d | loss %.4f | acc %.4f | %.0fs elapsed",
                    phase, batch_idx, n_batches,
                    total_loss / n, correct / n, elapsed,
                )

    return total_loss / n, correct / n


def _save_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, train_losses, label="Train")
    axes[0].plot(epochs, val_losses, label="Val")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("NLL Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, train_accs, label="Train")
    axes[1].plot(epochs, val_accs, label="Val")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle("ByteRCNN — FFT-75 Scenario #1 (512 bytes, 75 classes)")
    plt.tight_layout()
    out_path = out_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Training curves → {out_path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ByteRCNN full training.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--data_dir", type=Path, default=None,
                   help="Path to FFT-75/ root directory.")
    p.add_argument("--fragment_size", type=int, default=None)
    p.add_argument("--checkpoint_path", type=Path, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--resume", type=Path, default=None)
    return p.parse_args(argv)


def _load_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _r(cli, cfg_val, default):
    """Priority: CLI → config → default."""
    return cli if cli is not None else (cfg_val if cfg_val is not None else default)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = _load_config(args.config)

    train_cfg = cfg.get("training", {})
    ds_cfg    = cfg.get("dataset", {})
    path_cfg  = cfg.get("paths", {})
    model_cfg = cfg.get("model", {})

    # Hyperparameters
    seed         = _r(args.seed,         train_cfg.get("seed"),         _DEFAULTS["seed"])
    epochs       = _r(args.epochs,       train_cfg.get("epochs"),       _DEFAULTS["epochs"])
    batch_size   = _r(args.batch_size,   train_cfg.get("batch_size"),   _DEFAULTS["batch_size"])
    lr           = _r(args.lr,           train_cfg.get("lr"),           _DEFAULTS["lr"])
    weight_decay = _r(args.weight_decay, train_cfg.get("weight_decay"), _DEFAULTS["weight_decay"])
    patience     = _r(args.patience,     train_cfg.get("patience"),     _DEFAULTS["patience"])
    grad_clip    = _r(args.grad_clip,    train_cfg.get("grad_clip"),    _DEFAULTS["grad_clip"])
    dropout      = _r(args.dropout,      model_cfg.get("dropout"),      _DEFAULTS["dropout"])

    # Dataset
    data_dir      = _r(args.data_dir,      Path(ds_cfg["root_dir"]) if "root_dir" in ds_cfg else None, FFT75_DATA_DIR)
    fragment_size = _r(args.fragment_size, ds_cfg.get("fragment_size"), _DEFAULTS["fragment_size"])
    cache         = bool(ds_cfg.get("cache", _DEFAULTS["cache"]))

    # Paths
    ckpt_path   = _r(args.checkpoint_path, Path(path_cfg["best_checkpoint"]) if "best_checkpoint" in path_cfg else None, BYTERCNN_BEST_CKPT)
    run_outputs = Path(path_cfg.get("run_outputs", str(BYTERCNN_RUN_DIR)))

    # ---- Setup ----------------------------------------------------------
    set_seed(seed)
    ensure_dirs(CHECKPOINTS_DIR, LOGS_DIR, run_outputs)

    run_logger = RunLogger(run_name="bytercnn_fft75", base_dir=LOGS_DIR, config_path=args.config)
    run_logger.info("=== ByteRCNN Full Training — FFT-75 S1 fragment_size=%d ===", fragment_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_logger.info("Device: %s", device)
    if torch.cuda.is_available():
        run_logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # ---- Datasets -------------------------------------------------------
    run_logger.info("Loading train dataset …")
    train_ds = FragmentDataset(root_dir=data_dir, split="train", fragment_size=fragment_size, cache=cache)
    run_logger.info("%s", train_ds)

    run_logger.info("Loading val dataset …")
    val_ds = FragmentDataset(root_dir=data_dir, split="val", fragment_size=fragment_size, cache=cache)
    run_logger.info("%s", val_ds)

    train_loader = build_dataloader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = build_dataloader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    # ---- Model ----------------------------------------------------------
    model = build_bytercnn(num_classes=train_ds.num_classes, dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run_logger.info("ByteRCNN | classes=%d | params=%d", train_ds.num_classes, n_params)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # ---- Resume ---------------------------------------------------------
    start_epoch = 1
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        run_logger.info("Resumed from %s (epoch %d)", args.resume, start_epoch - 1)

    # ---- Training loop --------------------------------------------------
    best_val_acc = -1.0
    epochs_no_improve = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, device, optimizer, grad_clip)
        va_loss, va_acc = _run_epoch(model, val_loader,   criterion, device)
        elapsed = time.time() - t0

        scheduler.step(va_acc)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)

        run_logger.log_epoch(epoch, {
            "train_loss": tr_loss, "val_loss": va_loss,
            "train_acc":  tr_acc,  "val_acc":  va_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": elapsed,
        })

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": va_acc, "val_loss": va_loss,
                "seed": seed, "num_classes": train_ds.num_classes,
                "fragment_size": fragment_size,
            }, ckpt_path)
            run_logger.info("  ★ New best val_acc=%.4f → checkpoint saved.", va_acc)
        else:
            epochs_no_improve += 1
            run_logger.info("  No improvement (%d/%d patience).", epochs_no_improve, patience)

        if epochs_no_improve >= patience:
            run_logger.info("Early stopping at epoch %d (best val_acc=%.4f).", epoch, best_val_acc)
            break

    _save_curves(train_losses, val_losses, train_accs, val_accs, run_outputs)
    run_logger.info("=== Training complete. Best val_acc=%.4f ===", best_val_acc)
    run_logger.info("Best checkpoint → %s", ckpt_path)
    run_logger.close()


if __name__ == "__main__":
    main()
