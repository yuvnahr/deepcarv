"""
src/training/train_bytercnn.py
-------------------------------
Full training script for the ByteRCNN FFT-75 Scenario #1 baseline.

Key design decisions:
  - Loads the frozen train.csv and val.csv (never splits internally).
  - Uses NLLLoss (model outputs log-softmax).
  - Adam / AdamW optimiser with configurable LR.
  - Early stopping on validation accuracy (patience = 5 epochs default).
  - Saves the best checkpoint keyed on val accuracy.
  - Logs per-epoch metrics to a RunLogger (timestamped run dir + CSV).
  - Saves training curves as outputs/bytercnn_fft75/training_curves.png.

Usage
-----
    # From repo root (reads all settings from YAML):
    python -m src.training.train_bytercnn \\
        --config configs/fft75_s1_512_bytercnn.yaml

    # Manual override:
    python -m src.training.train_bytercnn \\
        --train_csv data/splits/fft75_s1_512/train.csv \\
        --val_csv   data/splits/fft75_s1_512/val.csv \\
        --class_map data/splits/fft75_s1_512/class_map.json \\
        --epochs 30 --batch_size 256 --lr 1e-3 --patience 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml

from src.data.dataset import FragmentDataset, build_dataloader
from src.models.bytercnn_wrapper import build_bytercnn
from src.utils.logging import RunLogger
from src.utils.paths import (
    BYTERCNN_BEST_CKPT,
    BYTERCNN_RUN_DIR,
    CHECKPOINTS_DIR,
    FFT75_CLASS_MAP,
    FFT75_TRAIN_CSV,
    FFT75_VAL_CSV,
    LOGS_DIR,
    OUTPUTS_DIR,
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
    "num_workers": None,      # auto-detect
    "cache_dataset": False,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _accuracy(log_probs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = log_probs.argmax(dim=1)
    return (preds == targets).float().mean().item()


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: float = 0.0,
) -> tuple[float, float]:
    """Run one training or evaluation epoch.

    If *optimizer* is None, runs in eval mode with no_grad.
    Returns (avg_loss, accuracy).
    """
    is_train = optimizer is not None
    model.train(is_train)
    context = torch.enable_grad() if is_train else torch.no_grad()

    total_loss = 0.0
    correct = 0
    n = 0

    with context:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            log_probs = model(x)
            loss = criterion(log_probs, y)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            correct += (log_probs.argmax(1) == y).sum().item()
            n += bs

    return total_loss / n, correct / n


def _save_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    out_dir: Path,
) -> None:
    """Save training/validation curves as a PNG."""
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
    p = argparse.ArgumentParser(description="ByteRCNN full training run.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--train_csv", type=Path, default=None)
    p.add_argument("--val_csv", type=Path, default=None)
    p.add_argument("--class_map", type=Path, default=None)
    p.add_argument("--checkpoint_path", type=Path, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--cache", action="store_true", default=False)
    p.add_argument("--resume", type=Path, default=None,
                   help="Path to a checkpoint to resume training from.")
    return p.parse_args(argv)


def _load_config(config_path: Path | None) -> dict:
    if config_path is None or not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _resolve(args_val, cfg_val, default):
    """Priority: CLI arg > config value > default."""
    if args_val is not None:
        return args_val
    if cfg_val is not None:
        return cfg_val
    return default


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = _load_config(args.config)

    train_cfg = cfg.get("training", {})
    path_cfg = cfg.get("paths", {})
    model_cfg = cfg.get("model", {})

    # ---- Resolve hyperparameters ----------------------------------------
    seed: int = _resolve(args.seed, train_cfg.get("seed"), _DEFAULTS["seed"])
    epochs: int = _resolve(args.epochs, train_cfg.get("epochs"), _DEFAULTS["epochs"])
    batch_size: int = _resolve(args.batch_size, train_cfg.get("batch_size"), _DEFAULTS["batch_size"])
    lr: float = _resolve(args.lr, train_cfg.get("lr"), _DEFAULTS["lr"])
    weight_decay: float = _resolve(args.weight_decay, train_cfg.get("weight_decay"), _DEFAULTS["weight_decay"])
    patience: int = _resolve(args.patience, train_cfg.get("patience"), _DEFAULTS["patience"])
    grad_clip: float = _resolve(args.grad_clip, train_cfg.get("grad_clip"), _DEFAULTS["grad_clip"])
    dropout: float = _resolve(args.dropout, model_cfg.get("dropout"), _DEFAULTS["dropout"])
    cache: bool = args.cache or train_cfg.get("cache_dataset", _DEFAULTS["cache_dataset"])
    num_workers: Optional[int] = train_cfg.get("num_workers", _DEFAULTS["num_workers"])

    # ---- Resolve paths --------------------------------------------------
    train_csv: Path = args.train_csv or Path(path_cfg.get("train_csv", str(FFT75_TRAIN_CSV)))
    val_csv: Path = args.val_csv or Path(path_cfg.get("val_csv", str(FFT75_VAL_CSV)))
    class_map: Path = args.class_map or Path(path_cfg.get("class_map", str(FFT75_CLASS_MAP)))
    ckpt_path: Path = args.checkpoint_path or Path(path_cfg.get("best_checkpoint", str(BYTERCNN_BEST_CKPT)))
    run_outputs: Path = Path(path_cfg.get("run_outputs", str(BYTERCNN_RUN_DIR)))

    # ---- Setup ----------------------------------------------------------
    set_seed(seed)
    ensure_dirs(CHECKPOINTS_DIR, LOGS_DIR, run_outputs)

    run_logger = RunLogger(
        run_name="bytercnn_fft75",
        base_dir=LOGS_DIR,
        config_path=args.config,
    )
    run_logger.info("=== ByteRCNN Full Training — FFT-75 S1 512B ===")
    run_logger.info("Seed=%d  Epochs=%d  BS=%d  LR=%.5f  Patience=%d",
                    seed, epochs, batch_size, lr, patience)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_logger.info("Device: %s", device)
    if torch.cuda.is_available():
        run_logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # ---- Datasets -------------------------------------------------------
    run_logger.info("Loading train dataset …")
    train_ds = FragmentDataset(
        csv_path=train_csv,
        class_map_path=class_map,
        cache=cache,
    )
    run_logger.info("Train: %s", train_ds)

    run_logger.info("Loading val dataset …")
    val_ds = FragmentDataset(
        csv_path=val_csv,
        class_map_path=class_map,
        cache=cache,
    )
    run_logger.info("Val  : %s", val_ds)

    train_loader = build_dataloader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    val_loader = build_dataloader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )

    # ---- Model ----------------------------------------------------------
    model = build_bytercnn(
        num_classes=train_ds.num_classes,
        dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run_logger.info("Model: ByteRCNN | Params: %d", n_params)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=True
    )

    # ---- Resume from checkpoint (optional) ------------------------------
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

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    run_logger.info("Starting training …")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = _run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, grad_clip=grad_clip,
        )
        va_loss, va_acc = _run_epoch(
            model, val_loader, criterion, device,
        )

        elapsed = time.time() - t0
        scheduler.step(va_acc)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)

        run_logger.log_epoch(
            epoch,
            {
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "train_acc": tr_acc,
                "val_acc": va_acc,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time_s": elapsed,
            },
        )

        # Best checkpoint
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": va_acc,
                    "val_loss": va_loss,
                    "seed": seed,
                    "num_classes": train_ds.num_classes,
                },
                ckpt_path,
            )
            run_logger.info("  ★ New best val_acc=%.4f → checkpoint saved.", va_acc)
        else:
            epochs_no_improve += 1
            run_logger.info(
                "  No improvement (%d/%d epochs patience).",
                epochs_no_improve, patience,
            )

        # Early stopping
        if epochs_no_improve >= patience:
            run_logger.info(
                "Early stopping triggered at epoch %d (best val_acc=%.4f).",
                epoch, best_val_acc,
            )
            break

    # ---- Save curves ----------------------------------------------------
    _save_curves(train_losses, val_losses, train_accs, val_accs, run_outputs)

    run_logger.info("=== Training complete. Best val_acc=%.4f ===", best_val_acc)
    run_logger.info("Best checkpoint → %s", ckpt_path)
    run_logger.close()


if __name__ == "__main__":
    main()
