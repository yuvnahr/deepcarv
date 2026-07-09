"""
benchmarks/ByteNet/scripts/train.py
-------------------------------------
Full training script for ByteNet on FFT-75 Scenario #1.

Implements the paper's exact training protocol (§IV-B):
  - Optimizer: AdamW, betas=(0.9, 0.999), weight decay=0.01
  - LR: linear warmup 5e-7 → 5e-4 over 2 epochs,
         cosine decay to 0 over remaining 48 epochs
  - Total epochs: 50
  - Batch size: 512
  - CutMix (p=1.0), Mixup (p=0.8), Random Erase (p=0.5),
    Horizontal Flip (p=0.5), Normalisation (p=1.0)
  - Soft-label NLL loss (required by CutMix / Mixup)
  - AMP (mixed precision) on CUDA

Usage
-----
    python benchmarks/ByteNet/scripts/train.py \\
        --data_dir data/FFT-75 \\
        --fragment_size 512 \\
        --variant bytenet_resnet

    python benchmarks/ByteNet/scripts/train.py \\
        --config configs/experiments/bytenet_fft75_512.yaml

    # Sanity run (2 epochs, 2000 samples):
    python benchmarks/ByteNet/scripts/train.py \\
        --data_dir data/FFT-75 --sanity

Constraints (from handover doc):
  - Does NOT regenerate FFT-75 splits.
  - Does NOT hardcode dataset paths.
  - Loads pre-split .npz files via FragmentDataset.
  - Does NOT bypass the registry/trainer for model construction.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import FragmentDataset, build_dataloader
from src.models.registry import build_model
from src.utils.paths import FFT75_DATA_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR, ensure_dirs
from src.utils.seed import set_seed

logger = logging.getLogger("bytenet_train")


# ---------------------------------------------------------------------------
# Default hyperparameters (paper §IV-B)
# ---------------------------------------------------------------------------
_DEFAULTS: dict = {
    "seed": 42,
    "epochs": 50,
    "warmup_epochs": 2,
    "batch_size": 512,
    "lr": 5e-4,
    "lr_warmup_start": 5e-7,
    "weight_decay": 0.01,
    "betas": [0.9, 0.999],
    "grad_clip": 1.0,
    "patience": 10,
    "fragment_size": 512,
    "variant": "bytenet_resnet",
    "ngram_n": 16,
    # Augmentation probs
    "p_hflip": 0.5,
    "p_random_erase": 0.5,
    "p_cutmix": 1.0,
    "p_mixup": 0.8,
    "mixup_alpha": 0.8,
    "cutmix_alpha": 1.0,
    # Logging / output
    "log_every": 200,
    "cache": True,
}

# Random erase parameters (standard defaults)
_ERASE_SCALE = (0.02, 0.33)
_ERASE_RATIO = (0.3, 3.3)


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def random_hflip(img: torch.Tensor, p: float) -> torch.Tensor:
    """Horizontal flip for [B, C, H, W] image batch."""
    if p > 0 and random.random() < p:
        return img.flip(-1)
    return img


def random_erase(img: torch.Tensor, p: float) -> torch.Tensor:
    """Random erasing on [B, C, H, W] batch (applied per-sample)."""
    if p <= 0:
        return img
    B, C, H, W = img.shape
    out = img.clone()
    for b in range(B):
        if random.random() < p:
            for _ in range(10):
                area = H * W
                target_area = random.uniform(*_ERASE_SCALE) * area
                aspect = random.uniform(*_ERASE_RATIO)
                eh = int(round(math.sqrt(target_area * aspect)))
                ew = int(round(math.sqrt(target_area / aspect)))
                if eh < H and ew < W:
                    i = random.randint(0, H - eh)
                    j = random.randint(0, W - ew)
                    out[b, :, i:i + eh, j:j + ew] = torch.rand(C, eh, ew, device=img.device)
                    break
    return out


def cutmix_batch(
    img: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CutMix augmentation (arXiv:1905.04899) on image batch.

    Returns (mixed_img, mixed_target) where mixed_target is soft.
    """
    B, C, H, W = img.shape
    lam = float(np.random.beta(alpha, alpha))

    rand_idx = torch.randperm(B, device=img.device)
    target_a = target
    target_b = target[rand_idx]

    # Random bounding box
    cut_rat = math.sqrt(1.0 - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)

    lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)

    mixed = img.clone()
    mixed[:, :, y1:y2, x1:x2] = img[rand_idx, :, y1:y2, x1:x2]
    return mixed, (target_a, target_b, lam)


def mixup_batch(
    img: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixup augmentation (arXiv:1710.09412) on image batch."""
    lam = float(np.random.beta(alpha, alpha))
    rand_idx = torch.randperm(img.size(0), device=img.device)
    mixed = lam * img + (1 - lam) * img[rand_idx]
    return mixed, (target, target[rand_idx], lam)


def soft_nll_loss(
    log_probs: torch.Tensor,
    mixed_target,
    num_classes: int,
) -> torch.Tensor:
    """Compute NLL loss for mixed (CutMix/Mixup) targets.

    mixed_target can be:
        (target_a, target_b, lam)  — mixed labels
        torch.Tensor               — hard labels (fallback)
    """
    if isinstance(mixed_target, tuple):
        ta, tb, lam = mixed_target
        loss_a = F.nll_loss(log_probs, ta)
        loss_b = F.nll_loss(log_probs, tb)
        return lam * loss_a + (1.0 - lam) * loss_b
    return F.nll_loss(log_probs, mixed_target)


# ---------------------------------------------------------------------------
# Scheduler: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
    lr_start: float,
    lr_peak: float,
) -> torch.optim.lr_scheduler.SequentialLR:
    """Build warmup (linear) + cosine scheduler operating per-step.

    During warmup: lr linearly increases from lr_start to lr_peak.
    After warmup: cosine annealing from lr_peak to 0.
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    cosine_steps = total_steps - warmup_steps

    # Warmup: LinearLR multiplier goes from lr_start/lr_peak to 1.0
    start_factor = lr_start / max(lr_peak, 1e-9)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cosine_steps, 1),
        eta_min=0.0,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_steps],
    )


# ---------------------------------------------------------------------------
# Image augmentation pipeline (applied in training loop to Byte2Image output)
# ---------------------------------------------------------------------------

def apply_image_augmentation(
    img: torch.Tensor,
    target: torch.Tensor,
    cfg: dict,
    training: bool,
) -> tuple[torch.Tensor, object]:
    """Apply paper's augmentation pipeline to [B, C, H, W] image tensor.

    Returns (augmented_img, mixed_target) where mixed_target may be a
    tuple (ta, tb, lam) when CutMix/Mixup is applied.
    """
    if not training:
        return img, target

    # Horizontal flip (applied per-batch)
    img = random_hflip(img, cfg["p_hflip"])

    # Random erase (applied per-sample)
    img = random_erase(img, cfg["p_random_erase"])

    # CutMix / Mixup — CutMix takes priority (paper uses both with prob)
    mixed_target = target
    if cfg["p_cutmix"] > 0 and random.random() < cfg["p_cutmix"]:
        img, mixed_target = cutmix_batch(img, target, cfg["cutmix_alpha"])
    elif cfg["p_mixup"] > 0 and random.random() < cfg["p_mixup"]:
        img, mixed_target = mixup_batch(img, target, cfg["mixup_alpha"])

    return img, mixed_target


# ---------------------------------------------------------------------------
# Training/eval epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[object],
    cfg: dict,
    scaler: torch.cuda.amp.GradScaler,
    num_classes: int,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """Run one training or evaluation epoch.

    The model's forward() returns log-probs; augmentation is applied
    to the Byte2Image output inside the model's forward path for the
    image branch.  However, since Byte2Image is inside the model, we
    apply augmentation BEFORE passing to the model by temporarily
    intercepting the image tensor.

    Design choice: augmentation is applied at the byte tensor level via
    the model's forward hook, or equivalently, we call Byte2Image
    explicitly here and augment the resulting image.

    For simplicity and correct gradient flow, we augment the raw byte
    tensor itself indirectly: CutMix/Mixup are applied at the image
    level by calling model._convert_to_image() for the augmented image,
    then running the rest of the forward pass.  Since model.forward()
    calls byte_branch(x) and image_branch(converted_image), we need
    to split the forward in a way that allows image augmentation.

    Simpler approach used here: call model.forward(x) which internally
    does byte2image, then apply augmentation to the mixed result.
    For augmentation to work with grad, we pre-compute images from two
    separate inputs and mix.  Since CutMix/Mixup only need the final
    loss, we compute:
        log_probs = model(x_a) * lam + ... (no, this is wrong)

    Correct implementation: the model takes raw bytes x. We apply
    CutMix/Mixup at the image level by:
    1. Extracting images from x: img = model._convert_to_image(x)
    2. Augmenting: img_mix, mixed_target = cutmix/mixup(img, y)
    3. Running image branch: xdf = model.image_branch(img_mix)
    4. Running byte branch: xsf = model.byte_branch(x)   [no mixing]
    5. Fusing and classifying.

    This requires calling model submodules directly. We do this here
    for training; for eval the full model.forward() is used unmodified.
    """
    is_train = optimizer is not None
    model.train(is_train)
    context = torch.enable_grad() if is_train else torch.no_grad()

    total_loss = 0.0
    correct = 0
    n = 0

    with context:
        for batch_idx, (x, y) in enumerate(loader, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                if is_train:
                    # ---- Augmentation-aware forward ----
                    img = model._convert_to_image(x)      # [B, C, H, W]
                    img, mixed_target = apply_image_augmentation(
                        img, y, cfg, training=True
                    )
                    # Byte branch (no augmentation on raw bytes)
                    xsf = model.byte_branch(x)
                    # Image branch
                    xdf = model.image_branch(img)
                    # Fusion
                    import torch.nn.functional as F_inner
                    fused = torch.cat([xsf, xdf], dim=1)
                    logits = model.classifier(fused)
                    log_probs = F_inner.log_softmax(logits, dim=1)
                    loss = soft_nll_loss(log_probs, mixed_target, num_classes)
                else:
                    log_probs = model(x)
                    loss = F.nll_loss(log_probs, y)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if cfg["grad_clip"] > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if cfg["grad_clip"] > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            correct += (log_probs.argmax(dim=1) == y).sum().item()
            n += bs

            if batch_idx % cfg["log_every"] == 0 or batch_idx == len(loader):
                phase = "train" if is_train else "val"
                logger.info(
                    "  Epoch %d/%d [%s] batch %4d/%d | loss=%.4f acc=%.4f",
                    epoch, total_epochs, phase, batch_idx, len(loader),
                    total_loss / n, correct / n,
                )

    return total_loss / max(n, 1), correct / max(n, 1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_curves(
    train_losses: list, val_losses: list,
    train_accs: list, val_accs: list,
    lrs: list,
    out_dir: Path,
    variant: str,
    fragment_size: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, train_losses, label="Train")
    axes[0].plot(epochs, val_losses, label="Val")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("NLL Loss")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, train_accs, label="Train")
    axes[1].plot(epochs, val_accs, label="Val")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(epochs, lrs)
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_yscale("log"); axes[2].grid(True)

    fig.suptitle(f"ByteNet ({variant}) — FFT-75 S1 ({fragment_size}B, 75 classes)")
    plt.tight_layout()
    out_path = out_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Training curves → %s", out_path)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="ByteNet FFT-75 training.")
    p.add_argument("--config", type=Path, default=None,
                   help="Experiment YAML config (configs/experiments/*.yaml)")
    p.add_argument("--data_dir", type=Path, default=None,
                   help="FFT-75 root directory (contains 512/ and 4096/ subdirs).")
    p.add_argument("--fragment_size", type=int, default=None, choices=[512, 4096])
    p.add_argument("--variant", type=str, default=None,
                   choices=["bytenet_resnet", "bytenet_former"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--checkpoint_path", type=Path, default=None)
    p.add_argument("--run_dir", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--sanity", action="store_true",
                   help="Quick sanity run: 2 epochs, 2000 samples.")
    return p.parse_args(argv)


def _load_yaml(path: Path) -> dict:
    if path is None or not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _r(cli_val, cfg_val, default):
    """Priority: CLI → config → default."""
    return cli_val if cli_val is not None else (cfg_val if cfg_val is not None else default)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    )

    args = _parse_args(argv)

    # Load experiment config if provided
    exp_cfg: dict = {}
    train_cfg: dict = {}
    model_cfg: dict = {}
    ds_cfg: dict = {}

    if args.config:
        from src.utils.config import load_experiment_config
        try:
            ecfg = load_experiment_config(str(args.config))
            exp_cfg = ecfg.to_dict()
            train_cfg = ecfg.training
            model_cfg = ecfg.model
            ds_cfg = ecfg.dataset
        except Exception:
            # Fallback: raw YAML load
            raw = _load_yaml(args.config)
            train_cfg = _load_yaml(Path("configs/training") / f"{raw.get('training', 'bytenet')}.yaml")
            model_cfg = _load_yaml(Path("configs/models") / f"{raw.get('model', 'bytenet')}.yaml")

    # ---- Hyperparameters (CLI > config > paper defaults) -----------------
    seed         = _r(args.seed,         train_cfg.get("seed"),         _DEFAULTS["seed"])
    epochs       = _r(args.epochs,       train_cfg.get("epochs"),       _DEFAULTS["epochs"])
    batch_size   = _r(args.batch_size,   train_cfg.get("batch_size"),   _DEFAULTS["batch_size"])
    lr           = _r(args.lr,           train_cfg.get("lr"),           _DEFAULTS["lr"])
    lr_warmup    = train_cfg.get("lr_warmup_start", _DEFAULTS["lr_warmup_start"])
    warmup_eps   = train_cfg.get("warmup_epochs",   _DEFAULTS["warmup_epochs"])
    weight_decay = train_cfg.get("weight_decay",     _DEFAULTS["weight_decay"])
    betas        = tuple(train_cfg.get("betas",      _DEFAULTS["betas"]))
    grad_clip    = train_cfg.get("grad_clip",         _DEFAULTS["grad_clip"])
    patience     = train_cfg.get("patience",          _DEFAULTS["patience"])
    fragment_size = _r(args.fragment_size,
                       ds_cfg.get("fragment_size") or model_cfg.get("fragment_size"),
                       _DEFAULTS["fragment_size"])
    variant       = _r(args.variant,
                        model_cfg.get("variant"),
                        _DEFAULTS["variant"])

    # Augmentation probs
    aug_cfg = {
        "p_hflip":       train_cfg.get("p_hflip",       _DEFAULTS["p_hflip"]),
        "p_random_erase":train_cfg.get("p_random_erase", _DEFAULTS["p_random_erase"]),
        "p_cutmix":      train_cfg.get("p_cutmix",       _DEFAULTS["p_cutmix"]),
        "p_mixup":       train_cfg.get("p_mixup",        _DEFAULTS["p_mixup"]),
        "mixup_alpha":   train_cfg.get("mixup_alpha",    _DEFAULTS["mixup_alpha"]),
        "cutmix_alpha":  train_cfg.get("cutmix_alpha",   _DEFAULTS["cutmix_alpha"]),
        "grad_clip":     grad_clip,
        "log_every":     _DEFAULTS["log_every"],
    }

    if args.sanity:
        epochs = 2
        batch_size = 64

    # ---- Paths -----------------------------------------------------------
    data_dir = _r(args.data_dir,
                  Path(ds_cfg["root_dir"]) if "root_dir" in ds_cfg else None,
                  FFT75_DATA_DIR)
    run_name = f"bytenet_{variant}_fft75_{fragment_size}b"
    ckpt_path = _r(args.checkpoint_path, None,
                   CHECKPOINTS_DIR / f"best_{run_name}.pt")
    run_dir = _r(args.run_dir, None, OUTPUTS_DIR / run_name)

    ensure_dirs(CHECKPOINTS_DIR, Path(run_dir))

    # ---- Setup -----------------------------------------------------------
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=== ByteNet Training: %s | fragment_size=%d | device=%s ===",
                variant, fragment_size, device)

    # ---- Datasets --------------------------------------------------------
    tiny = 2000 if args.sanity else False
    train_ds = FragmentDataset(data_dir, "train", fragment_size, cache=True,
                               tiny_subset=tiny)
    val_ds   = FragmentDataset(data_dir, "val",   fragment_size, cache=True,
                               tiny_subset=tiny)
    logger.info("Train: %s", train_ds)
    logger.info("Val  : %s", val_ds)

    train_loader = build_dataloader(train_ds, batch_size=batch_size, shuffle=True,
                                    drop_last=True)
    val_loader   = build_dataloader(val_ds,   batch_size=batch_size, shuffle=False)

    # ---- Model (via registry) -------------------------------------------
    model_kwargs = {
        "variant": variant,
        "fragment_size": fragment_size,
        "ngram_n": model_cfg.get("ngram_n", _DEFAULTS["ngram_n"]),
    }
    # Forward optional architecture overrides from model config
    for k in ("embed_dim", "stage_layers", "channels", "byte_branch_dim", "patch_size"):
        if k in model_cfg:
            model_kwargs[k] = model_cfg[k]

    model = build_model("bytenet", num_classes=train_ds.num_classes, **model_kwargs)
    model = model.to(device)
    n_params = model.num_parameters()
    logger.info("Model: %s | classes=%d | params=%s",
                variant, train_ds.num_classes, f"{n_params:,}")

    # ---- Optimizer & Scheduler ------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
    )
    steps_per_epoch = len(train_loader)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=warmup_eps,
        total_epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        lr_start=lr_warmup,
        lr_peak=lr,
    )
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    # ---- Resume ----------------------------------------------------------
    start_epoch = 1
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device)
        model._model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        logger.info("Resumed from %s", args.resume)

    # ---- Training loop --------------------------------------------------
    best_val_acc = -1.0
    no_improve = 0
    train_losses, val_losses, train_accs, val_accs, lrs_per_epoch = [], [], [], [], []

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(
            model._model, train_loader, device, optimizer, scheduler,
            aug_cfg, scaler, train_ds.num_classes, epoch, epochs
        )
        va_loss, va_acc = run_epoch(
            model._model, val_loader, device, None, None,
            aug_cfg, scaler, train_ds.num_classes, epoch, epochs
        )
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)
        lrs_per_epoch.append(current_lr)

        logger.info(
            "Epoch %3d/%d | tr_loss=%.4f tr_acc=%.4f | "
            "va_loss=%.4f va_acc=%.4f | lr=%.2e | %.0fs",
            epoch, epochs, tr_loss, tr_acc, va_loss, va_acc, current_lr, elapsed,
        )

        # ---- Checkpoint --------------------------------------------------
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model._model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": va_acc,
                "val_loss": va_loss,
                "fragment_size": fragment_size,
                "num_classes": train_ds.num_classes,
                "variant": variant,
                "seed": seed,
            }, ckpt_path)
            logger.info("  ★ New best val_acc=%.4f → %s", va_acc, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d (best=%.4f).", epoch, best_val_acc)
                break

    # ---- Save curves -----------------------------------------------------
    save_curves(
        train_losses, val_losses, train_accs, val_accs, lrs_per_epoch,
        Path(run_dir), variant, fragment_size,
    )
    logger.info("=== Training complete. Best val_acc=%.4f ===", best_val_acc)
    logger.info("Checkpoint → %s", ckpt_path)


if __name__ == "__main__":
    main()
