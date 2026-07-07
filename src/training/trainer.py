"""
src/training/trainer.py
--------------------------
Generic, model-agnostic trainer. Contains ZERO model-specific logic —
it only depends on the `FragmentClassifier` interface (forward + loss)
and standard torch DataLoaders.

Any model registered in src/models/registry.py can be trained by this
trainer without modification.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.core.interfaces import EpochResult
from src.training.callbacks import EarlyStopping
from src.training.checkpointing import load_checkpoint, resume_epoch, save_checkpoint
from src.utils.device import get_device

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str | None = "reduce_on_plateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    grad_clip: float = 1.0
    patience: int = 5
    monitor: str = "val_acc"
    monitor_mode: str = "max"
    amp: bool = True
    device: str | None = None  # "cuda" | "cpu" | "auto"
    seed: int = 42

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainerConfig":
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    epoch_time_s: list[float] = field(default_factory=list)

    def append(self, tr: EpochResult, va: EpochResult, lr: float, epoch_time_s: float) -> None:
        self.train_loss.append(tr.loss)
        self.val_loss.append(va.loss)
        self.train_acc.append(tr.accuracy)
        self.val_acc.append(va.accuracy)
        self.lr.append(lr)
        self.epoch_time_s.append(epoch_time_s)


def _build_optimizer(model: nn.Module, cfg: TrainerConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9)
    raise ValueError(f"Unsupported optimizer '{cfg.optimizer}'")


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainerConfig):
    if cfg.scheduler is None:
        return None
    if cfg.scheduler == "reduce_on_plateau":
        mode = cfg.monitor_mode
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=cfg.scheduler_factor, patience=cfg.scheduler_patience
        )
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    raise ValueError(f"Unsupported scheduler '{cfg.scheduler}'")


class Trainer:
    """Generic training loop for any `FragmentClassifier`.

    Usage
    -----
        trainer = Trainer(model, TrainerConfig.from_dict(training_cfg), run_dir)
        history = trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        run_dir: Path,
        callbacks: list | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.device = get_device(config.device)
        self.model.to(self.device)

        self.optimizer = _build_optimizer(self.model, config)
        self.scheduler = _build_scheduler(self.optimizer, config)

        self.use_amp = config.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)

        self.callbacks = callbacks if callbacks is not None else [
            EarlyStopping(monitor=config.monitor, mode=config.monitor_mode, patience=config.patience)
        ]

        self.best_ckpt_path = self.run_dir / "checkpoint_best.pt"
        self.last_ckpt_path = self.run_dir / "checkpoint_last.pt"
        self.history = TrainingHistory()
        self._best_metric: float | None = None
        self.best_epoch: int = 0
        self.checkpoint_metadata_extra: dict[str, Any] = {
            "optimizer": config.optimizer,
            "scheduler": config.scheduler,
            "training_config": config.__dict__,
        }

    # ------------------------------------------------------------------
    # Core loops
    # ------------------------------------------------------------------

    def _loss_fn(self, log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "loss"):
            return self.model.loss(log_probs, targets)
        return nn.functional.nll_loss(log_probs, targets)

    def _run_epoch(self, loader: DataLoader, train: bool) -> EpochResult:
        self.model.train(train)
        total_loss, correct, n = 0.0, 0, 0
        context = torch.enable_grad() if train else torch.no_grad()

        with context:
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    log_probs = self.model(x)
                    loss = self._loss_fn(log_probs, y)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    if self.config.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                bs = x.size(0)
                total_loss += loss.item() * bs
                correct += (log_probs.argmax(dim=1) == y).sum().item()
                n += bs

        return EpochResult(loss=total_loss / max(n, 1), accuracy=correct / max(n, 1))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        start_epoch: int = 1,
        on_epoch_end: Any = None,
    ) -> TrainingHistory:
        """Run the full training loop with validation, checkpointing, and
        early stopping. Returns the accumulated TrainingHistory."""
        logger.info(
            "Starting training: device=%s, epochs=%d, amp=%s",
            self.device, self.config.epochs, self.use_amp,
        )

        for epoch in range(start_epoch, self.config.epochs + 1):
            t0 = time.time()
            train_result = self._run_epoch(train_loader, train=True)
            val_result = self._run_epoch(val_loader, train=False)
            elapsed = time.time() - t0

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    monitor_value = (
                        val_result.accuracy
                        if self.config.monitor == "val_acc"
                        else val_result.loss
                    )
                    self.scheduler.step(monitor_value)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history.append(train_result, val_result, current_lr, elapsed)

            metrics = {
                "train_loss": train_result.loss,
                "val_loss": val_result.loss,
                "train_acc": train_result.accuracy,
                "val_acc": val_result.accuracy,
            }
            logger.info(
                "Epoch %3d/%d | train_loss=%.4f train_acc=%.4f | "
                "val_loss=%.4f val_acc=%.4f | lr=%.2e | %.1fs",
                epoch, self.config.epochs, train_result.loss, train_result.accuracy,
                val_result.loss, val_result.accuracy, current_lr, elapsed,
            )

            # Best-checkpoint selection
            monitor_value = metrics[self.config.monitor]
            is_best = (
                self._best_metric is None
                or (self.config.monitor_mode == "max" and monitor_value > self._best_metric)
                or (self.config.monitor_mode == "min" and monitor_value < self._best_metric)
            )
            if is_best:
                self._best_metric = monitor_value
                self.best_epoch = epoch
                save_checkpoint(
                    self.best_ckpt_path, self.model, self.optimizer, epoch, metrics,
                    extra={
                        "num_classes": getattr(self.model, "num_classes", None),
                        "parameter_count": self.model.num_parameters()
                        if hasattr(self.model, "num_parameters")
                        else None,
                        "best_validation_score": monitor_value,
                        **self.checkpoint_metadata_extra,
                    },
                )
                logger.info("  -> new best (%s=%.4f), checkpoint saved.", self.config.monitor, monitor_value)

            save_checkpoint(
                self.last_ckpt_path, self.model, self.optimizer, epoch, metrics,
                extra={
                    "num_classes": getattr(self.model, "num_classes", None),
                    "parameter_count": self.model.num_parameters()
                    if hasattr(self.model, "num_parameters")
                    else None,
                    "best_validation_score": self._best_metric,
                    **self.checkpoint_metadata_extra,
                },
            )

            if on_epoch_end is not None:
                on_epoch_end(epoch, metrics)

            should_stop = any(cb.on_epoch_end(epoch, metrics) for cb in self.callbacks)
            if should_stop:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

        return self.history

    def resume_from(self, path: Path) -> int:
        """Load a checkpoint and return the epoch to resume from."""
        ckpt = load_checkpoint(path, self.model, self.optimizer, map_location=self.device)
        return resume_epoch(ckpt)
