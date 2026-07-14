"""
src/training/callbacks.py
----------------------------
Lightweight callback system for the generic trainer. Kept minimal on
purpose — the trainer calls `on_epoch_end` after every validation pass
and stops training when any callback signals to.
"""

from __future__ import annotations

from typing import Any, Protocol


class Callback(Protocol):
    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> bool:
        """Return True to request the trainer stop after this epoch."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """Serializable state, so the callback survives a resumed run."""
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state saved by :meth:`state_dict`."""
        ...


class EarlyStopping:
    """Stops training when a monitored metric stops improving.

    Parameters
    ----------
    monitor : str
        Key in the metrics dict passed to on_epoch_end (e.g. "val_acc").
    mode : {"max", "min"}
        Whether higher or lower values of `monitor` are better.
    patience : int
        Number of epochs with no improvement before stopping.
    min_delta : float
        Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        monitor: str = "val_acc",
        mode: str = "max",
        patience: int = 5,
        min_delta: float = 0.0,
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta

        self.best: float | None = None
        self.epochs_no_improve = 0
        self.best_epoch = 0
        self.stopped_epoch: int | None = None

    def state_dict(self) -> dict[str, Any]:
        """Serialize the counters so early stopping survives a session restart.

        Without this, resuming a run would reset ``epochs_no_improve`` to 0 and
        the patience window would start over — training could then run far past
        the point it should have stopped.
        """
        return {
            "best": self.best,
            "epochs_no_improve": self.epochs_no_improve,
            "best_epoch": self.best_epoch,
            "stopped_epoch": self.stopped_epoch,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore counters saved by :meth:`state_dict`."""
        self.best = state.get("best")
        self.epochs_no_improve = int(state.get("epochs_no_improve", 0))
        self.best_epoch = int(state.get("best_epoch", 0))
        self.stopped_epoch = state.get("stopped_epoch")

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> bool:
        if self.monitor not in metrics:
            return False  # nothing to monitor yet; never block training

        value = metrics[self.monitor]
        if self._is_improvement(value):
            self.best = value
            self.best_epoch = epoch
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False

    def improved_this_epoch(self, epoch: int) -> bool:
        return epoch == self.best_epoch
