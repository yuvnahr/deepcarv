"""Tests for cross-session resume and the wall-clock time budget.

These cover the Kaggle-timeout scenario: a 12h session ends before training
finishes, and the run must continue in a *fresh process* without losing the LR
schedule, the best-score tracking, the early-stopping counters, or the history.
"""

import numpy as np
import pytest

from src.data.dataset_factory import build_dataloaders, build_datasets
from src.models.registry import build_model
from src.training.callbacks import EarlyStopping
from src.training.trainer import Trainer, TrainerConfig


@pytest.fixture
def tiny_data(tmp_path):
    root = tmp_path / "FFT-75"
    frag = root / "512"
    frag.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for split, n in (("train", 64), ("val", 32), ("test", 32)):
        np.savez(
            frag / f"{split}.npz",
            x=rng.integers(0, 256, (n, 512), dtype=np.uint8),
            y=rng.integers(0, 4, (n,), dtype=np.int64),
        )
    return root


def _make(tmp_path, data_root, **cfg_kwargs):
    datasets = build_datasets({"root_dir": str(data_root), "fragment_size": 512})
    loaders = build_dataloaders(datasets, {"batch_size": 16, "num_workers": 0})
    model = build_model("bytercnn", num_classes=datasets["train"].num_classes)
    cfg = TrainerConfig(device="cpu", amp=False, patience=99, **cfg_kwargs)
    trainer = Trainer(model, cfg, tmp_path / "run")
    return trainer, loaders


def test_resume_continues_and_does_not_restart(tiny_data, tmp_path):
    """A second Trainer must continue from the checkpoint, not retrain from 1."""
    t1, loaders = _make(tmp_path, tiny_data, epochs=2)
    h1 = t1.fit_or_resume(loaders["train"], loaders["val"])
    assert len(h1.train_loss) == 2

    # Fresh Trainer + fresh model, same run dir -> simulates a new session.
    t2, loaders2 = _make(tmp_path, tiny_data, epochs=4)
    h2 = t2.fit_or_resume(loaders2["train"], loaders2["val"])

    # History spans all 4 epochs, not just the 2 run in this process.
    assert len(h2.train_loss) == 4


def test_resume_restores_best_metric(tiny_data, tmp_path):
    """The best score must survive a resume, or a worse checkpoint_best is written."""
    t1, loaders = _make(tmp_path, tiny_data, epochs=2)
    t1.fit_or_resume(loaders["train"], loaders["val"])
    best_before = t1._best_metric
    assert best_before is not None

    t2, loaders2 = _make(tmp_path, tiny_data, epochs=2)
    t2.resume_from(t2.last_ckpt_path)
    assert t2._best_metric == best_before
    assert t2.best_epoch == t1.best_epoch


def test_resume_is_idempotent_when_complete(tiny_data, tmp_path):
    """Re-running a finished job must no-op, not train more epochs."""
    t1, loaders = _make(tmp_path, tiny_data, epochs=2)
    t1.fit_or_resume(loaders["train"], loaders["val"])

    t2, loaders2 = _make(tmp_path, tiny_data, epochs=2)
    h2 = t2.fit_or_resume(loaders2["train"], loaders2["val"])
    assert len(h2.train_loss) == 2  # restored history; no extra epochs run


def test_time_budget_stops_cleanly(tiny_data, tmp_path):
    """An impossibly small budget must stop training early, with a checkpoint."""
    trainer, loaders = _make(tmp_path, tiny_data, epochs=50, max_hours=1e-9)
    history = trainer.fit_or_resume(loaders["train"], loaders["val"])

    assert trainer.stopped_on_time_budget is True
    assert len(history.train_loss) < 50          # stopped early
    assert trainer.last_ckpt_path.exists()       # and is resumable


def test_early_stopping_state_roundtrip():
    """EarlyStopping counters must survive serialization."""
    es = EarlyStopping(monitor="val_acc", mode="max", patience=3)
    es.on_epoch_end(1, {"val_acc": 0.5})
    es.on_epoch_end(2, {"val_acc": 0.4})  # no improvement
    state = es.state_dict()

    restored = EarlyStopping(monitor="val_acc", mode="max", patience=3)
    restored.load_state_dict(state)
    assert restored.best == es.best
    assert restored.epochs_no_improve == es.epochs_no_improve
    assert restored.best_epoch == es.best_epoch
