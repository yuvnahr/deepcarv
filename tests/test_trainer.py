from pathlib import Path

from src.data.dataset_factory import build_dataloaders, build_datasets
from src.models.registry import build_model
from src.training.trainer import Trainer, TrainerConfig


def test_trainer_runs_one_epoch_on_bytercnn(tiny_npz_dataset, tmp_path):
    datasets = build_datasets({"format": "npz", "root_dir": str(tiny_npz_dataset), "fragment_size": 512})
    loaders = build_dataloaders(datasets, training_cfg={"batch_size": 8}, eval_cfg={"batch_size": 8})

    model = build_model("bytercnn", num_classes=datasets["train"].num_classes)
    config = TrainerConfig(epochs=1, patience=1, amp=False, device="cpu")
    trainer = Trainer(model, config, run_dir=tmp_path / "run")

    history = trainer.fit(loaders["train"], loaders["val"])

    assert len(history.train_loss) == 1
    assert (tmp_path / "run" / "checkpoint_best.pt").exists()
    assert (tmp_path / "run" / "checkpoint_last.pt").exists()
