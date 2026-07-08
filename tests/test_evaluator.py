from src.data.dataset_factory import build_dataloaders, build_datasets
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import compute_classification_metrics
from src.models.registry import build_model
import numpy as np


def test_metrics_on_dummy_outputs():
    y_true = np.array([0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 1, 1, 2, 1, 0])
    metrics = compute_classification_metrics(y_true, y_pred, labels=[0, 1, 2])
    assert 0.0 <= metrics.accuracy <= 1.0
    assert "0" in metrics.per_class
    assert metrics.confusion.shape == (3, 3)


def test_evaluator_end_to_end(tiny_npz_dataset, tmp_path):
    datasets = build_datasets({"format": "npz", "root_dir": str(tiny_npz_dataset), "fragment_size": 512})
    loaders = build_dataloaders(datasets, training_cfg={"batch_size": 8}, eval_cfg={"batch_size": 8})

    model = build_model("bytercnn", num_classes=datasets["train"].num_classes)
    evaluator = Evaluator(model, device="cpu")

    out_dir = tmp_path / "eval_out"
    result = evaluator.evaluate_and_save(loaders["test"], out_dir, run_name="smoke_test")

    assert result["n_samples"] == len(datasets["test"])
    for fname in (
        "metrics.json", "summary.json", "predictions.csv",
        "confusion_matrix.csv", "per_class_metrics.csv", "classification_report.txt",
    ):
        assert (out_dir / fname).exists(), f"missing {fname}"
