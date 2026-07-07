import pytest

from src.data.dataset_factory import build_datasets
from src.data.npz_dataset import NpzFragmentDataset
from src.data.validators import DatasetValidationError, validate_fragment_dir


def test_npz_dataset_loads(tiny_npz_dataset):
    ds = NpzFragmentDataset(root_dir=tiny_npz_dataset, split="train", fragment_size=512)
    assert len(ds) == 40
    x, y = ds[0]
    assert x.shape == (512,)
    assert y.ndim == 0


def test_dataset_factory_builds_all_splits(tiny_npz_dataset):
    datasets = build_datasets({"format": "npz", "root_dir": str(tiny_npz_dataset), "fragment_size": 512})
    assert set(datasets.keys()) == {"train", "val", "test"}
    assert len(datasets["train"]) == 40
    assert len(datasets["val"]) == 20


def test_validator_catches_missing_file(tmp_path):
    empty_dir = tmp_path / "empty"
    (empty_dir / "512").mkdir(parents=True)
    report = validate_fragment_dir(empty_dir / "512", fragment_size=512)
    assert not report.ok
    with pytest.raises(DatasetValidationError):
        report.raise_if_invalid()


def test_validator_catches_fragment_size_mismatch(tiny_npz_dataset):
    report = validate_fragment_dir(tiny_npz_dataset / "512", fragment_size=4096)
    assert not report.ok
    assert any("Fragment length mismatch" in e for e in report.errors)
