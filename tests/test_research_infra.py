from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.compatibility import build_compatibility_report
from src.dataset_tools.fingerprint import generate_dataset_fingerprint
from src.experiment_db.experiment import ExperimentRecord
from src.experiment_db.leaderboard import leaderboard_rows, write_leaderboard
from src.experiment_db.storage import ExperimentDatabase
from src.profiler import PerformanceProfile, save_performance_profile
from src.research.benchmark_report import generate_benchmark_report
from src.research.output_versioning import create_versioned_run_dir
from src.utils.config import load_experiment_config
from src.utils.config_validation import validate_experiment_dict


def _synthetic_fft75(root: Path) -> Path:
    fft = root / "FFT-75"
    frag = fft / "512"
    frag.mkdir(parents=True)
    for split in ("train", "val", "test"):
        x = np.zeros((3, 512), dtype=np.uint8)
        y = np.array([0, 1, 1], dtype=np.int64)
        np.savez(frag / f"{split}.npz", X=x, y=y)
    return fft


def test_experiment_database_and_leaderboard(tmp_path: Path) -> None:
    db = ExperimentDatabase(tmp_path / "experiments.jsonl")
    db.add(
        ExperimentRecord.create(
            model="bytercnn",
            dataset="fft75",
            fragment_size=512,
            accuracy=0.8,
            macro_f1=0.7,
            inference_latency_ms=2.0,
        )
    )
    db.add(
        ExperimentRecord.create(
            model="bytenet",
            dataset="fft75",
            fragment_size=512,
            accuracy=0.85,
            macro_f1=0.72,
            inference_latency_ms=3.0,
        )
    )

    records = db.load()
    rows = leaderboard_rows(records)
    paths = write_leaderboard(records, tmp_path)

    assert rows[0]["model"] == "bytenet"
    assert Path(paths["json"]).exists()
    assert Path(paths["csv"]).exists()
    assert Path(paths["markdown"]).exists()


def test_dataset_fingerprint(tmp_path: Path) -> None:
    root = _synthetic_fft75(tmp_path)

    fingerprint = generate_dataset_fingerprint(root, 512, fft75_version="test")

    assert fingerprint.fragment_size == 512
    assert fingerprint.splits["train"].sample_count == 3
    assert len(fingerprint.splits["train"].sha256) == 64


def test_config_validation_accepts_composed_bytercnn_config() -> None:
    cfg = load_experiment_config("bytercnn_fft75")
    result = validate_experiment_dict(cfg.to_dict())

    assert result.ok
    assert cfg.training["learning_rate"] == cfg.training["lr"]


def test_compatibility_report_defaults_to_additive_support() -> None:
    class MinimalModel:
        def supports_fragment_size(self, fragment_size: int) -> bool:
            return fragment_size == 512

    report = build_compatibility_report(
        MinimalModel(),
        "minimal",
        "fft75",
        fragment_size=512,
        num_classes=75,
        dtype=torch.uint8,
    )

    assert report.ok
    assert report.fragment_size_supported


def test_versioned_output_dirs_do_not_overwrite(tmp_path: Path) -> None:
    first = create_versioned_run_dir("ByteRCNN", "fft75", 512, tmp_path)
    second = create_versioned_run_dir("ByteRCNN", "fft75", 512, tmp_path)

    assert first != second
    assert (tmp_path / "history.json").exists()


def test_performance_and_benchmark_report_outputs(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text(
        '{"model": "bytercnn", "accuracy": 0.9, "macro_f1": 0.8, "weighted_f1": 0.85}'
    )
    (run / "benchmark_metadata.json").write_text(
        '{"gpu": "none", "cpu": "test", "python_version": "3", "torch_version": "x"}'
    )
    save_performance_profile(PerformanceProfile(parameter_count=10), run / "performance.json")

    report = generate_benchmark_report(run)

    assert report.exists()
    assert "DeepCarv Benchmark Report" in report.read_text()
