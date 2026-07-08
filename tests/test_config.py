from src.utils.config import load_experiment_config


def test_experiment_config_composes_all_sections():
    cfg = load_experiment_config("bytercnn_fft75")
    assert cfg.dataset["fragment_size"] == 512
    assert cfg.model["name"] == "bytercnn"
    assert cfg.training["optimizer"] == "adamw"
    assert cfg.evaluation["batch_size"] > 0


def test_experiment_config_overrides_apply():
    cfg = load_experiment_config("bytercnn_fft75", overrides={"training.epochs": 2})
    assert cfg.training["epochs"] == 2
