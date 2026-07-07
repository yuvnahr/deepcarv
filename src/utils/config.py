"""
src/utils/config.py
--------------------
Composable configuration loader for the DeepCarv benchmark framework.

An *experiment* config (configs/experiments/<name>.yaml) does not repeat
hyperparameters inline. Instead it references separate config files:

    dataset: fft75_512
    model: bytercnn
    training: default
    evaluation: benchmark

Each reference is resolved against configs/<kind>/<name>.yaml and merged
into a single, flat ExperimentConfig object. No component of the framework
(trainer, evaluator, dataset factory) should ever read a YAML file itself —
they only ever receive an already-composed dict/ExperimentConfig.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import CONFIGS_DIR


class ConfigError(RuntimeError):
    """Raised when a config file is missing, malformed, or fails composition."""


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a YAML mapping: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base`, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


_COMPONENT_DIRS = {
    "dataset": "datasets",
    "model": "models",
    "training": "training",
    "evaluation": "evaluation",
}


def _resolve_component(kind: str, ref: Any, configs_dir: Path) -> dict[str, Any]:
    """Resolve a component reference into its full config dict.

    `ref` may be:
      - a string naming a file under configs/<kind_dir>/<ref>.yaml
      - an inline dict, used as-is (useful for quick overrides/tests)
    """
    if isinstance(ref, dict):
        return ref
    if not isinstance(ref, str):
        raise ConfigError(f"Component '{kind}' must be a string name or inline dict, got {type(ref)}")

    subdir = _COMPONENT_DIRS.get(kind, kind)
    path = configs_dir / subdir / f"{ref}.yaml"
    return _load_yaml(path)


@dataclass
class ExperimentConfig:
    """Fully composed, flat configuration for a single benchmark run."""

    name: str
    dataset: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset": self.dataset,
            "model": self.model,
            "training": self.training,
            "evaluation": self.evaluation,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def load_experiment_config(
    experiment_name_or_path: str | Path,
    configs_dir: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load and compose an experiment config by name or path.

    Parameters
    ----------
    experiment_name_or_path : str | Path
        Either a bare experiment name (resolved to
        configs/experiments/<name>.yaml) or a direct path to a YAML file.
    configs_dir : Path, optional
        Root configs directory. Defaults to the repo's configs/ folder.
    overrides : dict, optional
        Dotted-path overrides applied after composition, e.g.
        {"training.epochs": 5, "dataset.fragment_size": 4096}.
    """
    configs_dir = configs_dir or CONFIGS_DIR

    candidate = Path(str(experiment_name_or_path))
    if candidate.suffix in (".yaml", ".yml") and candidate.exists():
        exp_path = candidate
    else:
        exp_path = configs_dir / "experiments" / f"{experiment_name_or_path}.yaml"

    exp_raw = _load_yaml(exp_path)

    composed: dict[str, Any] = {}
    for kind in ("dataset", "model", "training", "evaluation"):
        ref = exp_raw.get(kind)
        if ref is None:
            composed[kind] = {}
            continue
        base = _resolve_component(kind, ref, configs_dir)
        # Allow inline per-experiment overrides under the same key, e.g.
        # `training: {name: default, overrides: {epochs: 5}}`
        if isinstance(exp_raw.get(kind), dict) and "overrides" in exp_raw[kind]:
            base = _deep_merge(base, exp_raw[kind]["overrides"])
        composed[kind] = base

    cfg = ExperimentConfig(
        name=exp_raw.get("name", Path(str(experiment_name_or_path)).stem),
        dataset=composed["dataset"],
        model=composed["model"],
        training=composed["training"],
        evaluation=composed["evaluation"],
        raw=exp_raw,
    )

    if overrides:
        cfg = apply_dotted_overrides(cfg, overrides)

    return cfg


def apply_dotted_overrides(cfg: ExperimentConfig, overrides: dict[str, Any]) -> ExperimentConfig:
    """Apply overrides like {"training.epochs": 10} to a composed config."""
    section_map = {
        "dataset": cfg.dataset,
        "model": cfg.model,
        "training": cfg.training,
        "evaluation": cfg.evaluation,
    }
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        if parts[0] not in section_map:
            raise ConfigError(f"Unknown config section in override: '{dotted_key}'")
        target = section_map[parts[0]]
        for part in parts[1:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return cfg
