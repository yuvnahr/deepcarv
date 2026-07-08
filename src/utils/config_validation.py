"""Structured validation for DeepCarv YAML benchmark configs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigValidationError(RuntimeError):
    """Raised when benchmark configuration validation fails."""


@dataclass(frozen=True)
class ConfigValidationResult:
    """Result of validating one composed config or YAML file."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        """Raise with actionable messages when validation failed."""
        if not self.ok:
            details = "\n".join(f"- {error}" for error in self.errors)
            raise ConfigValidationError(f"Config validation failed:\n{details}")


def validate_experiment_dict(config: dict[str, Any]) -> ConfigValidationResult:
    """Validate a composed experiment dictionary."""
    errors: list[str] = []
    warnings: list[str] = []
    dataset = config.get("dataset", {})
    model = config.get("model", {})
    training = config.get("training", {})

    if "fragment_size" not in dataset:
        errors.append("dataset.fragment_size is required.")
    if "learning_rate" not in training and "lr" not in training:
        errors.append("training.learning_rate is required; training.lr is accepted as an alias.")
    if "learning_rate" not in training and "lr" in training:
        warnings.append("training.lr is accepted for compatibility; prefer learning_rate in new configs.")
    if "name" not in model and "model_name" not in model:
        errors.append("model.name is required and must match a registry key.")
    if "seed" not in training:
        warnings.append("training.seed is not set; reproducibility will rely on defaults.")
    return ConfigValidationResult(ok=not errors, errors=errors, warnings=warnings)


def validate_yaml_file(path: Path | str) -> ConfigValidationResult:
    """Validate an individual YAML file for basic structure."""
    yaml_path = Path(path)
    errors: list[str] = []
    if not yaml_path.exists():
        return ConfigValidationResult(False, [f"YAML file not found: {yaml_path}"])
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        errors.append(f"{yaml_path}: top-level YAML value must be a mapping.")
    return ConfigValidationResult(ok=not errors, errors=errors)


def validate_yaml_tree(configs_dir: Path | str) -> ConfigValidationResult:
    """Validate every YAML file under a configs directory."""
    root = Path(configs_dir)
    errors: list[str] = []
    warnings: list[str] = []
    for path in sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml")):
        result = validate_yaml_file(path)
        errors.extend(result.errors)
        warnings.extend(result.warnings)
    return ConfigValidationResult(ok=not errors, errors=errors, warnings=warnings)
