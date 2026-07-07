"""Future integration hooks for research workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExtensionContext:
    """Shared context passed into optional research extensions."""

    config: dict[str, Any]
    run_dir: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ResearchExtension(ABC):
    """Base class for future optional integrations."""

    name: str = "extension"

    def on_run_start(self, context: ExtensionContext) -> None:
        """Hook called before a benchmark run starts."""

    def on_run_end(self, context: ExtensionContext, summary: dict[str, Any]) -> None:
        """Hook called after a benchmark run ends."""


class HyperparameterSweepAdapter(ResearchExtension):
    """Placeholder adapter for future sweep runners."""

    name = "hyperparameter_sweeps"

    @abstractmethod
    def suggest(self) -> dict[str, Any]:
        """Return one set of hyperparameter suggestions."""


class OptunaAdapter(HyperparameterSweepAdapter):
    """Placeholder for Optuna integration."""


class ExperimentLoggerAdapter(ResearchExtension):
    """Placeholder for W&B, MLflow, or other experiment loggers."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to an external system."""


class DistributedTrainingAdapter(ResearchExtension):
    """Placeholder for distributed training launchers."""


class MultiDatasetAdapter(ResearchExtension):
    """Placeholder for multi-dataset benchmark orchestration."""


class OpenSetEvaluationAdapter(ResearchExtension):
    """Placeholder for future open-set evaluation protocols."""


class ContinualLearningAdapter(ResearchExtension):
    """Placeholder for continual learning benchmark protocols."""


class DeepCarvModuleAdapter(ResearchExtension):
    """Placeholder for future DeepCarv-specific research modules."""
