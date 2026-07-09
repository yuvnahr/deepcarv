"""Dataclasses that describe benchmark experiment records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ExperimentRecord:
    """A normalized benchmark run record stored by the experiment database."""

    run_id: str
    timestamp: str
    model: str
    dataset: str
    fragment_size: int | None = None
    seed: int | None = None
    accuracy: float | None = None
    macro_f1: float | None = None
    weighted_f1: float | None = None
    inference_latency_ms: float | None = None
    gpu: str | None = None
    training_time_s: float | None = None
    parameter_count: int | None = None
    checkpoint: str | None = None
    git_commit: str | None = None
    run_dir: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        model: str,
        dataset: str,
        run_dir: Path | str | None = None,
        **kwargs: Any,
    ) -> "ExperimentRecord":
        """Create a record with a generated run id and timestamp."""
        return cls(
            run_id=str(kwargs.pop("run_id", uuid4())),
            timestamp=str(kwargs.pop("timestamp", utc_timestamp())),
            model=model,
            dataset=dataset,
            run_dir=str(run_dir) if run_dir is not None else None,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "model": self.model,
            "dataset": self.dataset,
            "fragment_size": self.fragment_size,
            "seed": self.seed,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "inference_latency_ms": self.inference_latency_ms,
            "gpu": self.gpu,
            "training_time_s": self.training_time_s,
            "parameter_count": self.parameter_count,
            "checkpoint": self.checkpoint,
            "git_commit": self.git_commit,
            "run_dir": self.run_dir,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentRecord":
        """Build a record from persisted JSON data."""
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in allowed})
