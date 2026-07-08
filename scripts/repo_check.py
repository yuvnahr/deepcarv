"""DeepCarv repository health check.

Run:
    python scripts/repo_check.py
"""

from __future__ import annotations

import importlib
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset_tools.fingerprint import write_dataset_fingerprint  # noqa: E402
from src.models.registry import list_models  # noqa: E402
from src.utils.config import load_experiment_config  # noqa: E402
from src.utils.config_validation import validate_yaml_tree  # noqa: E402
from src.utils.paths import CONFIGS_DIR, OUTPUTS_DIR  # noqa: E402

logger = logging.getLogger("repo_check")


@dataclass(frozen=True)
class CheckResult:
    """Result of one repository health check."""

    name: str
    ok: bool
    message: str


def _ok(name: str, message: str = "ok") -> CheckResult:
    return CheckResult(name, True, message)


def _fail(name: str, message: str) -> CheckResult:
    return CheckResult(name, False, message)


def check_repo_structure() -> CheckResult:
    """Verify expected top-level directories exist."""
    required = ["configs", "src", "tests", "docs", "outputs"]
    missing = [item for item in required if not (REPO_ROOT / item).exists()]
    return _fail("repo structure", f"missing {missing}") if missing else _ok("repo structure")


def check_configs() -> CheckResult:
    """Validate YAML syntax and composed benchmark configs."""
    tree = validate_yaml_tree(CONFIGS_DIR)
    if not tree.ok:
        return _fail("configs", "; ".join(tree.errors))
    load_experiment_config("bytercnn_fft75")
    load_experiment_config("bytercnn_fft75_4096")
    return _ok("configs")


def check_runtime_dirs() -> CheckResult:
    """Ensure runtime directories are available."""
    for name in ("datasets", "checkpoints", "outputs"):
        (REPO_ROOT / name).mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return _ok("datasets/checkpoints/outputs")


def check_imports() -> CheckResult:
    """Verify key research infrastructure imports."""
    modules = [
        "src.experiment_db",
        "src.compatibility",
        "src.profiler",
        "src.research.metadata",
        "src.research.output_versioning",
        "src.dataset_tools.fingerprint",
    ]
    for module in modules:
        importlib.import_module(module)
    return _ok("imports")


def check_registry() -> CheckResult:
    """Verify the benchmark registry loads."""
    models = list_models()
    required = {"bytercnn", "carveformer", "bytenet", "deepcarv"}
    missing = sorted(required - set(models))
    return _fail("benchmark registration", f"missing {missing}") if missing else _ok("benchmark registration")


def check_environment() -> CheckResult:
    """Verify Python, dependencies, CUDA, and GPU are discoverable."""
    cuda = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if cuda else "none"
    return _ok("environment/dependencies/CUDA/GPU", f"torch={torch.__version__} cuda={cuda} gpu={gpu}")


def check_dataset_fingerprint() -> CheckResult:
    """Generate a synthetic dataset fingerprint."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir) / "FFT-75"
        frag = root / "512"
        frag.mkdir(parents=True)
        for split in ("train", "val", "test"):
            x = np.zeros((2, 512), dtype=np.uint8)
            y = np.array([0, 1], dtype=np.int64)
            np.savez(frag / f"{split}.npz", X=x, y=y)
        write_dataset_fingerprint(root, 512, Path(temp_dir) / "dataset_fingerprint.json")
    return _ok("dataset fingerprints")


def check_compatibility_api() -> CheckResult:
    """Verify compatibility API import and report creation."""
    from src.compatibility import build_compatibility_report

    class DummyModel:
        def supports_fragment_size(self, fragment_size: int) -> bool:
            return fragment_size in {512, 4096}

    report = build_compatibility_report(
        DummyModel(),
        model_name="dummy",
        dataset_name="fft75",
        fragment_size=512,
        num_classes=75,
    )
    return _ok("benchmark compatibility") if report.ok else _fail("benchmark compatibility", "dummy failed")


def run_checks() -> list[CheckResult]:
    """Run all repository health checks."""
    checks = [
        check_repo_structure,
        check_configs,
        check_runtime_dirs,
        check_imports,
        check_registry,
        check_environment,
        check_dataset_fingerprint,
        check_compatibility_api,
    ]
    results: list[CheckResult] = []
    for check in checks:
        try:
            results.append(check())
        except Exception as exc:
            results.append(_fail(check.__name__.replace("check_", ""), str(exc)))
    return results


def main() -> int:
    """Run repo health checks and return a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = run_checks()
    for result in results:
        symbol = "✓" if result.ok else "✗"
        logger.info("%s %s: %s", symbol, result.name, result.message)
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
