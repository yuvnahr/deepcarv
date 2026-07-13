"""
Root conftest.py — ensures repo root is in sys.path so all tests
(including those under benchmarks/) can import both src.* and benchmarks.*.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the repo root to sys.path if it isn't already there.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
