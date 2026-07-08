"""
src/utils/seed.py
-----------------
Deterministic seeding for the entire benchmark stack.

Usage
-----
    from src.utils.seed import set_seed
    set_seed(42)
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value. The benchmark uses 42; never change this for
        the frozen FFT-75 split.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU

    # Deterministic cuDNN ops where possible.
    # Note: some ops (e.g. certain GRU kernels) may not have deterministic
    # implementations; PyTorch will raise an error only if you also set
    # torch.use_deterministic_algorithms(True). We keep it at warn=True
    # so training still runs while logging any non-deterministic ops.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        # Available from PyTorch ≥ 1.11
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Older PyTorch — best effort
        torch.use_deterministic_algorithms(True)
