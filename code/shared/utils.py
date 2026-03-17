"""
shared/utils.py
===============
Shared utilities used by both the classification and regression pipelines.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any, Callable, TypeVar

import numpy as np
import torch

T = TypeVar("T")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(*dirs: str) -> None:
    """Create each directory (and parents) if it does not already exist."""
    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)


def timed(fn: Callable[[], T]) -> tuple[T, float]:
    """Call *fn* and return ``(result, elapsed_seconds)``."""
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0
