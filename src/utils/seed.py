"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (if installed)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:  # torch is optional for pure-utility usage
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:  # pragma: no cover - torch not installed
        pass


def worker_init_fn(worker_id: int) -> None:  # pragma: no cover - used by DataLoader
    """DataLoader worker seeding hook."""
    seed = 42 + worker_id
    random.seed(seed)
    np.random.seed(seed)
