"""Celeb-DF (v2) dataset builder.

Celeb-DF is the standard *cross-dataset* test set when training on FF++:
label 0 = real Celeb videos, 1 = Celeb-DF synthesized.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from src.datasets.base import FaceDataset


def build_celebdf_dataset(
    split: str = "test",
    splits_dir: str | Path = "data/splits",
    root: str | Path = "data/faces/celebdf",
    transform: Optional[Callable] = None,
    input_size: int = 380,
) -> FaceDataset:
    """Build a Celeb-DF face dataset for ``split``."""
    csv_path = Path(splits_dir) / f"celebdf_{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Split not found: {csv_path}. Create it with src.preprocessing.make_splits."
        )
    return FaceDataset(csv_path=csv_path, transform=transform, input_size=input_size, root=root)
