"""FaceForensics++ (FF++) dataset builder.

Expects faces extracted per video under ``data/faces/ffpp`` and a
video-level split CSV produced by ``src.preprocessing.make_splits``.
Label convention: 0 = real (original/youtube), 1 = fake (manipulated methods).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from src.datasets.base import FaceDataset


def build_ffpp_dataset(
    split: str = "train",
    splits_dir: str | Path = "data/splits",
    root: str | Path = "data/faces/ffpp",
    transform: Optional[Callable] = None,
    input_size: int = 380,
) -> FaceDataset:
    """Build an FF++ face dataset for ``split`` in {train, val, test}."""
    csv_path = Path(splits_dir) / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Split not found: {csv_path}. Run src.preprocessing.make_splits first."
        )
    return FaceDataset(csv_path=csv_path, transform=transform, input_size=input_size, root=root)
