"""Base PyTorch ``Dataset`` for face crops.

Reads a CSV with at least ``path`` and ``label`` columns (see
``src/preprocessing/make_splits.py``) and yields ``(image, label)`` pairs where
``image`` is a float tensor of shape ``(3, input_size, input_size)``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.logging import get_logger

logger = get_logger("dfdetect.dataset")


def read_split_csv(csv_path: str | Path) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class FaceDataset(Dataset):
    """Face-crop classification dataset.

    Parameters
    ----------
    rows / csv_path:
        Either a list of ``{"path", "label", ...}`` dicts or a CSV file.
    transform:
        Callable compatible with albumentations, i.e. ``transform(image=img)["image"]``.
    input_size:
        Used only when ``transform is None`` (plain resize + ImageNet normalize).
    """

    def __init__(
        self,
        rows: Optional[List[Dict[str, str]]] = None,
        csv_path: str | Path | None = None,
        transform: Optional[Callable] = None,
        input_size: int = 380,
        root: str | Path | None = None,
        return_path: bool = False,
    ) -> None:
        if rows is None and csv_path is None:
            raise ValueError("Provide either `rows` or `csv_path`")
        self.rows = rows if rows is not None else read_split_csv(csv_path)  # type: ignore[arg-type]
        self.transform = transform
        self.input_size = input_size
        self.root = Path(root) if root is not None else None
        self.return_path = return_path
        self._normalize = torch.nn.Identity()
        if transform is None:
            from torchvision.transforms import Normalize

            self._normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, rel_path: str) -> Path:
        p = Path(rel_path)
        if not p.is_absolute() and self.root is not None:
            p = self.root / p
        return p

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        path = self._resolve(row["path"])
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(image=img)["image"]
            tensor = torch.as_tensor(img)
            if tensor.ndim == 3 and tensor.shape[-1] == 3:
                tensor = tensor.permute(2, 0, 1)
            tensor = tensor.float()
            if tensor.max() > 1.5:  # albumentations ToTensorV2 not used -> scale
                tensor = tensor / 255.0
            if tensor.shape[-1] != tensor.shape[-2]:  # noqa: protect against odd transforms
                raise RuntimeError("Transform produced a non-square tensor")
        else:
            img = cv2.resize(img, (self.input_size, self.input_size))
            tensor = torch.as_tensor(img).permute(2, 0, 1).float() / 255.0
            tensor = self._normalize(tensor)

        label = float(row["label"])
        if self.return_path:
            return tensor, label, str(path)
        return tensor, label


def class_balance(rows: List[Dict[str, str]]) -> Dict[int, int]:
    """Return counts per label."""
    counts: Dict[int, int] = {}
    for r in rows:
        counts[int(r["label"])] = counts.get(int(r["label"]), 0) + 1
    return counts
