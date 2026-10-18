"""Tests for the face-crop dataset."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.base import FaceDataset, class_balance, read_split_csv  # noqa: E402


def _make_faces(root: Path, n_real: int = 2, n_fake: int = 2) -> list[dict]:
    rows = []
    for label, count in ((0, n_real), (1, n_fake)):
        for i in range(count):
            d = root / f"video_{'real' if label == 0 else 'fake'}_{i}"
            d.mkdir(parents=True, exist_ok=True)
            img_path = d / "000000_0.jpg"
            cv2.imwrite(str(img_path), np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            rows.append({"path": str(img_path), "label": str(label), "video_id": d.name})
    return rows


def test_face_dataset_returns_normalised_tensor(tmp_path):
    rows = _make_faces(tmp_path)
    ds = FaceDataset(rows=rows, input_size=380)
    assert len(ds) == len(rows)
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 380, 380)
    assert img.dtype == torch.float32
    assert label in (0.0, 1.0)


def test_face_dataset_from_csv(tmp_path):
    rows = _make_faces(tmp_path)
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("path,label,video_id\n" + "\n".join(f"{r['path']},{r['label']},{r['video_id']}" for r in rows) + "\n")
    ds = FaceDataset(csv_path=csv_path, input_size=64)
    assert len(ds) == len(rows)
    assert len(read_split_csv(csv_path)) == len(rows)


def test_return_path_flag(tmp_path):
    rows = _make_faces(tmp_path)
    ds = FaceDataset(rows=rows, input_size=64, return_path=True)
    img, label, path = ds[0]
    assert img.shape == (3, 64, 64)
    assert path.endswith(".jpg")


def test_class_balance(tmp_path):
    rows = _make_faces(tmp_path, n_real=3, n_fake=1)
    counts = class_balance(rows)
    assert counts[0] == 3
    assert counts[1] == 1


def test_dataset_requires_rows_or_csv():
    with pytest.raises(ValueError):
        FaceDataset()
