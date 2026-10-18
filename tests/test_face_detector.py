"""Tests for the face detector / geometry helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.face_detector import (  # noqa: E402
    FaceDetector,
    isotropically_resize_image,
    put_to_center,
)


def test_isotropic_resize_keeps_aspect_ratio():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out = isotropically_resize_image(img, size=380)
    assert max(out.shape[:2]) == 380
    # original aspect ratio 2:1 preserved
    assert abs(out.shape[1] / out.shape[0] - 2.0) < 0.05


def test_isotropic_resize_noop_when_already_sized():
    img = np.zeros((380, 380, 3), dtype=np.uint8)
    assert isotropically_resize_image(img, size=380).shape == (380, 380, 3)


def test_put_to_center_pads_smaller_image():
    img = np.full((100, 50, 3), 255, dtype=np.uint8)
    out = put_to_center(img, input_size=380)
    assert out.shape == (380, 380, 3)
    # border pixels untouched (black), centre contains the image
    assert out[0, 0].tolist() == [0, 0, 0]
    assert out[190, 190].tolist() == [255, 255, 255]


def test_detector_finds_face_in_astronaut_image():
    skimage = pytest.importorskip("skimage")
    pytest.importorskip("facenet_pytorch")
    img = skimage.data.astronaut()
    detector = FaceDetector(backend="mtcnn", device="cpu")
    detections = detector.detect(img, resize_factor=1.0)
    assert len(detections) >= 1
    (x, y, w, h), score = detections[0]
    assert w > 0 and h > 0 and 0.0 <= score <= 1.0


def test_detector_rejects_unknown_backend():
    with pytest.raises((ValueError, NotImplementedError)):
        FaceDetector(backend="nope")
