"""Overlay a Grad-CAM heatmap on a face image with OpenCV."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def apply_colormap(heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Convert a ``[0, 1]`` heatmap to a BGR color image."""
    hm = np.clip(heatmap, 0.0, 1.0)
    hm_uint8 = (hm * 255).astype(np.uint8)
    return cv2.applyColorMap(hm_uint8, colormap)


def overlay_heatmap(
    image_bgr: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
    threshold: float = 0.0,
) -> np.ndarray:
    """Blend a heatmap over ``image_bgr``.

    Parameters
    ----------
    image_bgr:
        Face image (uint8, BGR).
    heatmap:
        2-D array in [0, 1]; resized to the image if needed.
    alpha:
        Blend weight of the heatmap.
    threshold:
        Heatmap values below this are ignored (kept as the original image).
    """
    if heatmap.ndim != 2:
        heatmap = np.squeeze(heatmap)
    h, w = image_bgr.shape[:2]
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h))
    color = apply_colormap(heatmap, colormap=colormap)
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, color, alpha, 0)
    if threshold > 0:
        mask = (heatmap >= threshold).astype(np.uint8)[..., None]
        overlay = image_bgr * (1 - mask) + overlay * mask
        overlay = overlay.astype(np.uint8)
    return overlay


def heatmap_to_boxes(heatmap: np.ndarray, threshold: float = 0.6, min_area: int = 64) -> list[Tuple[int, int, int, int]]:
    """Extract bounding boxes of the hottest regions (useful for reports)."""
    mask = (np.clip(heatmap, 0, 1) >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            boxes.append((x, y, w, h))
    return boxes
