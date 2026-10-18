"""Face detection & cropping.

Default detector is MTCNN (via ``facenet-pytorch``). The :class:`FaceDetector`
wrapper is intentionally small so a RetinaFace / MediaPipe backend can be added
later without touching callers.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

try:  # albumentations relocated image_compression across versions
    from albumentations.augmentations.functional import image_compression
except ImportError:  # pragma: no cover
    try:
        from albumentations.augmentations.pixel.functional import image_compression
    except ImportError:
        def image_compression(image, quality=50, image_type=None, **kwargs):  # type: ignore
            return image

from src.utils.logging import get_logger

logger = get_logger("dfdetect.face_detector")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Box = Tuple[int, int, int, int]  # x, y, w, h


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def isotropically_resize_image(
    img: np.ndarray,
    size: int,
    interpolation_down: int = cv2.INTER_AREA,
    interpolation_up: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    """Resize keeping aspect ratio so the longest side equals ``size``."""
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h, w = h * scale, size
    else:
        scale = size / h
        w, h = w * scale, size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    return cv2.resize(img, (int(w), int(h)), interpolation=interpolation)


def put_to_center(img: np.ndarray, input_size: int) -> np.ndarray:
    """Center-pad/crop ``img`` onto a black canvas of ``input_size`` x ``input_size``."""
    img = img[:input_size, :input_size]
    image = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    start_w = (input_size - img.shape[1]) // 2
    start_h = (input_size - img.shape[0]) // 2
    image[start_h : start_h + img.shape[0], start_w : start_w + img.shape[1], :] = img
    return image


# --------------------------------------------------------------------------
# detector wrapper
# --------------------------------------------------------------------------
class FaceDetector:
    """Thin wrapper over a face detector backend.

    Currently only ``"mtcnn"`` is implemented.
    """

    def __init__(
        self,
        backend: str = "mtcnn",
        device: str = DEVICE,
        thresholds: Sequence[float] = (0.7, 0.8, 0.8),
        margin: int = 0,
        min_face_size: int = 20,
    ) -> None:
        self.backend = backend
        self.device = device
        self.thresholds = list(thresholds)
        self.margin = margin
        self.min_face_size = min_face_size

        if backend == "mtcnn":
            from facenet_pytorch.models.mtcnn import MTCNN

            self._detector = MTCNN(
                margin=margin,
                thresholds=self.thresholds,
                min_face_size=min_face_size,
                device=device,
            )
        elif backend in ("retinaface", "mediapipe"):
            raise NotImplementedError(
                f"backend='{backend}' is not implemented yet; use 'mtcnn'."
            )
        else:
            raise ValueError(f"Unknown detector backend: {backend}")

    def detect(self, image_rgb: np.ndarray, resize_factor: float = 1.0) -> List[Tuple[Box, float]]:
        """Detect faces. Returns ``[(box_xywh_at_original_scale, score), ...]``."""
        pil = Image.fromarray(image_rgb.astype(np.uint8))
        if resize_factor != 1.0:
            pil = pil.resize(size=[max(1, int(s * resize_factor)) for s in pil.size])
        boxes, probs = self._detector.detect(pil, landmarks=False)
        results: List[Tuple[Box, float]] = []
        if boxes is None:
            return results
        for bbox, score in zip(boxes, probs):
            if bbox is None:
                continue
            inv = 1.0 / resize_factor
            xmin, ymin, xmax, ymax = [int(b * inv) for b in bbox]
            results.append(((xmin, ymin, xmax - xmin, ymax - ymin), float(score)))
        return results


class FaceExtractor:
    """Detect faces on sampled video frames and crop them (with margin)."""

    def __init__(
        self,
        video_read_fn: Callable[..., Optional[tuple]],
        detector: Optional[FaceDetector] = None,
        resize_factor: float = 0.5,
    ) -> None:
        self.video_read_fn = video_read_fn
        self.detector = detector or FaceDetector()
        self.resize_factor = resize_factor

    def process_videos(self, input_dir: str, filenames: Sequence[str], video_idxs: Sequence[int]) -> list[dict]:
        import os

        results: list[dict] = []
        for video_idx in video_idxs:
            filename = filenames[video_idx]
            video_path = os.path.join(input_dir, filename)
            result = self.video_read_fn(video_path)
            if result is None:
                continue
            my_frames, my_idxs = result
            for i, frame in enumerate(my_frames):
                h_frame, w_frame = frame.shape[:2]
                logger.info(
                    "[FaceExtractor] Detecting faces in video=%s frame=%s size=%dx%d",
                    filename, my_idxs[i], w_frame, h_frame,
                )
                detections = self.detector.detect(frame, resize_factor=self.resize_factor)
                faces, scores, boxes = [], [], []
                for (xmin, ymin, bw, bh), score in detections:
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(w_frame - 1, xmin + bw)
                    ymax = min(h_frame - 1, ymin + bh)
                    p_h = (ymax - ymin) // 3
                    p_w = (xmax - xmin) // 3
                    crop = frame[
                        max(ymin - p_h, 0) : ymax + p_h,
                        max(xmin - p_w, 0) : xmax + p_w,
                    ]
                    faces.append(crop)
                    scores.append(score)
                    boxes.append((xmin, ymin, xmax - xmin, ymax - ymin))
                logger.info("[FaceExtractor] Found %d faces in frame %s of %s", len(faces), my_idxs[i], filename)
                if not faces:
                    continue
                results.append(
                    {
                        "video_idx": video_idx,
                        "frame_idx": my_idxs[i],
                        "frame_w": w_frame,
                        "frame_h": h_frame,
                        "faces": faces,
                        "scores": scores,
                        "boxes": boxes,
                    }
                )
        return results

    def process_video(self, video_path: str) -> list[dict]:
        import os

        input_dir = os.path.dirname(video_path) or "."
        return self.process_videos(input_dir, [os.path.basename(video_path)], [0])
