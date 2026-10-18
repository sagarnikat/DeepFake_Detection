"""Video -> frames sampling.

Unified from the original ``kernel_utils.VideoReader`` and the ``VideoReader``
embedded in the Flask backend.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.utils.logging import get_logger

logger = get_logger("dfdetect.frame_sampler")


def sample_frame_indices(frame_count: int, num_frames: Optional[int]) -> np.ndarray:
    """Evenly spaced, strictly-increasing frame indices.

    ``num_frames=None`` -> every frame. Requesting more frames than exist is
    clamped, and duplicates produced by integer casting are removed (the reader
    below assumes strictly increasing indices).
    """
    if frame_count <= 0:
        return np.zeros((0,), dtype=int)
    if num_frames is None or num_frames >= frame_count:
        return np.arange(frame_count, dtype=int)
    num_frames = max(1, int(num_frames))
    idxs = np.linspace(0, frame_count - 1, num_frames, endpoint=True).astype(int)
    return np.unique(idxs)


def compute_frames_to_sample(
    video_path: str | Path,
    min_frames: int = 16,
    max_frames: int = 64,
    frames_per_second: float = 1.0,
) -> int:
    """How many frames to sample from a video based on its duration/FPS."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return min_frames
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if frame_count <= 0:
            return min_frames
        duration = frame_count / fps
        num = int(np.ceil(duration * frames_per_second))
        num = max(min_frames, min(max_frames, num))
        return int(max(1, min(num, frame_count)))
    except Exception:
        return min_frames


class VideoReader:
    """Read frames from a video file as RGB numpy arrays."""

    def __init__(self, verbose: bool = True, insets: Tuple[float, float] = (0, 0)) -> None:
        self.verbose = verbose
        self.insets = insets

    def read_frames(self, path: str | Path, num_frames: Optional[int], jitter: int = 0, seed: Optional[int] = None):
        """Return ``(frames, indices)`` or ``None`` if the video can't be read."""
        if num_frames is not None:
            assert num_frames > 0
        capture = cv2.VideoCapture(str(path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            capture.release()
            return None

        frame_idxs = sample_frame_indices(frame_count, num_frames)
        if jitter > 0:
            np.random.seed(seed)
            offsets = np.random.randint(-jitter, jitter, len(frame_idxs))
            frame_idxs = np.clip(frame_idxs + offsets, 0, frame_count - 1)

        result = self._read_frames_at_indices(path, capture, frame_idxs)
        capture.release()
        return result

    def read_random_frames(self, path: str | Path, num_frames: int, seed: Optional[int] = None):
        capture = cv2.VideoCapture(str(path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            capture.release()
            return None
        np.random.seed(seed)
        num_frames = min(num_frames, frame_count)
        frame_idxs = sorted(np.random.choice(np.arange(0, frame_count), num_frames, replace=False))
        result = self._read_frames_at_indices(path, capture, np.asarray(frame_idxs))
        capture.release()
        return result

    def read_frames_at_indices(self, path: str | Path, frame_idxs: Sequence[int]):
        assert len(frame_idxs) > 0
        capture = cv2.VideoCapture(str(path))
        result = self._read_frames_at_indices(path, capture, np.asarray(frame_idxs))
        capture.release()
        return result

    def read_middle_frame(self, path: str | Path):
        capture = cv2.VideoCapture(str(path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        result = self._read_frame_at_index(path, capture, frame_count // 2)
        capture.release()
        return result

    def read_frame_at_index(self, path: str | Path, frame_idx: int):
        capture = cv2.VideoCapture(str(path))
        result = self._read_frame_at_index(path, capture, frame_idx)
        capture.release()
        return result

    def _read_frames_at_indices(self, path, capture, frame_idxs: np.ndarray):
        try:
            frames, idxs_read = [], []
            for frame_idx in range(int(frame_idxs[0]), int(frame_idxs[-1]) + 1):
                if not capture.grab():
                    if self.verbose:
                        logger.warning("Error grabbing frame %d from %s", frame_idx, path)
                    break
                current = len(idxs_read)
                if current < len(frame_idxs) and frame_idx == frame_idxs[current]:
                    ret, frame = capture.retrieve()
                    if not ret or frame is None:
                        if self.verbose:
                            logger.warning("Error retrieving frame %d from %s", frame_idx, path)
                        break
                    frames.append(self._postprocess_frame(frame))
                    idxs_read.append(frame_idx)
                    if len(idxs_read) == len(frame_idxs):
                        break
            if frames:
                return np.stack(frames), idxs_read
            if self.verbose:
                logger.info("No frames read from %s", path)
            return None
        except Exception:  # pragma: no cover - defensive
            if self.verbose:
                logger.exception("Exception while reading %s", path)
            return None

    def _read_frame_at_index(self, path, capture, frame_idx: int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = capture.read()
        if not ret or frame is None:
            if self.verbose:
                logger.warning("Error retrieving frame %d from %s", frame_idx, path)
            return None
        return np.expand_dims(self._postprocess_frame(frame), axis=0), [frame_idx]

    def _postprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.insets[0] > 0:
            p = int(frame.shape[1] * self.insets[0])
            frame = frame[:, p:-p, :]
        if self.insets[1] > 0:
            q = int(frame.shape[0] * self.insets[1])
            frame = frame[q:-q, :, :]
        return frame


def extract_frames_cli() -> None:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description="Extract N evenly spaced frames from videos")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=32)
    args = parser.parse_args()

    reader = VideoReader()
    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    for video in sorted(in_dir.glob("*.mp4")):
        res = reader.read_frames(video, num_frames=args.num_frames)
        if res is None:
            continue
        frames, idxs = res
        dst = out_dir / video.stem
        dst.mkdir(parents=True, exist_ok=True)
        for frame, idx in zip(frames, idxs):
            cv2.imwrite(str(dst / f"{idx:06d}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        logger.info("Wrote %d frames for %s", len(idxs), video.name)


if __name__ == "__main__":  # pragma: no cover
    extract_frames_cli()
