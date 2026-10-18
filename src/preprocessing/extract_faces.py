"""CLI: extract cropped faces from videos into a faces directory.

Example
-------
python -m src.preprocessing.extract_faces \
    --input-dir data/raw/ffpp --output-dir data/faces/ffpp --num-frames 32
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2

from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.frame_sampler import VideoReader, compute_frames_to_sample
from src.utils.io import ensure_dir, list_videos
from src.utils.logging import get_logger

logger = get_logger("dfdetect.extract_faces")


def extract_from_video(
    video_path: str | Path,
    output_root: str | Path,
    detector: FaceDetector,
    num_frames: int | None = None,
    min_faces: int = 1,
) -> int:
    """Detect + crop faces from one video. Returns number of saved crops."""
    video_path = Path(video_path)
    reader = VideoReader(verbose=False)
    if num_frames is None:
        num_frames = compute_frames_to_sample(video_path)
    res = reader.read_frames(video_path, num_frames=num_frames)
    if res is None:
        logger.warning("Could not read %s", video_path)
        return 0

    frames, idxs = res
    out_dir = ensure_dir(Path(output_root) / video_path.stem)
    saved = 0
    for frame, idx in zip(frames, idxs):
        detections = detector.detect(frame, resize_factor=0.5)
        for j, ((x, y, w, h), _score) in enumerate(detections):
            x, y = max(0, x), max(0, y)
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            out = out_dir / f"{idx:06d}_{j}.jpg"
            cv2.imwrite(str(out), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            saved += 1
    if saved < min_faces:
        logger.warning("Only %d face crops found in %s", saved, video_path.name)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract cropped faces from videos")
    parser.add_argument("--input-dir", required=True, help="directory with videos")
    parser.add_argument("--output-dir", required=True, help="directory for face crops")
    parser.add_argument("--num-frames", type=int, default=None, help="frames per video (default: FPS-aware)")
    parser.add_argument("--detector", default="mtcnn", choices=["mtcnn"])
    args = parser.parse_args()

    detector = FaceDetector(backend=args.detector)
    videos = list_videos(args.input_dir)
    logger.info("Found %d videos in %s", len(videos), args.input_dir)

    total = 0
    for name in videos:
        total += extract_from_video(
            os.path.join(args.input_dir, name),
            args.output_dir,
            detector,
            num_frames=args.num_frames,
        )
    logger.info("Done. Saved %d face crops to %s", total, args.output_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
