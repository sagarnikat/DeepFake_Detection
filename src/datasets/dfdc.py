"""DFDC (DeepFake Detection Challenge) dataset builder.

DFDC ships a ``metadata.json`` with per-video labels. This builder attaches the
label to every extracted face crop of a video (video-level label -> frame rows).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.datasets.base import FaceDataset
from src.utils.logging import get_logger

logger = get_logger("dfdetect.dfdc")


def load_dfdc_labels(metadata_path: str | Path) -> Dict[str, int]:
    """Return ``{video_filename: label}`` from a DFDC ``metadata.json``."""
    with open(metadata_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    labels: Dict[str, int] = {}
    for filename, info in meta.items():
        labels[filename] = int(info["label"]) if "label" in info else int(info.get("is_fake", 0))
    return labels


def build_rows(
    faces_root: str | Path,
    metadata_path: str | Path,
) -> List[Dict[str, str]]:
    """Build dataset rows from extracted DFDC faces + metadata.json."""
    labels = load_dfdc_labels(metadata_path)
    root = Path(faces_root)
    rows: List[Dict[str, str]] = []
    for video_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        video_name = video_dir.name
        stems = {video_name, f"{video_name}.mp4", video_name.split(".")[0]}
        label = next((labels[s] for s in stems if s in labels), None)
        if label is None:
            logger.warning("No label for %s, skipping", video_name)
            continue
        for img in sorted(video_dir.glob("*.jpg")):
            rows.append({"path": str(img), "label": str(label), "video_id": video_name})
    return rows


def build_dfdc_dataset(
    faces_root: str | Path = "data/faces/dfdc",
    metadata_path: str | Path = "data/raw/dfdc/metadata.json",
    transform: Optional[Callable] = None,
    input_size: int = 380,
) -> FaceDataset:
    """Build a DFDC face dataset (used mostly for cross-dataset evaluation)."""
    rows = build_rows(faces_root, metadata_path)
    if not rows:
        raise FileNotFoundError(
            f"No labeled faces under {faces_root}. Extract faces first."
        )
    return FaceDataset(rows=rows, transform=transform, input_size=input_size)
