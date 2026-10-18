"""CLI: build train/val/test splits **by video** (not by frame).

Splitting by frame leaks near-identical frames from the same video across
splits and inflates metrics, so splits are always done at the video level.

Input is a directory of face-crop folders (one folder per video) or a manifest.
Produces ``data/splits/{train,val,test}.csv`` with columns:
``path,label,video_id``.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from src.utils.io import ensure_dir
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("dfdetect.make_splits")

# folder-name -> label heuristics for the common benchmarks
FAKE_TOKENS = ("fake", "deepfake", "manipulated", "synthesis", "fs", "df")
REAL_TOKENS = ("real", "original", "youtube", "actors")


def _infer_label(video_id: str) -> int:
    low = video_id.lower()
    if any(t in low for t in FAKE_TOKENS):
        return 1
    if any(t in low for t in REAL_TOKENS):
        return 0
    raise ValueError(f"Cannot infer label for '{video_id}'. Pass --label-from-dir.")


def collect_manifest(faces_root: str | Path, label_from_dir: bool = True) -> List[Dict[str, str]]:
    """Collect ``{path,label,video_id}`` rows from a faces directory."""
    root = Path(faces_root)
    rows: List[Dict[str, str]] = []
    for video_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        video_id = video_dir.name
        if label_from_dir:
            label = _infer_label(video_id)
        else:
            raise ValueError("label_from_dir=False is not supported yet")
        for img in sorted(video_dir.glob("*.jpg")):
            rows.append({"path": str(img), "label": str(label), "video_id": video_id})
    return rows


def split_by_video(
    rows: List[Dict[str, str]],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Dict[str, List[Dict[str, str]]]:
    """Split rows so all frames of a video stay in the same split."""
    set_seed(seed)
    per_video: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        per_video[r["video_id"]].append(r)

    videos = sorted(per_video)
    random.shuffle(videos)

    n = len(videos)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    splits = {
        "train": videos[:n_train],
        "val": videos[n_train : n_train + n_val],
        "test": videos[n_train + n_val :],
    }
    return {name: [r for v in vids for r in per_video[v]] for name, vids in splits.items()}


def _write_csv(rows: List[Dict[str, str]], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "label", "video_id"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create video-level train/val/test splits")
    parser.add_argument("--faces-dir", required=True, help="directory of per-video face folders")
    parser.add_argument("--out-dir", default="data/splits")
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.7, 0.15, 0.15))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = collect_manifest(args.faces_dir)
    logger.info("Collected %d crops from %d videos", len(rows), len({r['video_id'] for r in rows}))
    splits = split_by_video(rows, ratios=tuple(args.ratios), seed=args.seed)

    out_dir = Path(args.out_dir)
    for name, split_rows in splits.items():
        _write_csv(split_rows, out_dir / f"{name}.csv")
        logger.info("%s: %d videos, %d crops", name, len({r['video_id'] for r in split_rows}), len(split_rows))


if __name__ == "__main__":  # pragma: no cover
    main()
