"""File / data IO helpers used across preprocessing, training and inference."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if missing and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=_json_default)
    return out


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def encode_image_b64(image_bgr: np.ndarray, ext: str = ".jpg") -> str:
    """Encode an OpenCV BGR image to a base64 JPEG/PNG string."""
    ok, buff = cv2.imencode(ext, image_bgr)
    if not ok:
        raise ValueError("cv2.imencode failed")
    return base64.b64encode(buff).decode("utf-8")


def save_image(image_bgr: np.ndarray, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), image_bgr)
    return out


def list_videos(directory: str | Path, extensions: Sequence[str] = (".mp4", ".avi", ".mov", ".mkv")) -> list[str]:
    """Return sorted video filenames inside ``directory``."""
    d = Path(directory)
    return sorted(f.name for f in d.iterdir() if f.suffix.lower() in extensions)


def write_lines(lines: Iterable[str], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
