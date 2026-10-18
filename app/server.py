"""Flask backend for the DeepFake Detection System.

Thin HTTP layer on top of :class:`src.inference.predict_video.VideoPredictor`.
Serves the React frontend (``app/frontend``) via /predict.

Run:
    python -m app.server          # http://127.0.0.1:5000
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS

from src.inference.predict_video import (
    WEIGHTS_DIR,
    VideoPredictor,
    resolve_weight_names,
)
from src.utils.logging import get_logger

logger = get_logger("dfdetect.server")

# ---------------------------------------------------------------------------
# app + JSON provider that understands numpy scalars/arrays
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)


class NumpyJSONProvider(DefaultJSONProvider):
    @staticmethod
    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return DefaultJSONProvider.default(o)


app.json = NumpyJSONProvider(app)

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
UPLOADS_BASE_DIR = os.environ.get("DFDETECT_UPLOADS", "uploads")
MAX_UPLOAD_AGE_SECONDS = int(os.environ.get("DFDETECT_MAX_UPLOAD_AGE", 60 * 60))
CLEANUP_INTERVAL_SECONDS = int(os.environ.get("DFDETECT_CLEANUP_INTERVAL", 10 * 60))
KEEP_FRAMES_AFTER_PRED = os.environ.get("DFDETECT_KEEP_FRAMES", "1") == "1"

_predictor: VideoPredictor | None = None


def get_predictor() -> VideoPredictor:
    """Lazily build the (heavy) predictor singleton."""
    global _predictor
    if _predictor is None:
        logger.info("Loading model(s) from %s ...", WEIGHTS_DIR)
        _predictor = VideoPredictor.from_weights(WEIGHTS_DIR, resolve_weight_names())
        logger.info("Predictor ready on device=%s", _predictor.device)
    return _predictor


# ---------------------------------------------------------------------------
# upload cleanup
# ---------------------------------------------------------------------------
def cleanup_uploads(base_dir: str = UPLOADS_BASE_DIR, max_age_seconds: int = MAX_UPLOAD_AGE_SECONDS, dry_run: bool = False) -> int:
    removed = 0
    base = Path(base_dir)
    if not base.is_dir():
        return 0
    now = time.time()
    for entry in base.iterdir():
        if not entry.is_dir() or entry.name == "test":
            continue
        try:
            if now - entry.stat().st_mtime < max_age_seconds:
                continue
            logger.info("[Cleanup] Removing stale upload dir %s", entry)
            if not dry_run:
                shutil.rmtree(entry, ignore_errors=True)
            removed += 1
        except Exception:
            logger.exception("[Cleanup] Could not remove %s", entry)
    return removed


def _cleanup_worker(interval: int = CLEANUP_INTERVAL_SECONDS, max_age: int = MAX_UPLOAD_AGE_SECONDS) -> None:
    while True:
        try:
            cleanup_uploads(max_age_seconds=max_age)
        except Exception:
            logger.exception("[CleanupWorker] error")
        time.sleep(interval)


def start_cleanup_thread() -> threading.Thread:
    t = threading.Thread(target=_cleanup_worker, daemon=True)
    t.start()
    logger.info("[Cleanup] Background cleanup thread started")
    return t


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "no file provided"}), 400

    file = request.files["file"]
    filename = file.filename or f"video_{uuid.uuid4().hex[:6]}.mp4"
    uid = uuid.uuid4().hex[:8]
    save_dir = Path(UPLOADS_BASE_DIR) / uid
    video_dir = save_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    final_video_path = video_dir / filename
    file.save(str(final_video_path))
    logger.info("[Route] Received upload -> %s", final_video_path)

    try:
        predictor = get_predictor()
        result = predictor.predict(str(final_video_path), save_dir=save_dir)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Prediction failed")
        return jsonify({"error": "prediction_failed", "message": str(exc)}), 500

    if result["label"] == "NO_FACES":
        return jsonify(
            {
                "error": "no_faces",
                "message": "No people detected in the uploaded video. Please upload a video containing a clear face.",
                "forgery_reasons": result["forgery_reasons"],
                "paths": {"video_path": str(final_video_path), **result.get("saved", {})},
            }
        ), 200

    return jsonify(
        {
            "label": result["label"],
            "confidence": result["confidence"],
            "frames": result["frames"],
            "frame_scores": result["frame_scores"],
            "forgery_reasons": result["forgery_reasons"],
            "paths": {"video_path": str(final_video_path), **result.get("saved", {})},
        }
    )


@app.route("/upload-test", methods=["POST"])
def upload_test():
    try:
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "no file provided"}), 400
        f = request.files["file"]
        dst_dir = Path(UPLOADS_BASE_DIR) / "test"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / (f.filename or f"upload_{uuid.uuid4().hex[:6]}.bin")
        f.save(str(dst))
        return jsonify({"ok": True, "path": str(dst), "filename": dst.name}), 200
    except Exception as exc:
        logger.exception("upload-test error")
        return jsonify({"ok": False, "error": str(exc)}), 500


def main() -> None:  # pragma: no cover
    start_cleanup_thread()
    if os.environ.get("DFDETECT_EAGER_LOAD", "1") == "1":
        get_predictor()
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1", use_reloader=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


if __name__ == "__main__":  # pragma: no cover
    main()
