"""Video-level deepfake prediction.

This module contains the prediction pipeline that previously lived inside the
Flask ``app.py`` (plus the batch logic from ``predict_folder.py``), refactored
into a reusable :class:`VideoPredictor`. The web app, Gradio demo and the CLI
all share this single implementation.

Pipeline: sample frames -> detect faces (MTCNN) -> crop/resize to 380x380 ->
EfficientNet-B7 classifier (optionally an ensemble) -> aggregate to a
video-level probability -> collect top suspicious frames + forgery reasons.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch

from src.models.classifier import DeepFakeClassifier
from src.preprocessing.face_detector import (
    FaceExtractor,
    image_compression,
    isotropically_resize_image,
    put_to_center,
)
from src.preprocessing.frame_sampler import VideoReader, compute_frames_to_sample
from src.utils.io import ensure_dir, encode_image_b64, list_videos
from src.utils.logging import get_logger

logger = get_logger("dfdetect.predict")

# ---------------------------------------------------------------------------
# configuration (env-overridable, matching the original backend defaults)
# ---------------------------------------------------------------------------
INPUT_SIZE = int(os.environ.get("DFDETECT_INPUT_SIZE", "380"))
TOP_K_FRAMES = int(os.environ.get("DFDETECT_TOP_K", "8"))
MIN_SAMPLE_FRAMES = int(os.environ.get("DFDETECT_MIN_FRAMES", "16"))
MAX_SAMPLE_FRAMES = int(os.environ.get("DFDETECT_MAX_FRAMES", "64"))
SAMPLES_PER_SECOND = float(os.environ.get("DFDETECT_SPS", "1.0"))
WEIGHTS_DIR = os.environ.get("DFDETECT_WEIGHTS_DIR", "weights")
DEFAULT_WEIGHTS_NAME = os.environ.get("DFDETECT_WEIGHTS", "model_v1")


def resolve_device(device: Optional[str] = None) -> str:
    return device or os.environ.get("DFDETECT_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def resolve_weight_names(weights: Union[str, Sequence[str], None] = None) -> Union[str, Tuple[str, ...]]:
    """Normalise the ``DFDETECT_WEIGHTS`` env var into a name or tuple of names."""
    value = weights if weights is not None else DEFAULT_WEIGHTS_NAME
    if isinstance(value, (list, tuple)):
        return tuple(value)
    value = str(value)
    if "," in value:
        return tuple(w.strip() for w in value.split(",") if w.strip())
    return value.strip()


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------
def load_models(
    weights_dir: str | Path = WEIGHTS_DIR,
    weight_names: Union[str, Sequence[str], None] = None,
    device: Optional[str] = None,
    encoder: str = "tf_efficientnet_b7_ns",
) -> List[torch.nn.Module]:
    """Load one or more ``DeepFakeClassifier`` checkpoints."""
    device = resolve_device(device)
    names = resolve_weight_names(weight_names)
    paths = [Path(weights_dir) / n for n in (names if isinstance(names, (list, tuple)) else [names])]

    models: List[torch.nn.Module] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Weight file not found: {path}. Run ./scripts/download_weights.sh first."
            )
        model = DeepFakeClassifier(encoder=encoder).to(device)
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:  # torch < 2.0
            checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        cleaned = {re.sub(r"^module\.", "", k): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=True)
        model.eval()
        if device == "cuda":
            model = model.half()
        models.append(model)
    logger.info("Loaded %d model(s) on %s", len(models), device)
    return models


# ---------------------------------------------------------------------------
# aggregation / labels
# ---------------------------------------------------------------------------
def confident_strategy(pred, t: float = 0.8) -> float:
    pred = np.asarray(pred, dtype=float)
    sz = len(pred)
    if sz == 0:
        return 0.5
    fakes = np.count_nonzero(pred > t)
    if fakes > sz // 2.5 and fakes > 11:
        return float(np.mean(pred[pred > t]))
    if np.count_nonzero(pred < 0.2) > 0.9 * sz:
        return float(np.mean(pred[pred < 0.2]))
    return float(np.mean(pred))


def classify_label(prob: float, threshold: float = 0.5) -> Tuple[str, float]:
    prob = float(prob)
    if prob >= threshold:
        return "FAKE", round(prob * 100, 2)
    return "REAL", round((1 - prob) * 100, 2)


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def draw_box_and_label(frame_bgr: np.ndarray, box, label_text: str, score: float, label_col=(0, 200, 150)) -> None:
    x, y, w, h = [int(v) for v in box]
    thickness = max(1, int(round(min(frame_bgr.shape[0], frame_bgr.shape[1]) / 200)))
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), label_col, thickness=thickness)
    text = f"{label_text} {round(score * 100, 2)}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, min(frame_bgr.shape[1] / 1200.0, 0.9))
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 6
    cv2.rectangle(frame_bgr, (x, max(0, y - text_h - 2 * pad)), (x + text_w + 2 * pad, max(0, y)), (10, 10, 10), cv2.FILLED)
    cv2.putText(frame_bgr, text, (x + pad, max(0, y) - pad), font, font_scale, label_col, thickness=thickness, lineType=cv2.LINE_AA)


# ---------------------------------------------------------------------------
# forgery-reason heuristics
# ---------------------------------------------------------------------------
def high_freq_energy(gray_crop: np.ndarray) -> float:
    try:
        f = np.fft.fft2(gray_crop)
        magnitude = np.abs(np.fft.fftshift(f))
        h, w = magnitude.shape
        band = magnitude.copy()
        cy, cx = h // 2, w // 2
        band[cy - h // 4 : cy + h // 4, cx - w // 4 : cx + w // 4] = 0
        return float(np.sum(band) / (h * w) + 1e-6)
    except Exception:
        return 0.0


def boundary_artifact_score(face_gray: np.ndarray) -> float:
    try:
        h, w = face_gray.shape[:2]
        border = np.concatenate(
            [
                face_gray[0:3, :].ravel() if h > 3 else np.array([]),
                face_gray[-3:, :].ravel() if h > 3 else np.array([]),
                face_gray[:, 0:3].ravel() if w > 3 else np.array([]),
                face_gray[:, -3:].ravel() if w > 3 else np.array([]),
            ]
        )
        inner = face_gray[3:-3, 3:-3].ravel() if (h > 8 and w > 8) else face_gray.ravel()
        if inner.size == 0 or border.size == 0:
            return 0.0
        b_var = np.var(border)
        i_var = np.var(inner) + 1e-6
        return float(np.tanh(abs(b_var - i_var) / (i_var + b_var) * 2.0))
    except Exception:
        return 0.0


def color_consistency_score(face_rgb: np.ndarray, neck_rgb: np.ndarray) -> float:
    try:
        face_hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV).mean(axis=(0, 1))
        neck_hsv = cv2.cvtColor(neck_rgb, cv2.COLOR_RGB2HSV).mean(axis=(0, 1))
        dist = np.linalg.norm(face_hsv - neck_hsv) / (np.linalg.norm([255, 255, 255]) + 1e-6)
        return float(min(1.0, dist * 2.0))
    except Exception:
        return 0.0


def analyze_video_temporal_for_flow(frames_rgb, frame_face_boxes_map: Dict[int, list], idxs_read: list) -> dict:
    flow_mags = []
    prev_gray = None
    for pos, fidx in enumerate(idxs_read):
        gray = cv2.cvtColor(frames_rgb[pos], cv2.COLOR_RGB2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            boxes = frame_face_boxes_map.get(fidx, [])
            vals = []
            for box in boxes:
                x, y, w, h = [int(v) for v in box]
                x2, y2 = min(x + w, mag.shape[1] - 1), min(y + h, mag.shape[0] - 1)
                if x < 0 or y < 0 or x >= mag.shape[1] or y >= mag.shape[0]:
                    continue
                crop = mag[y:y2, x:x2]
                if crop.size:
                    vals.append(np.mean(crop))
            flow_mags.append(float(np.mean(vals)) if vals else float(np.mean(mag)))
        prev_gray = gray
    return {
        "flow_variance": float(np.var(flow_mags)) if flow_mags else 0.0,
        "flow_mean": float(np.mean(flow_mags)) if flow_mags else 0.0,
        "flow_samples": flow_mags,
    }


def generate_forgery_reasons(frame_analyses: list, temporal_metrics: dict, thresholds: Optional[dict] = None, top_k_frames_map=None) -> dict:
    thresholds = thresholds or {
        "boundary": 0.45,
        "hf_low": 5000,
        "hf_high": 40000,
        "color_consistency": 0.25,
        "flow_var": 0.6,
        "model_conf": 0.6,
    }
    reasons = []
    model_scores = [f.get("model_score") for f in frame_analyses if f.get("model_score") is not None]
    mean_model = float(np.mean(model_scores)) if model_scores else None
    if mean_model is not None and mean_model >= thresholds["model_conf"]:
        reasons.append(
            {
                "type": "model_confidence",
                "score": mean_model,
                "reason": f"Classifier indicates manipulation (average model score {mean_model:.3f}).",
                "evidence_frames": [int(f[0]) for f in (top_k_frames_map or [])],
            }
        )

    boundary_vals = [f["boundary_score"] for f in frame_analyses]
    if boundary_vals and float(np.mean(boundary_vals)) > thresholds["boundary"]:
        sorted_b = sorted(frame_analyses, key=lambda x: x["boundary_score"], reverse=True)[:3]
        reasons.append(
            {
                "type": "boundary_artifact",
                "score": float(np.mean(boundary_vals)),
                "reason": "Visible blending/edge artifacts detected around face boundaries.",
                "evidence_frames": [int(f["frame_idx"]) for f in sorted_b],
            }
        )

    hf_vals = [f["hf_energy"] for f in frame_analyses]
    if hf_vals:
        hf_mean = float(np.mean(hf_vals))
        if hf_mean < thresholds["hf_low"]:
            sel = sorted(frame_analyses, key=lambda x: x["hf_energy"])[:3]
            reasons.append(
                {
                    "type": "texture_smooth",
                    "score": hf_mean,
                    "reason": "Face texture appears overly smooth (lack of high-frequency detail), which is typical for many generated faces.",
                    "evidence_frames": [int(f["frame_idx"]) for f in sel],
                }
            )
        elif hf_mean > thresholds["hf_high"]:
            sel = sorted(frame_analyses, key=lambda x: x["hf_energy"], reverse=True)[:3]
            reasons.append(
                {
                    "type": "texture_noise",
                    "score": hf_mean,
                    "reason": "Unnatural high-frequency noise present on the face region (possible generation artifacts).",
                    "evidence_frames": [int(f["frame_idx"]) for f in sel],
                }
            )

    color_vals = [f["color_consistency"] for f in frame_analyses]
    if color_vals and float(np.mean(color_vals)) > thresholds["color_consistency"]:
        sel = sorted(frame_analyses, key=lambda x: x["color_consistency"], reverse=True)[:3]
        reasons.append(
            {
                "type": "lighting_mismatch",
                "score": float(np.mean(color_vals)),
                "reason": "Lighting/color on face differs noticeably from surrounding skin/background (possible compositing).",
                "evidence_frames": [int(f["frame_idx"]) for f in sel],
            }
        )

    if temporal_metrics.get("flow_variance", 0) > thresholds["flow_var"]:
        reasons.append(
            {
                "type": "temporal_inconsistency",
                "score": float(temporal_metrics.get("flow_variance", 0)),
                "reason": "Temporal inconsistency detected (irregular motion/jitter across frames).",
                "evidence_frames": [int(f[0]) for f in (top_k_frames_map or [])],
            }
        )

    if not reasons:
        reasons.append(
            {
                "type": "no_strong_evidence",
                "score": 0.0,
                "reason": "No strong forgery evidence found by heuristics. Rely on model confidence and inspect frames if unsure.",
                "evidence_frames": [int(f[0]) for f in (top_k_frames_map or [])],
            }
        )

    return {"reasons": reasons, "mean_model_score": mean_model, "temporal_metrics": temporal_metrics}


def _build_fallback_frame_analyses(processed_faces, face_frame_refs, face_boxes, per_face_probs) -> list:
    analyses = []
    for face_idx, img in enumerate(processed_faces):
        try:
            face_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            H, W = img.shape[:2]
            neck = img[int(H * 0.75) : H, int(W * 0.25) : int(W * 0.75)] if H > 10 else img
            analyses.append(
                {
                    "frame_idx": int(face_frame_refs[face_idx]) if face_idx < len(face_frame_refs) else -1,
                    "boundary_score": float(boundary_artifact_score(cv2.resize(face_gray, (128, 128)))),
                    "hf_energy": float(high_freq_energy(cv2.resize(face_gray, (128, 128)))),
                    "color_consistency": float(color_consistency_score(img, neck if neck.size else img)),
                    "model_score": float(per_face_probs[face_idx]) if face_idx < len(per_face_probs) else None,
                }
            )
        except Exception:
            continue
    return analyses


# ---------------------------------------------------------------------------
# predictor
# ---------------------------------------------------------------------------
class VideoPredictor:
    """End-to-end video deepfake predictor."""

    def __init__(
        self,
        models: Sequence[torch.nn.Module],
        device: Optional[str] = None,
        input_size: int = INPUT_SIZE,
        top_k: int = TOP_K_FRAMES,
        batch_size: Optional[int] = None,
        video_reader: Optional[VideoReader] = None,
        face_extractor: Optional[FaceExtractor] = None,
        min_frames: int = MIN_SAMPLE_FRAMES,
        max_frames: int = MAX_SAMPLE_FRAMES,
        frames_per_second: float = SAMPLES_PER_SECOND,
        save_all_frames: bool = True,
        save_sample_frames: bool = True,
        save_annotated_frames: bool = True,
    ) -> None:
        self.device = resolve_device(device)
        self.models = list(models)
        self.input_size = input_size
        self.top_k = top_k
        self.batch_size = batch_size or int(
            os.environ.get("DFDETECT_BATCH", "8" if self.device == "cpu" else "128")
        )
        self.video_reader = video_reader or VideoReader()
        self.min_frames, self.max_frames, self.fps = min_frames, max_frames, frames_per_second
        self.video_read_fn = lambda path: self.video_reader.read_frames(
            path, num_frames=self.frames_to_sample(path)
        )
        self.face_extractor = face_extractor or FaceExtractor(self.video_read_fn)
        self.save_all_frames = save_all_frames
        self.save_sample_frames = save_sample_frames
        self.save_annotated_frames = save_annotated_frames

    @classmethod
    def from_weights(
        cls,
        weights_dir: str | Path = WEIGHTS_DIR,
        weights: Union[str, Sequence[str], None] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> "VideoPredictor":
        """Build a predictor by loading checkpoints from ``weights_dir``."""
        device = resolve_device(device)
        models = load_models(weights_dir, weights, device=device)
        return cls(models, device=device, **kwargs)

    def frames_to_sample(self, video_path: str | Path) -> int:
        return compute_frames_to_sample(
            video_path, min_frames=self.min_frames, max_frames=self.max_frames, frames_per_second=self.fps
        )

    # ------------------------------------------------------------------
    def _forward(self, faces_np: np.ndarray) -> np.ndarray:
        """Run the ensemble over a ``(N, H, W, 3)`` uint8 stack; return per-face probs."""
        from torchvision.transforms import Normalize

        normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        x = torch.tensor(faces_np, device=self.device).float().permute(0, 3, 1, 2)
        for i in range(x.shape[0]):
            x[i] = normalize(x[i] / 255.0)
        if self.device == "cuda":
            x = x.half()

        n = x.shape[0]
        per_model = [[] for _ in self.models]
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                batch = x[start : start + self.batch_size]
                for mi, model in enumerate(self.models):
                    y = model(batch)
                    per_model[mi].append(torch.sigmoid(y).reshape(-1).cpu().numpy())
        stacked = np.array([np.concatenate(per_model[mi]) for mi in range(len(self.models))])
        return stacked.mean(axis=0)

    # ------------------------------------------------------------------
    def predict(
        self,
        video_path: str | Path,
        save_dir: Optional[str | Path] = None,
        top_k: Optional[int] = None,
        apply_compression: bool = False,
    ) -> dict:
        """Predict a single video. Returns a JSON-serialisable result dict."""
        top_k = top_k or self.top_k
        video_path = str(video_path)
        save_dir = Path(save_dir) if save_dir else None

        if save_dir and self.save_all_frames:
            self._save_all_frames(video_path, save_dir)

        faces_info = self.face_extractor.process_video(video_path)
        if not faces_info:
            return self._no_faces("No faces found")

        processed_faces, face_frame_refs, face_boxes = [], [], []
        for frame_data in faces_info:
            fidx = frame_data["frame_idx"]
            boxes = frame_data.get("boxes", [])
            for i, face in enumerate(frame_data.get("faces", [])):
                centered = put_to_center(isotropically_resize_image(face, self.input_size), self.input_size)
                if apply_compression:
                    centered = image_compression(centered, quality=90, image_type=".jpg")
                processed_faces.append(centered)
                face_frame_refs.append(fidx)
                face_boxes.append(boxes[i] if i < len(boxes) else None)

        if not processed_faces:
            return self._no_faces("No faces after processing")

        per_face_probs = self._forward(np.stack(processed_faces, axis=0))

        frame_to_probs: Dict[int, list] = {}
        for idx_face, frame_idx in enumerate(face_frame_refs):
            frame_to_probs.setdefault(frame_idx, []).append(float(per_face_probs[idx_face]))
        frame_level = {fidx: max(probs) for fidx, probs in frame_to_probs.items()}

        video_prob = confident_strategy(per_face_probs)
        label, confidence = classify_label(video_prob)

        top_frames = sorted(frame_level.items(), key=lambda x: x[1], reverse=True)[:top_k]
        fallback_analyses = _build_fallback_frame_analyses(processed_faces, face_frame_refs, face_boxes, per_face_probs)

        result = {
            "label": label,
            "confidence": confidence,
            "video_probability": float(video_prob),
            "per_face_probs": [float(p) for p in per_face_probs],
            "frame_scores": [round(float(s), 4) for _, s in top_frames],
            "frames": [],
            "forgery_reasons": {},
            "paths": {},
            "saved": {},
        }

        full_res = self.video_reader.read_frames(video_path, num_frames=self.frames_to_sample(video_path))
        if full_res is None:
            temporal = {"flow_variance": 0.0, "flow_mean": 0.0, "flow_samples": []}
            result["forgery_reasons"] = generate_forgery_reasons(fallback_analyses, temporal, top_k_frames_map=top_frames)
            return result

        full_frames, idxs_read = full_res
        self._save_frames(video_path, save_dir, full_frames, idxs_read, face_frame_refs, face_boxes, per_face_probs)
        result["saved"] = self._saved_paths(save_dir)

        for fidx, score in top_frames:
            if fidx in idxs_read:
                pos = idxs_read.index(fidx)
                result["frames"].append(encode_image_b64(cv2.cvtColor(full_frames[pos], cv2.COLOR_RGB2BGR)))

        # per-frame heuristics
        idx_to_pos = {fidx: pos for pos, fidx in enumerate(idxs_read)}
        frame_face_boxes_map: Dict[int, list] = {}
        for face_idx, fidx in enumerate(face_frame_refs):
            if face_boxes[face_idx] is not None:
                frame_face_boxes_map.setdefault(fidx, []).append(face_boxes[face_idx])

        frame_analyses = []
        for face_idx, fidx in enumerate(face_frame_refs):
            pos = idx_to_pos.get(fidx)
            box = face_boxes[face_idx]
            if pos is None or box is None:
                continue
            frame_rgb = full_frames[pos]
            x, y, w, h = [int(v) for v in box]
            x2, y2 = min(x + w, frame_rgb.shape[1] - 1), min(y + h, frame_rgb.shape[0] - 1)
            face_crop = frame_rgb[y:y2, x:x2]
            ny = min(frame_rgb.shape[0], y2 + max(1, h // 6))
            neck_crop = frame_rgb[y2:ny, x:x2] if ny > y2 else face_crop
            if face_crop.size == 0:
                continue
            face_gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
            frame_analyses.append(
                {
                    "frame_idx": int(fidx),
                    "boundary_score": float(boundary_artifact_score(cv2.resize(face_gray, (128, 128)))),
                    "hf_energy": float(high_freq_energy(cv2.resize(face_gray, (128, 128)))),
                    "color_consistency": float(color_consistency_score(face_crop, neck_crop if neck_crop.size else face_crop)),
                    "model_score": float(per_face_probs[face_idx]) if face_idx < len(per_face_probs) else None,
                }
            )

        temporal = analyze_video_temporal_for_flow(full_frames, frame_face_boxes_map, idxs_read)
        result["forgery_reasons"] = generate_forgery_reasons(frame_analyses, temporal, top_k_frames_map=top_frames)
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _no_faces(message: str) -> dict:
        return {
            "label": "NO_FACES",
            "confidence": 0.0,
            "video_probability": 0.5,
            "per_face_probs": [],
            "frame_scores": [],
            "frames": [],
            "forgery_reasons": {"reasons": [{"type": "no_faces", "score": 0.0, "reason": message}], "mean_model_score": None, "temporal_metrics": {}},
            "paths": {},
            "saved": {},
        }

    def _save_all_frames(self, video_path: str, save_dir: Path) -> None:
        try:
            res = self.video_reader.read_frames(video_path, num_frames=None)
            if res is None:
                return
            frames, idxs = res
            out = ensure_dir(save_dir / "all_frames")
            for pos, fidx in enumerate(idxs):
                cv2.imwrite(str(out / f"frame_{fidx:06d}.jpg"), cv2.cvtColor(frames[pos], cv2.COLOR_RGB2BGR))
            logger.info("[Predict] Saved all frames to %s (%d files)", out, len(idxs))
        except Exception:
            logger.exception("Error while saving all frames")

    def _save_frames(self, video_path, save_dir, full_frames, idxs_read, face_frame_refs, face_boxes, per_face_probs) -> None:
        if save_dir is None:
            return
        frame_face_map: Dict[int, list] = {}
        for face_idx, fidx in enumerate(face_frame_refs):
            frame_face_map.setdefault(fidx, []).append(face_idx)

        annotated_dir = ensure_dir(save_dir / "frames") if self.save_annotated_frames else None
        sample_dir = ensure_dir(save_dir / "sample_frames") if self.save_sample_frames else None

        for pos, fidx in enumerate(idxs_read):
            frame_rgb = full_frames[pos]
            if annotated_dir is not None:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()
                for face_idx in frame_face_map.get(fidx, []):
                    prob = float(per_face_probs[face_idx])
                    lab, _ = classify_label(prob)
                    col = (30, 30, 200) if lab == "FAKE" else (50, 200, 50)
                    if face_boxes[face_idx] is not None:
                        draw_box_and_label(frame_bgr, face_boxes[face_idx], lab, prob, label_col=col)
                cv2.imwrite(str(annotated_dir / f"frame_{fidx:06d}.jpg"), frame_bgr)
            if sample_dir is not None:
                cv2.imwrite(str(sample_dir / f"frame_{fidx:06d}.jpg"), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    @staticmethod
    def _saved_paths(save_dir: Optional[Path]) -> dict:
        if save_dir is None:
            return {}
        return {
            "annotated_frames_path": str(save_dir / "frames"),
            "sample_frames_path": str(save_dir / "sample_frames"),
            "all_frames_path": str(save_dir / "all_frames"),
        }


# ---------------------------------------------------------------------------
# batch CLI (replaces the old predict_folder.py)
# ---------------------------------------------------------------------------
def main() -> None:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description="Predict deepfake videos with an ensemble")
    parser.add_argument("--weights-dir", default=WEIGHTS_DIR)
    parser.add_argument("--models", nargs="+", required=True, help="checkpoint filenames")
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--save-dir", default=None, help="optional dir to save annotated frames")
    args = parser.parse_args()

    import csv

    predictor = VideoPredictor.from_weights(args.weights_dir, tuple(args.models))
    videos = list_videos(args.test_dir)
    logger.info("Predicting %d videos", len(videos))

    rows = []
    for name in videos:
        out_dir = (Path(args.save_dir) / Path(name).stem) if args.save_dir else None
        res = predictor.predict(str(Path(args.test_dir) / name), save_dir=out_dir)
        rows.append({"filename": name, "probability": res["video_probability"], "prediction": res["label"], "accuracy(%)": res["confidence"]})
        logger.info("%s -> %s (%s%%)", name, res["label"], res["confidence"])

    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "probability", "prediction", "accuracy(%)"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved predictions to %s", args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
