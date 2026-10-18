"""Single-image deepfake prediction (+ optional Grad-CAM).

Example
-------
python -m src.inference.predict_image --image face.jpg --weights weights/model_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch

from src.inference.predict_video import classify_label, load_models, resolve_device
from src.preprocessing.face_detector import FaceDetector, isotropically_resize_image, put_to_center
from src.utils.io import encode_image_b64
from src.utils.logging import get_logger

logger = get_logger("dfdetect.predict_image")


@torch.no_grad()
def predict_image(
    image_path: str | Path,
    models: list[torch.nn.Module],
    device: Optional[str] = None,
    input_size: int = 380,
    threshold: float = 0.5,
    detect_face: bool = True,
) -> dict:
    """Predict a single face image. Returns a result dict."""
    device = resolve_device(device)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if detect_face:
        detections = FaceDetector(device=device).detect(img, resize_factor=0.5)
        if detections:
            (x, y, w, h), _ = max(detections, key=lambda d: d[1])
            x, y = max(0, x), max(0, y)
            img = img[y : y + h, x : x + w]

    crop = put_to_center(isotropically_resize_image(img, input_size), input_size)
    tensor = torch.as_tensor(crop, device=device).float().permute(2, 0, 1) / 255.0
    tensor = (tensor - torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]) / torch.tensor(
        [0.229, 0.224, 0.225], device=device
    )[:, None, None]
    tensor = tensor.unsqueeze(0)
    if device == "cuda":
        tensor = tensor.half()

    probs = []
    for model in models:
        logit = model(tensor).view(-1)[0]
        probs.append(float(torch.sigmoid(logit.float())))
    prob = float(np.mean(probs))
    label, confidence = classify_label(prob, threshold=threshold)
    return {
        "label": label,
        "confidence": confidence,
        "probability": prob,
        "image_b64": encode_image_b64(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)),
    }


def gradcam_overlay(image_path: str | Path, model: torch.nn.Module, device: Optional[str] = None, input_size: int = 380) -> np.ndarray:
    """Return a Grad-CAM overlay (BGR) for a single image."""
    from src.explainability.gradcam import GradCAM
    from src.explainability.overlay import overlay_heatmap

    device = resolve_device(device)
    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    crop = put_to_center(isotropically_resize_image(img, input_size), input_size)
    tensor = torch.as_tensor(crop, device=device).float().permute(2, 0, 1) / 255.0
    tensor = tensor.unsqueeze(0)
    cam = GradCAM(model).generate(tensor, target=1.0)
    return overlay_heatmap(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), cam)


def main() -> None:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description="Predict a single image")
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--weights", default="model_v1")
    parser.add_argument("--no-face-detect", action="store_true")
    parser.add_argument("--gradcam", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    models = load_models(args.weights_dir, args.weights)
    res = predict_image(args.image, models, detect_face=not args.no_face_detect)
    logger.info("label=%s confidence=%.2f", res["label"], res["confidence"])
    print({k: v for k, v in res.items() if k != "image_b64"})

    if args.gradcam and args.out:
        overlay = gradcam_overlay(args.image, models[0])
        cv2.imwrite(args.out, overlay)
        logger.info("Saved Grad-CAM overlay to %s", args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
