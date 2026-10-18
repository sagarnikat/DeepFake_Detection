"""Gradio demo for the DeepFake Detection System.

Requires ``gradio`` (optional dependency):
    pip install gradio
Run:
    python -m app.demo
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from src.inference.predict_video import WEIGHTS_DIR, VideoPredictor, resolve_weight_names
from src.utils.logging import get_logger

logger = get_logger("dfdetect.demo")

_predictor: VideoPredictor | None = None


def _get_predictor() -> VideoPredictor:
    global _predictor
    if _predictor is None:
        _predictor = VideoPredictor.from_weights(WEIGHTS_DIR, resolve_weight_names())
    return _predictor


def predict_video_fn(video_path: str):
    """Gradio callback: video file path -> (label, confidence, frames, reasons)."""
    if not video_path:
        return "No video", 0.0, [], {}
    predictor = _get_predictor()
    result = predictor.predict(video_path, save_dir=None)
    if result["label"] == "NO_FACES":
        return "NO_FACES", 0.0, [], result["forgery_reasons"]
    return result["label"], float(result["confidence"]), result["frames"], result["forgery_reasons"]


def build_demo():
    """Construct and return the Gradio Blocks app."""
    import gradio as gr

    with gr.Blocks(title="DeepFake Detection System") as demo:
        gr.Markdown("# DeepFake Video Detection System\nUpload a video to check if it is REAL or FAKE.")
        with gr.Row():
            video_in = gr.Video(label="Input video")
            with gr.Column():
                label_out = gr.Label(label="Verdict")
                conf_out = gr.Number(label="Confidence (%)")
        frames_out = gr.Gallery(label="Top suspicious frames")
        reasons_out = gr.JSON(label="Forgery reasoning")
        btn = gr.Button("Analyze")
        btn.click(predict_video_fn, inputs=video_in, outputs=[label_out, conf_out, frames_out, reasons_out])
    return demo


def main() -> None:  # pragma: no cover
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_PORT", 7860)))


if __name__ == "__main__":  # pragma: no cover
    main()
