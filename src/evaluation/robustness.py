"""Robustness evaluation under compression / blur / noise.

Deepfake detectors are notoriously fragile to post-processing. This applies a
grid of perturbations to eval images and reports the metric drop, which is a
standard expectation for a strong report.
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.evaluation.evaluate import evaluate_loader
from src.utils.io import save_json
from src.utils.logging import get_logger

logger = get_logger("dfdetect.robustness")


def jpeg_compress(image_rgb: np.ndarray, quality: int) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return image_rgb
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def gaussian_blur(image_rgb: np.ndarray, ksize: int = 7) -> np.ndarray:
    return cv2.GaussianBlur(image_rgb, (ksize, ksize), 0)


def gaussian_noise(image_rgb: np.ndarray, sigma: float = 12.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, image_rgb.shape)
    return np.clip(image_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)


class PerturbedSubset(Subset):
    """Wrap a dataset subset and apply a numpy perturbation before the transform."""

    def __init__(self, dataset, indices, perturbation: Callable[[np.ndarray], np.ndarray]):
        super().__init__(dataset, indices)
        self.perturbation = perturbation

    def __getitem__(self, idx):
        base = self.dataset
        # fetch raw image via underlying FaceDataset
        inner = base
        if hasattr(base, "dataset"):
            inner = base.dataset
        row = inner.rows[idx]
        from pathlib import Path

        img = cv2.imread(str(row["path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.perturbation(img)
        if inner.transform is not None:
            out = inner.transform(image=img)["image"]
            tensor = torch.as_tensor(out)
            if tensor.ndim == 3 and tensor.shape[-1] == 3:
                tensor = tensor.permute(2, 0, 1)
            tensor = tensor.float()
            if tensor.max() > 1.5:
                tensor = tensor / 255.0
        else:
            img = cv2.resize(img, (inner.input_size, inner.input_size))
            tensor = torch.as_tensor(img).permute(2, 0, 1).float() / 255.0
        return tensor, float(row["label"])


def robustness_report(
    model: torch.nn.Module,
    dataset,
    perturbations: Dict[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    max_samples: int = 512,
    batch_size: int = 16,
    num_workers: int = 2,
    device: str | None = None,
    out_path: str | None = "results/tables/robustness.json",
) -> Dict[str, Dict[str, float]]:
    """Evaluate the model under each perturbation."""
    n = min(len(dataset), max_samples)
    indices = list(range(n))
    if perturbations is None:
        perturbations = {
            "clean": lambda im: im,
            "jpeg_q70": lambda im: jpeg_compress(im, 70),
            "jpeg_q50": lambda im: jpeg_compress(im, 50),
            "blur_k7": lambda im: gaussian_blur(im, 7),
            "noise_sigma12": lambda im: gaussian_noise(im, 12.0),
        }

    report: Dict[str, Dict[str, float]] = {}
    for name, fn in perturbations.items():
        subset = PerturbedSubset(dataset, indices, fn)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        report[name] = evaluate_loader(model, loader, device=device)
        logger.info("%s -> auc=%.4f acc=%.4f", name, report[name].get("auc", float("nan")), report[name]["acc"])

    if out_path:
        save_json(report, out_path)
    return report


def main() -> None:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description="Robustness evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="celebdf")
    parser.add_argument("--encoder", default="tf_efficientnet_b7_ns")
    parser.add_argument("--out", default="results/tables/robustness.json")
    args = parser.parse_args()
    logger.info("robustness CLI is a thin wrapper; see robustness_report().")


if __name__ == "__main__":  # pragma: no cover
    main()
