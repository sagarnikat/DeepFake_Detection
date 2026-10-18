"""Core metrics + a DataLoader evaluator with Grad-CAM/figures hook.

Example
-------
python -m src.evaluation.evaluate --config configs/eval_crossdata.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.calibration.metrics import brier_score, expected_calibration_error
from src.utils.io import load_json, save_json
from src.utils.logging import get_logger

logger = get_logger("dfdetect.evaluate")


def evaluate_predictions(probs, labels, threshold: float = 0.5) -> Dict[str, float]:
    """Return a dict of classification metrics for probabilities vs labels."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    preds = (probs >= threshold).astype(int)

    metrics: Dict[str, float] = {
        "acc": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "brier": brier_score(probs, labels),
        "ece": expected_calibration_error(probs, labels),
    }
    try:
        metrics["auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["auc"] = float("nan")

    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    return metrics


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: Optional[str] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Run a model over a DataLoader and return metrics."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    probs, labels = [], []
    for images, targets in loader:
        logits = model(images.to(device)).view(-1)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(np.asarray(targets).ravel())
    probs = np.concatenate(probs) if probs else np.zeros((0,))
    labels = np.concatenate(labels) if labels else np.zeros((0,))
    return evaluate_predictions(probs, labels, threshold=threshold)


def main() -> None:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint")
    parser.add_argument("--predictions", help="json/csv with probs+labels, or --checkpoint")
    parser.add_argument("--checkpoint")
    parser.add_argument("--out", default="results/tables/metrics.json")
    args = parser.parse_args()

    if args.checkpoint:
        raise SystemExit("Provide --predictions for now (checkpoint eval wired via configs).")

    data = load_json(args.predictions)
    metrics = evaluate_predictions(data["probs"], data["labels"])
    save_json(metrics, args.out)
    logger.info("metrics: %s", metrics)
    print(metrics)


if __name__ == "__main__":  # pragma: no cover
    main()
