"""Calibration metrics: Expected Calibration Error and reliability diagrams."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _bin_edges(n_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def expected_calibration_error(probs, labels, n_bins: int = 15) -> float:
    """Expected Calibration Error for binary probabilities.

    Confidence is ``max(p, 1-p)`` and accuracy is whether that prediction is
    correct, which is the standard ECE formulation for binary classifiers.
    """
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    confidences = np.maximum(probs, 1 - probs)
    predictions = (probs >= 0.5).astype(int)
    accuracies = (predictions == labels).astype(float)

    edges = _bin_edges(n_bins)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        prop = mask.mean()
        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        ece += prop * abs(avg_acc - avg_conf)
    return float(ece)


def reliability_diagram(probs, labels, n_bins: int = 15) -> Dict[str, List[float]]:
    """Return bin statistics to draw a reliability diagram."""
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    confidences = np.maximum(probs, 1 - probs)
    predictions = (probs >= 0.5).astype(int)
    accuracies = (predictions == labels).astype(float)

    edges = _bin_edges(n_bins)
    centers, accs, confs, counts = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        centers.append(float((lo + hi) / 2))
        if np.any(mask):
            accs.append(float(accuracies[mask].mean()))
            confs.append(float(confidences[mask].mean()))
        else:
            accs.append(0.0)
            confs.append(0.0)
        counts.append(int(mask.sum()))
    return {"bin_centers": centers, "accuracy": accs, "confidence": confs, "counts": counts}


def brier_score(probs, labels) -> float:
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    return float(np.mean((probs - labels) ** 2))


def negative_log_likelihood(probs, labels) -> float:
    probs = np.clip(np.asarray(probs, dtype=float).ravel(), 1e-7, 1 - 1e-7)
    labels = np.asarray(labels, dtype=float).ravel()
    return float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))
