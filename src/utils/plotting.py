"""Plotting helpers (training curves, loss curves).

Ported from the original ``plot_loss.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np


def _import_plt():
    import matplotlib

    matplotlib.use("Agg")  # headless-safe default
    from matplotlib import pyplot as plt

    return plt


def plot_losses(
    train_losses: Sequence[float],
    val_losses: Sequence[float] | None = None,
    out_path: str | Path | None = "results/figures/loss_curve.png",
    title: str = "Training loss",
):
    """Plot train/val loss curves and (optionally) save them."""
    plt = _import_plt()
    plt.figure(figsize=(8, 5))
    epochs = np.arange(len(train_losses))
    plt.plot(epochs, train_losses, label="train")
    if val_losses is not None:
        plt.plot(np.arange(len(val_losses)), val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close()
    return out_path


def plot_weighted_loss(log_file: str, fake_weight: float = 1.4, out_path: str | Path | None = None):
    """Parse ``fake_loss``/``real_loss`` lines from a log file and plot them.

    Mirrors the behaviour of the original script.
    """
    plt = _import_plt()
    real_losses, fake_losses = [], []
    with open(log_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("fake_loss"):
                fake_losses.append(float(line.split(" ")[-1]))
            elif line.startswith("real_loss"):
                real_losses.append(float(line.split(" ")[-1]))

    real_losses = np.array(real_losses)
    fake_losses = np.array(fake_losses)
    loss = (fake_losses * fake_weight + real_losses) / 2

    plt.figure(figsize=(8, 5))
    plt.title(f"Weighted loss ({fake_weight}*fake_loss + real_loss)/2")
    best_loss_idx = np.argsort(loss)[:5]
    best_loss_idx = best_loss_idx[best_loss_idx > 16]
    plt.scatter(best_loss_idx, loss[best_loss_idx], c="red")
    for idx in best_loss_idx:
        plt.annotate(str(idx), (idx, loss[idx]))
    plt.plot(loss)
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close()
    else:
        plt.show()
    return out_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Plot losses from log")
    parser.add_argument("--log-file", help="path to log file", required=True)
    parser.add_argument("--fake-weight", help="weight for fake loss", default=1.4, type=float)
    parser.add_argument("--out", help="output image path", default="results/figures/weighted_loss.png")
    args = parser.parse_args()
    plot_weighted_loss(args.log_file, fake_weight=args.fake_weight, out_path=args.out)


if __name__ == "__main__":
    _main()
