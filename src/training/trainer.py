"""Generic train/val loop with AMP, checkpointing and early stopping."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.io import ensure_dir, save_json
from src.utils.logging import get_logger

logger = get_logger("dfdetect.trainer")


def _binary_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    preds = (probs >= 0.5).astype(int)
    metrics: Dict[str, float] = {"acc": float(accuracy_score(labels, preds))}
    try:
        metrics["auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["auc"] = float("nan")
    metrics["f1"] = float(f1_score(labels, preds, zero_division=0))
    return metrics


class Trainer:
    """Train a binary (real/fake) classifier."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        device: Optional[str] = None,
        scheduler: Any = None,
        checkpoint_dir: str | Path = "results/checkpoints",
        amp: bool = True,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.checkpoint_dir = ensure_dir(checkpoint_dir)
        self.amp = amp and self.device == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.history: list[dict] = []
        self.best_metric = -np.inf

    # ------------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        total_loss, all_probs, all_labels = 0.0, [], []
        n = 0
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).view(-1)
            if self.amp:
                images = images.half()

            with torch.set_grad_enabled(train):
                with torch.autocast(device_type="cuda", enabled=self.amp):
                    logits = self.model(images).view(-1)
                    loss = self.loss_fn(logits, labels)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            probs = torch.sigmoid(logits.detach().float()).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.detach().float().cpu().numpy())
            total_loss += float(loss.item()) * len(labels)
            n += len(labels)

        probs = np.concatenate(all_probs) if all_probs else np.zeros((0,))
        labels = np.concatenate(all_labels) if all_labels else np.zeros((0,))
        metrics = _binary_metrics(probs, labels) if n else {"acc": 0.0, "auc": float("nan"), "f1": 0.0}
        metrics["loss"] = total_loss / max(1, n)
        return metrics

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 20,
        patience: int = 5,
        monitor: str = "auc",
    ) -> list[dict]:
        """Run training. Returns the per-epoch history."""
        epochs_without_improvement = 0
        for epoch in range(1, epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True)
            record = {"epoch": epoch, "train": train_metrics}

            if val_loader is not None:
                val_metrics = self._run_epoch(val_loader, train=False)
                record["val"] = val_metrics
                score = val_metrics.get(monitor, -np.inf)
            else:
                score = -train_metrics["loss"]

            if self.scheduler is not None:
                self.scheduler.step()

            logger.info("epoch %d | train %s | val %s", epoch, train_metrics, record.get("val"))

            if score > self.best_metric:
                self.best_metric = score
                self.save_checkpoint("best.pt", extra={"epoch": epoch, "metrics": record})
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            self.history.append(record)
            save_json(self.history, self.checkpoint_dir / "history.json")

            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %d (no %s improvement for %d epochs)", epoch, monitor, patience)
                break

        return self.history

    # ------------------------------------------------------------------
    def save_checkpoint(self, name: str = "last.pt", extra: Optional[Dict[str, Any]] = None) -> Path:
        path = self.checkpoint_dir / name
        torch.save({"state_dict": self.model.state_dict(), **(extra or {})}, path)
        return path

    def load_checkpoint(self, path: str | Path, strict: bool = True) -> Dict[str, Any]:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        self.model.load_state_dict(state, strict=strict)
        return ckpt
