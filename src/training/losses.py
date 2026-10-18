"""Loss functions.

The original DFDC solution used a weighted sum of ``fake`` and ``real`` losses
to counter the strong class imbalance (few fakes). :func:`build_loss` supports
that as ``fake_weight`` on top of ``BCEWithLogitsLoss``.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCEWithLogitsLoss(nn.Module):
    """BCE-with-logits that up-weights the positive (fake) class.

    ``fake_weight`` > 1 makes the model pay more attention to fake samples,
    matching the heuristic used in the reference implementation.
    """

    def __init__(self, fake_weight: float = 1.0, label_smoothing: float = 0.0, pos_weight: Optional[float] = None) -> None:
        super().__init__()
        self.fake_weight = float(fake_weight)
        self.label_smoothing = float(label_smoothing)
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=None)
        if self.fake_weight != 1.0:
            weight = torch.where(targets > 0.5, self.fake_weight, 1.0)
            loss = loss * weight
        return loss.mean()


def build_loss(name: str = "bce", fake_weight: float = 1.0, label_smoothing: float = 0.0, **kwargs: Any) -> nn.Module:
    """Build a loss by name.

    Supported: ``"bce"`` / ``"weighted_bce"`` / ``"bce_logits"``.
    """
    name = name.lower()
    if name in ("bce", "bce_logits", "weighted_bce"):
        return WeightedBCEWithLogitsLoss(
            fake_weight=fake_weight,
            label_smoothing=label_smoothing,
            pos_weight=kwargs.get("pos_weight"),
        )
    raise KeyError(f"Unknown loss '{name}'")
