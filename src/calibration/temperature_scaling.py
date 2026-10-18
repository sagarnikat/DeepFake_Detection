"""Temperature scaling for binary classifiers (Guo et al., 2017)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """Learn a single temperature ``T`` so that ``sigmoid(logit / T)`` is calibrated.

    Fit on a held-out validation set by minimizing NLL.
    """

    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(float(init_temperature)).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Return calibrated probabilities for raw ``logits``."""
        return torch.sigmoid(logits.view(-1) / self.temperature)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, max_iter: int = 100) -> "TemperatureScaler":
        """Fit ``T`` on validation logits/labels."""
        logits = logits.detach().view(-1).float()
        labels = labels.detach().view(-1).float()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter)

        def _closure():
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(_closure)
        return self
