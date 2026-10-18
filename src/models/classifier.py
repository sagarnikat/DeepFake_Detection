"""DeepFake classifier heads built on top of the encoders in :mod:`backbone`."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d

from src.models.backbone import build_backbone, encoder_feature_dim


def setup_srm_weights(input_channels: int = 3) -> torch.Tensor:
    """SRM kernels for noise analysis (Zhou et al., CVPR 2018)."""
    srm_kernel = torch.from_numpy(
        np.array(
            [
                [  # srm 1/2 horiz
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, -2.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [  # srm 1/4
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 2.0, -1.0, 0.0],
                    [0.0, 2.0, -4.0, 2.0, 0.0],
                    [0.0, -1.0, 2.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [  # srm 1/12
                    [-1.0, 2.0, -2.0, 2.0, -1.0],
                    [2.0, -6.0, 8.0, -6.0, 2.0],
                    [-2.0, 8.0, -12.0, 8.0, -2.0],
                    [2.0, -6.0, 8.0, -6.0, 2.0],
                    [-1.0, 2.0, -2.0, 2.0, -1.0],
                ],
            ]
        )
    ).float()
    srm_kernel[0] /= 2
    srm_kernel[1] /= 4
    srm_kernel[2] /= 12
    return srm_kernel.view(3, 1, 5, 5).repeat(1, input_channels, 1, 1)


def setup_srm_layer(input_channels: int = 3) -> nn.Module:
    """Conv layer with frozen SRM kernels."""
    weights = setup_srm_weights(input_channels)
    conv = nn.Conv2d(input_channels, out_channels=3, kernel_size=5, stride=1, padding=2, bias=False)
    with torch.no_grad():
        conv.weight = nn.Parameter(weights, requires_grad=False)
    return conv


class GlobalWeightedAvgPool2d(nn.Module):
    """Global Weighted Average Pooling."""

    def __init__(self, features: int, flatten: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(features, 1, kernel_size=1, bias=True)
        self.flatten = flatten

    def fscore(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).sigmoid().exp()

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.sum(dim=[2, 3], keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_x = x
        x = self.norm(self.fscore(x)) * input_x
        return x.sum(dim=[2, 3], keepdim=not self.flatten)


class DeepFakeClassifier(nn.Module):
    """EfficientNet encoder + global pool + dropout + linear head."""

    def __init__(self, encoder: str, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.encoder = build_backbone(encoder)
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_feature_dim(encoder), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class DeepFakeClassifierSRM(nn.Module):
    """Variant that feeds SRM noise residuals into the encoder."""

    def __init__(self, encoder: str, dropout_rate: float = 0.5) -> None:
        super().__init__()
        self.encoder = build_backbone(encoder)
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.srm_conv = setup_srm_layer(3)
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_feature_dim(encoder), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = self.srm_conv(x)
        x = self.encoder.forward_features(noise)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class DeepFakeClassifierGWAP(nn.Module):
    """Variant using global weighted average pooling instead of plain GAP."""

    def __init__(self, encoder: str, dropout_rate: float = 0.5) -> None:
        super().__init__()
        self.encoder = build_backbone(encoder)
        self.avg_pool = GlobalWeightedAvgPool2d(encoder_feature_dim(encoder))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_feature_dim(encoder), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def load_classifier_weights(model: nn.Module, path: str, device: str = "cpu") -> nn.Module:
    """Load a checkpoint into ``model`` (handles ``state_dict`` / ``module.`` prefix)."""
    import re

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only kwarg
        checkpoint = torch.load(path, map_location=device)

    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    cleaned = {re.sub(r"^module\.", "", k): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model
