"""Grad-CAM (Selvaraju et al., 2017) — a small self-contained implementation.

Works with :class:`src.models.classifier.DeepFakeClassifier`: the target layer
defaults to the encoder's final convolution (``model.encoder.conv_head`` when
available, else the last ``Conv2d`` found).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Compute Grad-CAM heatmaps for a (single-logit) classifier."""

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None) -> None:
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or self._infer_target_layer(model)
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._handles: list = []

    @staticmethod
    def _infer_target_layer(model: torch.nn.Module) -> torch.nn.Module:
        encoder = getattr(model, "encoder", None)
        if encoder is not None and hasattr(encoder, "conv_head"):
            return encoder.conv_head
        last_conv = None
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv2d layer found to attach Grad-CAM to")
        return last_conv

    def _register(self) -> None:
        def forward_hook(_m, _i, output):
            self._activations = output.detach()

        def backward_hook(_m, _gi, grad_output):
            self._gradients = grad_output[0].detach()

        self._handles = [
            self.target_layer.register_forward_hook(forward_hook),
            self.target_layer.register_full_backward_hook(backward_hook),
        ]

    def _remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def generate(self, image: torch.Tensor, target: float = 1.0) -> np.ndarray:
        """Return a normalized ``HxW`` heatmap (float in [0, 1]) for ``image``.

        ``image``: ``(1, 3, H, W)`` tensor. ``target``: 1.0 for "fake" evidence
        (the model has a single logit), 0.0 for "real".
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        self._register()
        try:
            image = image.requires_grad_(True)
            logit = self.model(image).view(-1)[0]
            self.model.zero_grad(set_to_none=True)
            score = logit if target >= 0.5 else -logit
            score.backward()

            acts = self._activations  # (1, C, h, w)
            grads = self._gradients  # (1, C, h, w)
            weights = grads.mean(dim=(2, 3), keepdim=True)  # GAP over spatial
            cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
            cam = cam.squeeze().cpu().numpy()
            cam -= cam.min()
            if cam.max() > 1e-8:
                cam /= cam.max()
            return cam.astype(np.float32)
        finally:
            self._remove()

    def __call__(self, image: torch.Tensor, target: float = 1.0) -> np.ndarray:
        return self.generate(image, target=target)
