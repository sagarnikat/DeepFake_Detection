"""Model registry: map a short name to a builder so backbones are easy to swap."""

from __future__ import annotations

from typing import Any, Callable, Dict

from src.models.classifier import (
    DeepFakeClassifier,
    DeepFakeClassifierGWAP,
    DeepFakeClassifierSRM,
)

MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_model(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator registering a model builder under ``name``."""

    def wrapper(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in MODEL_REGISTRY:
            raise KeyError(f"Model '{name}' is already registered")
        MODEL_REGISTRY[name] = fn
        return fn

    return wrapper


@register_model("resnet_gap")
def build_resnet_gap(encoder: str = "tf_efficientnet_b7_ns", dropout_rate: float = 0.0, **_: Any) -> DeepFakeClassifier:
    """Standard classifier (encoder + GAP + linear)."""
    return DeepFakeClassifier(encoder=encoder, dropout_rate=dropout_rate)


@register_model("srm")
def build_srm(encoder: str = "tf_efficientnet_b7_ns", dropout_rate: float = 0.5, **_: Any) -> DeepFakeClassifierSRM:
    """SRM noise-augmented classifier."""
    return DeepFakeClassifierSRM(encoder=encoder, dropout_rate=dropout_rate)


@register_model("gwap")
def build_gwap(encoder: str = "tf_efficientnet_b7_ns", dropout_rate: float = 0.5, **_: Any) -> DeepFakeClassifierGWAP:
    """Global weighted average pooling classifier."""
    return DeepFakeClassifierGWAP(encoder=encoder, dropout_rate=dropout_rate)


def build_model(name: str = "resnet_gap", encoder: str = "tf_efficientnet_b7_ns", **kwargs: Any):
    """Build a model by registered ``name``.

    Parameters
    ----------
    name:
        One of :data:`MODEL_REGISTRY` (``"resnet_gap"``, ``"srm"``, ``"gwap"``).
    encoder:
        Encoder/backbone name (see :mod:`src.models.backbone`).
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](encoder=encoder, **kwargs)
