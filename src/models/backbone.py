"""Backbone (encoder) construction via ``timm``.

Adapted from the public DFDC challenge solution by Selim Seferbekov
(https://github.com/selimsef/dfdc_deepfake_challenge).

The original code imported the NoisyStudent EfficientNet builders
(``tf_efficientnet_*_ns``) directly from ``timm.models.efficientnet``. Newer
``timm`` releases (>= 1.0) removed those module-level functions, so
:func:`build_backbone` falls back to ``timm.create_model`` which still resolves
the legacy ``*_ns`` names through its deprecated-model map.

Encoders default to ``pretrained=False``: the ImageNet weights are fully
overwritten by the checkpoint we load for inference, so downloading them is
pointless (and breaks offline use).
"""

from __future__ import annotations

from typing import Any, Callable, Dict

try:  # timm <= 0.9
    from timm.models.efficientnet import (  # type: ignore
        tf_efficientnet_b2_ns,
        tf_efficientnet_b3_ns,
        tf_efficientnet_b4_ns,
        tf_efficientnet_b5_ns,
        tf_efficientnet_b6_ns,
        tf_efficientnet_b7_ns,
    )

    _DIRECT_BUILDERS: Dict[str, Callable[..., Any]] = {
        "tf_efficientnet_b2_ns": tf_efficientnet_b2_ns,
        "tf_efficientnet_b3_ns": tf_efficientnet_b3_ns,
        "tf_efficientnet_b4_ns": tf_efficientnet_b4_ns,
        "tf_efficientnet_b5_ns": tf_efficientnet_b5_ns,
        "tf_efficientnet_b6_ns": tf_efficientnet_b6_ns,
        "tf_efficientnet_b7_ns": tf_efficientnet_b7_ns,
    }
except ImportError:  # timm >= 1.0
    _DIRECT_BUILDERS = {}


def _make_creator(arch_name: str) -> Callable[..., Any]:
    """Return a callable ``fn(pretrained=False, **kwargs)`` for ``arch_name``."""

    def build(pretrained: bool = False, **kwargs: Any):
        from timm import create_model

        return create_model(arch_name, pretrained=pretrained, **kwargs)

    build.__name__ = arch_name
    return build


def _builder_for(arch_name: str) -> Callable[..., Any]:
    return _DIRECT_BUILDERS.get(arch_name) or _make_creator(arch_name)


# feature dim + construction options for each supported encoder
encoder_params: Dict[str, Dict[str, Any]] = {
    "tf_efficientnet_b3_ns": {"features": 1536, "drop_path_rate": 0.2},
    "tf_efficientnet_b2_ns": {"features": 1408, "drop_path_rate": 0.2},
    "tf_efficientnet_b4_ns": {"features": 1792, "drop_path_rate": 0.5},
    "tf_efficientnet_b5_ns": {"features": 2048, "drop_path_rate": 0.2},
    "tf_efficientnet_b4_ns_03d": {"features": 1792, "drop_path_rate": 0.3},
    "tf_efficientnet_b5_ns_03d": {"features": 2048, "drop_path_rate": 0.3},
    "tf_efficientnet_b5_ns_04d": {"features": 2048, "drop_path_rate": 0.4},
    "tf_efficientnet_b6_ns": {"features": 2304, "drop_path_rate": 0.2},
    "tf_efficientnet_b7_ns": {"features": 2560, "drop_path_rate": 0.2},
    "tf_efficientnet_b6_ns_04d": {"features": 2304, "drop_path_rate": 0.4},
}


def build_backbone(encoder: str, pretrained: bool = False, **overrides: Any):
    """Build a ``timm`` encoder by name.

    Parameters
    ----------
    encoder:
        Key from :data:`encoder_params` (e.g. ``"tf_efficientnet_b7_ns"``).
    pretrained:
        Whether to download ImageNet weights. Defaults to ``False`` because the
        checkpoint we load overwrites every parameter.
    **overrides:
        Extra kwargs forwarded to the backbone constructor (e.g. ``drop_path_rate``).
    """
    if encoder not in encoder_params:
        raise KeyError(
            f"Unknown encoder '{encoder}'. Available: {sorted(encoder_params)}"
        )
    params = encoder_params[encoder]
    kwargs: Dict[str, Any] = {"drop_path_rate": params["drop_path_rate"]}
    kwargs.update(overrides)
    builder = _builder_for(encoder)
    return builder(pretrained=pretrained, **kwargs)


def encoder_feature_dim(encoder: str) -> int:
    """Number of output features produced by ``encoder`` before the head."""
    return int(encoder_params[encoder]["features"])
