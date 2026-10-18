"""Model definitions and registry."""

from src.models.backbone import build_backbone, encoder_params  # noqa: F401
from src.models.classifier import DeepFakeClassifier  # noqa: F401
from src.models.registry import MODEL_REGISTRY, build_model, register_model  # noqa: F401
