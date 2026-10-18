"""Training loop, losses, augmentations, entry point."""

from src.training.augmentations import get_train_transform, get_val_transform  # noqa: F401
from src.training.losses import build_loss  # noqa: F401
from src.training.trainer import Trainer  # noqa: F401
