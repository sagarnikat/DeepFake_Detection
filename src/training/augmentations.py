"""Albumentations pipelines for face-crop training/eval.

Returns callables compatible with :class:`src.datasets.base.FaceDataset`
(``transform(image=arr)["image"]``). Images are converted to float CHW tensors
*without* ImageNet normalization here (normalization is applied by the dataset
when ``transform is None``); to keep a single source of truth this module exposes
``get_train_transform`` / ``get_val_transform`` that include ``Normalize`` and
``ToTensorV2``.
"""

from __future__ import annotations

from typing import Any


def _require_albumentations():
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        return A, ToTensorV2
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "albumentations (with albumentations.pytorch) is required for augmentations"
        ) from exc


def get_train_transform(input_size: int = 380, **kwargs: Any):
    """Training augmentation pipeline."""
    A, ToTensorV2 = _require_albumentations()
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=input_size),
            A.RandomCrop(height=input_size, width=input_size),
            A.HorizontalFlip(p=0.5),
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.GaussNoise(p=1.0),
                    A.ISONoise(p=1.0),
                ],
                p=0.3,
            ),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_val_transform(input_size: int = 380, **kwargs: Any):
    """Deterministic eval/val pipeline."""
    A, ToTensorV2 = _require_albumentations()
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=input_size),
            A.CenterCrop(height=input_size, width=input_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
