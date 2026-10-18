"""Training entry point.

Example
-------
python -m src.training.train --config configs/train_ffpp.yaml
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from src.datasets.ffpp import build_ffpp_dataset
from src.models.registry import build_model
from src.training.augmentations import get_train_transform, get_val_transform
from src.training.losses import build_loss
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.logging import get_logger, setup_logging
from src.utils.seed import set_seed, worker_init_fn

logger = get_logger("dfdetect.train")


def build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    name = cfg.get("name", "adam").lower()
    lr = float(cfg.get("lr", 3e-4))
    wd = float(cfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    raise KeyError(f"Unknown optimizer '{name}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a deepfake classifier")
    parser.add_argument("--config", default="configs/train_ffpp.yaml")
    parser.add_argument("--base-config", default="configs/base.yaml")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config, default_path=args.base_config)
    set_seed(int(cfg.get("seed", 42)))

    input_size = int(cfg.data.get("image_size", 380))
    train_ds = build_ffpp_dataset("train", transform=get_train_transform(input_size))
    val_ds = build_ffpp_dataset("val", transform=get_val_transform(input_size))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.data.get("num_workers", 4)),
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.eval.get("batch_size", cfg.train.batch_size)),
        shuffle=False,
        num_workers=int(cfg.data.get("num_workers", 4)),
        pin_memory=True,
    )

    model = build_model(
        name=cfg.model.get("name", "resnet_gap"),
        encoder=cfg.model.get("encoder", "tf_efficientnet_b7_ns"),
        dropout_rate=float(cfg.model.get("dropout_rate", 0.0)),
    )
    loss_fn = build_loss(cfg.train.get("loss", "bce"), fake_weight=float(cfg.train.get("fake_weight", 1.0)))
    optimizer = build_optimizer(model, cfg.train.optimizer)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        checkpoint_dir=cfg.paths.get("checkpoints", "results/checkpoints"),
        amp=bool(cfg.train.get("amp", True)),
    )
    trainer.fit(
        train_loader,
        val_loader,
        epochs=int(cfg.train.epochs),
        patience=int(cfg.train.get("early_stopping_patience", 5)),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
