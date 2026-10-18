"""Cross-dataset evaluation: train on FF++ -> test on Celeb-DF / DFDC.

This module wires the *datasets* to the shared evaluator. It's kept independent
from training so a checkpoint trained on FF++ can be transferred without
re-training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.datasets import build_celebdf_dataset, build_dfdc_dataset, build_ffpp_dataset
from src.evaluation.evaluate import evaluate_loader
from src.models.registry import build_model
from src.training.augmentations import get_val_transform
from src.utils.io import save_json
from src.utils.logging import get_logger

logger = get_logger("dfdetect.cross_dataset")

DATASET_BUILDERS: Dict[str, Callable] = {
    "ffpp": build_ffpp_dataset,
    "celebdf": build_celebdf_dataset,
    "dfdc": build_dfdc_dataset,
}


def build_test_dataset(name: str, transform, **kwargs):
    """Build a test dataset by name."""
    if name not in DATASET_BUILDERS:
        raise KeyError(f"Unknown dataset '{name}'. Options: {sorted(DATASET_BUILDERS)}")
    return DATASET_BUILDERS[name](transform=transform, **kwargs)


def cross_dataset_report(
    checkpoint: str | Path,
    datasets: tuple[str, ...] = ("celebdf", "dfdc"),
    encoder: str = "tf_efficientnet_b7_ns",
    model_name: str = "resnet_gap",
    input_size: int = 380,
    batch_size: int = 32,
    num_workers: int = 4,
    out_path: str | Path = "results/tables/cross_dataset.json",
    device: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate ``checkpoint`` on each cross-dataset test set."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(name=model_name, encoder=encoder)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("state_dict", state), strict=False)
    model.to(device).eval()

    transform = get_val_transform(input_size)
    report: Dict[str, Dict[str, float]] = {}
    for ds_name in datasets:
        try:
            ds = build_test_dataset(ds_name, transform=transform, input_size=input_size)
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", ds_name, exc)
            continue
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        report[ds_name] = evaluate_loader(model, loader, device=device)
        logger.info("%s -> %s", ds_name, report[ds_name])

    save_json(report, out_path)
    return report


def main() -> None:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--datasets", nargs="+", default=["celebdf", "dfdc"])
    parser.add_argument("--encoder", default="tf_efficientnet_b7_ns")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default="results/tables/cross_dataset.json")
    args = parser.parse_args()
    cross_dataset_report(
        args.checkpoint,
        datasets=tuple(args.datasets),
        encoder=args.encoder,
        batch_size=args.batch_size,
        out_path=args.out,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
