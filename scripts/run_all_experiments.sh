#!/usr/bin/env bash
#
# Reproducibility helper: run the full pipeline end to end with one command.
#   1. download pretrained weights (if missing)
#   2. preprocess (extract faces + splits)
#   3. train on FF++
#   4. cross-dataset evaluation (Celeb-DF / DFDC)
#   5. robustness evaluation
#
# Usage:
#   ./scripts/run_all_experiments.sh [RAW_DIR]
#
set -euo pipefail

RAW_DIR=${1:-data/raw/ffpp}
export PYTHONPATH=.

echo "### 1/5  Downloading weights"
if [ ! -f weights/model_v1 ]; then
  ./scripts/download_weights.sh
else
  echo "weights/model_v1 already present, skipping."
fi

echo "### 2/5  Preprocessing"
./scripts/run_preprocess.sh "${RAW_DIR}"

echo "### 3/5  Training on FF++"
./scripts/train.sh configs/train_ffpp.yaml

echo "### 4/5  Cross-dataset evaluation"
python -m src.evaluation.cross_dataset \
  --checkpoint results/checkpoints/best.pt \
  --datasets celebdf dfdc \
  --out results/tables/cross_dataset.json

echo "### 5/5  Robustness evaluation"
python -m src.evaluation.robustness \
  --checkpoint results/checkpoints/best.pt \
  --dataset celebdf \
  --out results/tables/robustness.json

echo "### All experiments finished."
