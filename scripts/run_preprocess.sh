#!/usr/bin/env bash
#
# Preprocess: extract + crop faces from raw videos, then build video-level splits.
#
# Usage:
#   ./scripts/run_preprocess.sh [RAW_DIR] [FACES_DIR]
#
set -euo pipefail

RAW_DIR=${1:-data/raw/ffpp}
FACES_DIR=${2:-data/faces/ffpp}
NUM_FRAMES=${NUM_FRAMES:-32}

export PYTHONPATH=.

echo "==> Extracting + cropping faces from ${RAW_DIR}"
python -m src.preprocessing.extract_faces \
  --input-dir "${RAW_DIR}" \
  --output-dir "${FACES_DIR}" \
  --num-frames "${NUM_FRAMES}"

echo "==> Building video-level train/val/test splits"
python -m src.preprocessing.make_splits \
  --faces-dir "${FACES_DIR}" \
  --out-dir data/splits

echo "==> Preprocessing done."
