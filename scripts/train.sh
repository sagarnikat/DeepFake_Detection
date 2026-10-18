#!/usr/bin/env bash
#
# Train on FaceForensics++.
#
# Usage:
#   ./scripts/train.sh [CONFIG]
#
set -euo pipefail

CONFIG=${1:-configs/train_ffpp.yaml}
export PYTHONPATH=.

python -m src.training.train --config "${CONFIG}" --base-config configs/base.yaml
