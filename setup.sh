#!/usr/bin/env bash
#
# One-shot setup for a machine WITHOUT an NVIDIA GPU.
#
# Creates ./.venv, installs the dependencies (CPU-only PyTorch so the ~3 GB
# CUDA stack is not downloaded), and fetches the pretrained model weights.
#
#   ./setup.sh
#
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.10}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: '$PYTHON_BIN' not found." >&2
  echo "Install Python 3.10, e.g.:  uv python install 3.10" >&2
  exit 1
fi

echo "==> Creating virtualenv (.venv) with $PYTHON_BIN"
"$PYTHON_BIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

echo "==> Installing CPU-only PyTorch"
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

echo "==> Installing the remaining requirements"
pip install -r requirements.txt

echo "==> Downloading pretrained weights"
chmod +x scripts/*.sh
./scripts/download_weights.sh

echo
echo "Setup complete."
echo "Start the backend with:"
echo "    source .venv/bin/activate && python -m app.server"
