#!/usr/bin/env bash
#
# Downloads the pretrained DFDC DeepFake weights (Selim Seferbekov's ensemble)
# and exposes one of them as `weights/model_v1`, which is the file the model
# loader (src/inference/predict_video.py) loads by default.
#
# Usage (from the repository root):
#   ./scripts/download_weights.sh
#
# Then start the backend with:
#   python -m app.server
#
# To use the full 7-model ensemble instead of the single default model, set:
#   export DFDETECT_WEIGHTS="final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36,final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19,final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29,final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31,final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_37,final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40,final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23"

set -euo pipefail

tag=0.0.1
base_url="https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/${tag}"

# Default single model, symlinked as `model_v1`
default_model="final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36"

models=(
  final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36
  final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19
  final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29
  final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31
  final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_37
  final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40
  final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23
)

mkdir -p weights

for name in "${models[@]}"; do
  if [ -f "weights/${name}" ]; then
    echo "Already present: weights/${name}"
    continue
  fi
  echo "Downloading ${name} ..."
  wget -O "weights/${name}" "${base_url}/${name}"
done

# The default model is `weights/model_v1`.
if [ ! -f "weights/model_v1" ]; then
  echo "Linking weights/model_v1 -> ${default_model}"
  ln -sf "${default_model}" "weights/model_v1"
fi

echo "Done. Weights are in ./weights"
