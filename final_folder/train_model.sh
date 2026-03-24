#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./train_model.sh [model_type] [include_neg]
# Example:
#   ./train_model.sh d3pm_cross_attention False

MODEL_TYPE="${1:-d3pm_cross_attention}"
INCLUDE_NEG="${2:-False}"

PY_BIN=".venv/bin/python"
if [[ ! -x "$PY_BIN" && -x "../.venv/bin/python" ]]; then
  PY_BIN="../.venv/bin/python"
fi
if [[ ! -x "$PY_BIN" ]]; then
  echo "Virtual env not found. Run: ./setup_local.sh"
  exit 1
fi

export MODEL_TYPE
export INCLUDE_NEG
export HF_HOME="${PWD}/.hf_cache"
export HF_DATASETS_CACHE="${PWD}/.hf_cache/datasets"
export HF_HUB_CACHE="${PWD}/.hf_cache/hub"
export MPLCONFIGDIR="${PWD}/.mplconfig"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$MPLCONFIGDIR"

echo "======================================================"
echo "Training model"
echo "MODEL_TYPE=${MODEL_TYPE}"
echo "INCLUDE_NEG=${INCLUDE_NEG}"
echo "======================================================"

"$PY_BIN" train.py

echo "Training completed."
