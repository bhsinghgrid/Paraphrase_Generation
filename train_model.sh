#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./train_model.sh [model_type] [include_neg]
# Example:
#   ./train_model.sh d3pm_cross_attention False

MODEL_TYPE="${1:-d3pm_cross_attention}"
INCLUDE_NEG="${2:-False}"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts_common.sh"

PY_BIN="$(find_python_with_torch || true)"
if [[ -z "$PY_BIN" ]]; then
  echo "Virtual env not found. Run: ./setup_local.sh"
  exit 1
fi

export MODEL_TYPE
export INCLUDE_NEG

echo "======================================================"
echo "Training model"
echo "MODEL_TYPE=${MODEL_TYPE}"
echo "INCLUDE_NEG=${INCLUDE_NEG}"
echo "======================================================"

"$PY_BIN" train.py

echo "Training completed."
