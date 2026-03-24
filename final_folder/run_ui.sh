#!/usr/bin/env bash
set -euo pipefail

PY_BIN=".venv/bin/python"
if [[ ! -x "$PY_BIN" && -x "../.venv/bin/python" ]]; then
  PY_BIN="../.venv/bin/python"
fi
if [[ ! -x "$PY_BIN" ]]; then
  echo "Virtual env not found. Run: ./setup_local.sh"
  exit 1
fi

export HF_HOME="${PWD}/.hf_cache"
export HF_DATASETS_CACHE="${PWD}/.hf_cache/datasets"
export HF_HUB_CACHE="${PWD}/.hf_cache/hub"
export MPLCONFIGDIR="${PWD}/.mplconfig"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$MPLCONFIGDIR"

"$PY_BIN" app.py
