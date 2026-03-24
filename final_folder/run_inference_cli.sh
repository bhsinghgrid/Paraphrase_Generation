#!/usr/bin/env bash
set -euo pipefail

TEXT="${1:-dharmo rakṣati rakṣitaḥ}"
CKPT="${2:-ablation_results/T4/best_model.pt}"

PY_BIN=".venv/bin/python"
if [[ ! -x "$PY_BIN" && -x "../.venv/bin/python" ]]; then
  PY_BIN="../.venv/bin/python"
fi
if [[ ! -x "$PY_BIN" ]]; then
  echo "Virtual env not found. Run: ./setup_local.sh"
  exit 1
fi

"$PY_BIN" inference.py --mode demo --checkpoint "$CKPT" --text "$TEXT"
