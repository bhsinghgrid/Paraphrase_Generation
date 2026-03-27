#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts_common.sh"

PY_BIN="$(find_python_with_torch || true)"
if [[ -z "$PY_BIN" ]]; then
  echo "Virtual env not found. Run: ./setup_local.sh"
  exit 1
fi

PORT="${GRADIO_SERVER_PORT:-${PORT:-7860}}"
"$PY_BIN" app.py --port "$PORT"
