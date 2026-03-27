#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

PY_BIN=".venv/bin/python"
if [[ ! -x "$PY_BIN" && -x "../.venv/bin/python" ]]; then
  PY_BIN="../.venv/bin/python"
fi
if [[ ! -x "$PY_BIN" ]]; then
  echo "Virtual env not found. Run: ./setup_local.sh"
  exit 1
fi

"$PY_BIN" - <<'PY'
import os
from huggingface_hub import HfApi

token = os.getenv("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN not found in environment or .env")

repo_id = os.getenv("HF_SPACE_REPO", "bhsinghgrid/devflow")
folder_path = os.getenv("HF_SPACE_FOLDER", "hf_space_repo")

api = HfApi(token=token)
api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="space",
    ignore_patterns=["__pycache__/*", "*.pyc", ".DS_Store"],
)
print(f"Uploaded to https://huggingface.co/spaces/{repo_id}")
PY
