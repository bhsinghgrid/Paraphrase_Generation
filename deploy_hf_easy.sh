#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: ./deploy_hf_easy.sh <hf_username> <project_name>"
  echo "Example: ./deploy_hf_easy.sh <model_repo_name>"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts_common.sh"

HF_USER="$1"
PROJECT="$2"
MODEL_REPO="${HF_USER}/${PROJECT}"
SPACE_REPO="${HF_USER}/${PROJECT}"

PY_BIN="$(find_python_with_torch || true)"
if [[ -z "$PY_BIN" ]]; then
  echo "Virtual env not found. Run: ./setup_local.sh"
  exit 1
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Install huggingface-hub CLI first: pip install -U huggingface_hub"
  exit 1
fi

echo "Step 1/3: Hugging Face login check"
if [[ -n "${HF_TOKEN:-}" ]]; then
  huggingface-cli login --token "${HF_TOKEN}" >/dev/null 2>&1 || true
fi
huggingface-cli whoami >/dev/null 2>&1 || huggingface-cli login

echo "Step 2/3: Upload model files to ${MODEL_REPO}"
"$PY_BIN" upload_hf_model.py \
  --repo-id "${MODEL_REPO}" \
  --checkpoint "${HF_MODEL_CHECKPOINT:-ablation_results/T4/best_model.pt}"

echo "Step 3/3: Upload Space files to ${SPACE_REPO}"
"$PY_BIN" upload_hf_space.py \
  --repo-id "${SPACE_REPO}"

echo "Deployment complete."
echo "Model: https://huggingface.co/${MODEL_REPO}"
echo "Space: https://huggingface.co/spaces/${SPACE_REPO}"
