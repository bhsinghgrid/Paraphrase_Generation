#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
#  echo "Usage: $0 <hf_username> <model_repo_name> [space_repo_name]"
  echo "Usage: $0 <model_repo_name>"
  echo "Example: $0 <model_repo_name>"
  exit 1
fi

HF_USER="$1"
MODEL_REPO="$2"
SPACE_REPO="${3:-${MODEL_REPO}-space}"

echo "Logging in to Hugging Face (if not already)..."
.venv/bin/hf auth whoami >/dev/null 2>&1 || .venv/bin/hf auth login

echo "Creating model repo: ${HF_USER}/${MODEL_REPO}"
.venv/bin/hf repo create "${HF_USER}/${MODEL_REPO}" --type model --yes || true

echo "Creating space repo: ${HF_USER}/${SPACE_REPO}"
.venv/bin/hf repo create "${HF_USER}/${SPACE_REPO}" --type space --space-sdk gradio --yes || true

push_repo () {
  local folder="$1"
  local remote_url="$2"
  pushd "$folder" >/dev/null
  if [[ ! -d .git ]]; then
    git init
    git lfs install
    git remote add origin "$remote_url"
  else
    git remote remove origin >/dev/null 2>&1 || true
    git remote add origin "$remote_url"
  fi
  git add .
  git commit -m "Initial upload from local project" || true
  git branch -M main
  git push -u origin main
  popd >/dev/null
}

echo "Pushing model repo..."
push_repo "hf_model_repo" "https://huggingface.co/${HF_USER}/${MODEL_REPO}"

echo "Pushing space repo..."
push_repo "hf_space_repo" "https://huggingface.co/spaces/${HF_USER}/${SPACE_REPO}"

echo "Done."
echo "Set Space variable HF_CHECKPOINT_REPO=${HF_USER}/${MODEL_REPO}"
