#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

INPUT_TEXT="${1:-dharmo rakṣati rakṣitaḥ}"
OUT_ROOT="analysis/outputs_ablation"
mkdir -p "$OUT_ROOT"
mkdir -p ".hf_cache/datasets" ".mplconfig"
export HF_HOME="$ROOT_DIR/.hf_cache"
export HF_DATASETS_CACHE="$ROOT_DIR/.hf_cache/datasets"
export MPLCONFIGDIR="$ROOT_DIR/.mplconfig"

# Prime local HF dataset cache from user cache (read-only source) to avoid
# sandbox lock permission issues under /Users/.../.cache.
SRC_CACHE="/Users/bhsingh/.cache/huggingface/datasets/paws___sanskrit-verses-gretil"
DST_CACHE="$HF_DATASETS_CACHE/paws___sanskrit-verses-gretil"
if [[ -d "$SRC_CACHE" && ! -d "$DST_CACHE" ]]; then
  cp -R "$SRC_CACHE" "$DST_CACHE"
fi

PY_BIN=".venv/bin/python"
if [[ ! -x "$PY_BIN" ]]; then
  PY_BIN="python3"
fi

for T in 4 8 16 32 64; do
  CKPT="ablation_results/T${T}/best_model.pt"
  OUT_DIR="${OUT_ROOT}/T${T}"
  mkdir -p "$OUT_DIR"

  if [[ ! -f "$CKPT" ]]; then
    echo "Skipping T${T}: checkpoint not found at ${CKPT}"
    continue
  fi

  echo "============================================================"
  echo "Running Task 1,2,3,5 for T${T}"
  echo "Checkpoint: ${CKPT}"
  echo "Output    : ${OUT_DIR}"
  echo "============================================================"

  "$PY_BIN" analysis/run_analysis.py \
    --task 1 \
    --checkpoint "$CKPT" \
    --output_dir "$OUT_DIR"

  "$PY_BIN" analysis/run_analysis.py \
    --task 2 \
    --checkpoint "$CKPT" \
    --output_dir "$OUT_DIR" \
    --input "$INPUT_TEXT"

  "$PY_BIN" analysis/run_analysis.py \
    --task 3 \
    --checkpoint "$CKPT" \
    --output_dir "$OUT_DIR"

  "$PY_BIN" analysis/run_analysis.py \
    --task 5 \
    --checkpoint "$CKPT" \
    --output_dir "$OUT_DIR"
done

# Task 4 is cross-model ablation analysis; run once and share artifacts.
TASK4_DIR="${OUT_ROOT}/task4_global"
mkdir -p "$TASK4_DIR"
"$PY_BIN" analysis/run_analysis.py \
  --task 4 \
  --phase analyze \
  --checkpoint "ablation_results/T4/best_model.pt" \
  --output_dir "$TASK4_DIR"

for T in 4 8 16 32 64; do
  OUT_DIR="${OUT_ROOT}/T${T}"
  mkdir -p "$OUT_DIR"
  cp -f "${TASK4_DIR}"/task4_* "$OUT_DIR"/ 2>/dev/null || true
done

echo "Done. Outputs are under ${OUT_ROOT}/T{4,8,16,32,64}"
