#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts_common.sh"

INPUT_TEXT="${1:-dharmo rakṣati rakṣitaḥ}"
MODEL_ROOT="${MODEL_ROOT:-ablation_results/encoder_decoder}"
OUT_ROOT="${OUT_ROOT:-analysis/outputs_all_models_20260325/encoder_decoder}"
mkdir -p "$OUT_ROOT"

if [[ ! -d "$MODEL_ROOT" ]]; then
  echo "Model root not found: $MODEL_ROOT"
  exit 1
fi

PY_BIN="$(find_python_with_torch || true)"
if [[ -z "$PY_BIN" ]]; then
  echo "No Python environment with torch found. Run setup_local.sh first."
  exit 1
fi

declare -a CKPTS=()
while IFS= read -r ckpt; do
  CKPTS+=("$ckpt")
done < <(find "$MODEL_ROOT" -maxdepth 2 -type f -name "best_model.pt" | sort -V)

if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "No checkpoints found under: $MODEL_ROOT"
  exit 1
fi

declare -a FOUND_TS=()

for CKPT in "${CKPTS[@]}"; do
  T_DIR="$(basename "$(dirname "$CKPT")")"
  T_LABEL="${T_DIR#T}"
  if [[ "$T_DIR" == "$T_LABEL" ]]; then
    echo "Skipping checkpoint with unexpected step folder name: $CKPT"
    continue
  fi

  FOUND_TS+=("$T_LABEL")
  OUT_DIR="${OUT_ROOT}/${T_DIR}"
  mkdir -p "$OUT_DIR"

  echo "============================================================"
  echo "Running Task 1,2,3,5 for ${T_DIR}"
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

# Task 4 is cross-model ablation analysis; run once using the first available
# checkpoint inside the selected model root, then copy shared artifacts.
TASK4_DIR="${OUT_ROOT}/task4_global"
mkdir -p "$TASK4_DIR"
TASK4_CKPT="${CKPTS[0]}"
"$PY_BIN" analysis/run_analysis.py \
  --task 4 \
  --phase analyze \
  --checkpoint "$TASK4_CKPT" \
  --output_dir "$TASK4_DIR"

for T in "${FOUND_TS[@]}"; do
  OUT_DIR="${OUT_ROOT}/T${T}"
  mkdir -p "$OUT_DIR"
  cp -f "${TASK4_DIR}"/task4_* "$OUT_DIR"/ 2>/dev/null || true
done

echo "Done. Outputs are under ${OUT_ROOT}/T*"
