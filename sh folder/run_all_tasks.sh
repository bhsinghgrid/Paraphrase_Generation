#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts_common.sh"

PY_BIN="$(find_python_with_torch || true)"
if [[ -z "$PY_BIN" ]]; then
  echo "No Python environment with torch found. Run: ./setup_local.sh"
  exit 1
fi

INPUT_TEXT="${1:-dharmo rakṣati rakṣitaḥ}"
MODEL_ROOT="${MODEL_ROOT:-ablation_results}"
OUT_ROOT="${OUT_ROOT:-analysis/outputs_ablation}"

if [[ ! -d "$MODEL_ROOT" ]]; then
  echo "Model root not found: $MODEL_ROOT"
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

# Run tasks 1,2,3,5 for each discovered ablation checkpoint
for CKPT in "${CKPTS[@]}"; do
  T_DIR="$(basename "$(dirname "$CKPT")")"
  T_LABEL="${T_DIR#T}"
  if [[ "$T_DIR" == "$T_LABEL" ]]; then
    echo "Skipping checkpoint with unexpected step folder name: $CKPT"
    continue
  fi

  FOUND_TS+=("$T_LABEL")
  OUT="${OUT_ROOT}/${T_DIR}"
  mkdir -p "$OUT"

  "$PY_BIN" analysis/run_analysis.py --task 1 --checkpoint "$CKPT" --output_dir "$OUT"
  "$PY_BIN" analysis/run_analysis.py --task 2 --checkpoint "$CKPT" --output_dir "$OUT" --input "$INPUT_TEXT"
  "$PY_BIN" analysis/run_analysis.py --task 3 --task3_samples 120 --checkpoint "$CKPT" --output_dir "$OUT"
  "$PY_BIN" analysis/run_analysis.py --task 5 --checkpoint "$CKPT" --output_dir "$OUT"
done

# Task 4 uses shared cross-model ablation summary
TASK4_DIR="${OUT_ROOT}/task4_global"
mkdir -p "$TASK4_DIR"
"$PY_BIN" analysis/run_analysis.py \
  --task 4 \
  --phase analyze \
  --checkpoint "${CKPTS[0]}" \
  --output_dir "$TASK4_DIR"

for T in "${FOUND_TS[@]}"; do
  mkdir -p "${OUT_ROOT}/T${T}"
  cp -f "${TASK4_DIR}"/task4_* "${OUT_ROOT}/T${T}/" || true
done

echo "Done. See ${OUT_ROOT}/T*/"
