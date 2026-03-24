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

INPUT_TEXT="${1:-dharmo rakṣati rakṣitaḥ}"

# Run tasks 1,2,3,5 for each ablation checkpoint
for T in 4 8 16 32 64; do
  CKPT="ablation_results/T${T}/best_model.pt"
  OUT="analysis/outputs_ablation/T${T}"
  mkdir -p "$OUT"
  if [[ ! -f "$CKPT" ]]; then
    echo "Skipping T${T}: checkpoint not found."
    continue
  fi

  "$PY_BIN" analysis/run_analysis.py --task 1 --checkpoint "$CKPT" --output_dir "$OUT"
  "$PY_BIN" analysis/run_analysis.py --task 2 --checkpoint "$CKPT" --output_dir "$OUT" --input "$INPUT_TEXT"
  "$PY_BIN" analysis/run_analysis.py --task 3 --task3_samples 120 --checkpoint "$CKPT" --output_dir "$OUT"
  "$PY_BIN" analysis/run_analysis.py --task 5 --checkpoint "$CKPT" --output_dir "$OUT"
done

# Task 4 uses shared cross-model ablation summary
mkdir -p analysis/outputs_ablation/task4_global
if [[ -f analysis/outputs/task4_3d.png ]]; then
  cp -f analysis/outputs/task4_* analysis/outputs_ablation/task4_global/
  for T in 4 8 16 32 64; do
    mkdir -p "analysis/outputs_ablation/T${T}"
    cp -f analysis/outputs_ablation/task4_global/task4_* "analysis/outputs_ablation/T${T}/" || true
  done
fi

echo "Done. See analysis/outputs_ablation/T*/"
