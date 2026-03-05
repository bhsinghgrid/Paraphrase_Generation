#!/bin/bash
export MODEL_TYPE="baseline_cross_attention"
export INCLUDE_NEG="False"

echo "======================================================"
echo "🚀 STARTING: Baseline Cross-Attention | Negatives: False"
echo "======================================================"

uv run train.py
uv run evaluate_test.py