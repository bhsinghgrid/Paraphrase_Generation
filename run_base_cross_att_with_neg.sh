#!/bin/bash
export MODEL_TYPE="baseline_cross_attention"
export INCLUDE_NEG="True"

echo "======================================================"
echo "🚀 STARTING: Baseline Cross-Attention | Negatives: True"
echo "======================================================"

uv run train.py
uv run evaluate_test.py