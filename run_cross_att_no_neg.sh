#!/bin/bash
export MODEL_TYPE="d3pm_cross_attention"
export INCLUDE_NEG="False"

echo "======================================================"
echo "🚀 STARTING: Cross-Attention | Negatives: False"
echo "======================================================"

uv run train.py
uv run evaluate_test.py