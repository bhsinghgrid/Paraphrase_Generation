#!/bin/bash
export MODEL_TYPE="d3pm_cross_attention"
export INCLUDE_NEG="True"

echo "======================================================"
echo "🚀 STARTING: Cross-Attention | Negatives: True"
echo "======================================================"

uv run train.py
uv run evaluate_test.py