#!/bin/bash
export MODEL_TYPE="baseline_encoder_decoder"
export INCLUDE_NEG="False"

echo "======================================================"
echo "🚀 STARTING: Baseline Encoder-Decoder | Negatives: False"
echo "======================================================"

uv run train.py
uv run evaluate_test.py