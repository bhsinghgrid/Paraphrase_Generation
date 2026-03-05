#!/bin/bash
export MODEL_TYPE="baseline_encoder_decoder"
export INCLUDE_NEG="True"

echo "======================================================"
echo "🚀 STARTING: Encoder-Decoder | Negatives: False"
echo "======================================================"

uv run train.py
uv run evaluate_test.py