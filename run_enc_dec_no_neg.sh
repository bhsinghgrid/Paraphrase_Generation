#!/bin/bash
export MODEL_TYPE="d3pm_encoder_decoder"
export INCLUDE_NEG="False"

echo "======================================================"
echo "🚀 STARTING: Encoder-Decoder | Negatives: False"
echo "======================================================"

uv run train.py
uv run evaluate_test.py