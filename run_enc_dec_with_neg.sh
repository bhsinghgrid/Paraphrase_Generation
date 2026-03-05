#!/bin/bash
export MODEL_TYPE="d3pm_encoder_decoder"
export INCLUDE_NEG="True"

echo "======================================================"
echo "🚀 STARTING: Encoder-Decoder | Negatives: True"
echo "======================================================"

uv run train.py
uv run evaluate_test.py