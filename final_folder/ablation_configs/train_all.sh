#!/bin/bash
set -euo pipefail
# Run this script to train all ablation models

MODEL_TYPE=${MODEL_TYPE:-d3pm_cross_attention}
INCLUDE_NEG=${INCLUDE_NEG:-False}
TRAIN_DEVICE=${TRAIN_DEVICE:-mps}

echo '=== Training T=4 ==='
mkdir -p ablation_results/T4
MODEL_TYPE="$MODEL_TYPE" INCLUDE_NEG="$INCLUDE_NEG" TRAIN_DEVICE="$TRAIN_DEVICE" DIFFUSION_STEPS=4 INFERENCE_NUM_STEPS=4 TRAIN_OUTPUT_DIR="ablation_results/T4" python train.py

echo '=== Training T=8 ==='
mkdir -p ablation_results/T8
MODEL_TYPE="$MODEL_TYPE" INCLUDE_NEG="$INCLUDE_NEG" TRAIN_DEVICE="$TRAIN_DEVICE" DIFFUSION_STEPS=8 INFERENCE_NUM_STEPS=8 TRAIN_OUTPUT_DIR="ablation_results/T8" python train.py

echo '=== Training T=16 ==='
mkdir -p ablation_results/T16
MODEL_TYPE="$MODEL_TYPE" INCLUDE_NEG="$INCLUDE_NEG" TRAIN_DEVICE="$TRAIN_DEVICE" DIFFUSION_STEPS=16 INFERENCE_NUM_STEPS=16 TRAIN_OUTPUT_DIR="ablation_results/T16" python train.py

echo '=== Training T=32 ==='
mkdir -p ablation_results/T32
MODEL_TYPE="$MODEL_TYPE" INCLUDE_NEG="$INCLUDE_NEG" TRAIN_DEVICE="$TRAIN_DEVICE" DIFFUSION_STEPS=32 INFERENCE_NUM_STEPS=32 TRAIN_OUTPUT_DIR="ablation_results/T32" python train.py

echo '=== Training T=64 ==='
mkdir -p ablation_results/T64
MODEL_TYPE="$MODEL_TYPE" INCLUDE_NEG="$INCLUDE_NEG" TRAIN_DEVICE="$TRAIN_DEVICE" DIFFUSION_STEPS=64 INFERENCE_NUM_STEPS=64 TRAIN_OUTPUT_DIR="ablation_results/T64" python train.py

