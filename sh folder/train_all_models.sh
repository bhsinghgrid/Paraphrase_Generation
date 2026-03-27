#!/usr/bin/env bash
set -euo pipefail

# Runs the 4 standard training combinations one-by-one:
# 1) d3pm_cross_attention + no negatives
# 2) d3pm_cross_attention + negatives
# 3) d3pm_encoder_decoder + no negatives
# 4) d3pm_encoder_decoder + negatives

chmod +x ./train_model.sh

echo "========================================"
echo "Starting full training matrix"
echo "========================================"

./train_model.sh d3pm_cross_attention False
./train_model.sh d3pm_cross_attention True
./train_model.sh d3pm_encoder_decoder False
./train_model.sh d3pm_encoder_decoder True

echo "========================================"
echo "All training runs completed."
echo "========================================"
