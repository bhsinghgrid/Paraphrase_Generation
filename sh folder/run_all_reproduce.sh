#!/usr/bin/env bash
# Sanskrit Diffusion - Master Reproducibility Script
# Performance Benchmark & Task Execution Suite

set -euo pipefail

echo "===================================================="
echo "SANSKRIT DIFFUSION - REPRODUCIBILITY SUITE"
echo "===================================================="

# 1. Environment Health Check
echo "[1/4] Checking environment..."
if [[ ! -d ".venv" ]]; then
    echo "ERROR: .venv not found. Run ./setup_local.sh first."
    exit 1
fi

if [[ ! -f ".env" ]]; then
    echo "ERROR: .env not found. Ensure .env is populated with model paths."
    exit 1
fi

# 2. Hardware Validation (MPS/CPU)
echo "[2/4] Validating hardware acceleration..."
.venv/bin/python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'MPS Available: {torch.backends.mps.is_available()}')"

# 3. Sample Inference Verification
echo "[3/4] Verifying model integrity (Sample Inference)..."
.venv/bin/python inference.py --src "धर्मो रक्षति रक्षितः" --steps 4 --cli

# 4. Trigger Analysis Tasks (Task 1-5)
echo "[4/4] Executing Technical Analysis Suite (T4 Optimized)..."
# We run a subset or the full suite depending on flags.
# Default: Runs the T4 ablation set.
MODEL_ROOT=ablation_results/T4 \
OUT_ROOT=analysis/outputs_reproducibility_test \
bash run_all_tasks.sh

echo "===================================================="
echo "REPRODUCTION SUCCESSFUL: All tasks verified."
echo "Results available in: analysis/outputs_reproducibility_test"
echo "===================================================="
