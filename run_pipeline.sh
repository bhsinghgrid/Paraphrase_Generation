#!/bin/bash
# =============================================================================
# run_pipeline.sh
# Full project pipeline: sets up environment, installs dependencies,
# runs training scripts, then testing/inference scripts.
# Logs are saved in the logs/ folder.
# =============================================================================

# -------------------------------
# 1️⃣  Setup Environment
# -------------------------------
echo "🛠 Setting up Python environment..."

# Activate existing virtual environment, or create a new one
if [ -d ".venv" ]; then
    echo "✅ Virtual environment found. Activating..."
    source .venv/bin/activate
else
    echo "⚠️  Virtual environment not found. Creating .venv..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "✅ Virtual environment created and activated."
fi

# Install or upgrade dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed."

# Ensure logs directory exists
mkdir -p logs
echo "📂 Logs directory ready at ./logs"

# -------------------------------
# 2️⃣  Training Scripts
# -------------------------------
TRAIN_SCRIPTS=(
    "bl_cross_train.py"
    "bl_ed_train.py"
    "train_cross.py"
    "train_ed.py"
    "train_edB.py"
)

echo "🚀 Starting training scripts..."

for script in "${TRAIN_SCRIPTS[@]}"; do
    logfile="logs/${script%.py}.log"
    echo "🔹 Running $script → logging to $logfile..."
    python "$script" > "$logfile" 2>&1
    echo "✅ $script completed."
done

# -------------------------------
# 3️⃣  Testing / Inference Scripts
# -------------------------------
TEST_SCRIPTS=(
    "test_cross.py"
    "test_ed.py"
    "main.py"
)

echo "🔹 Starting testing / inference scripts..."

for script in "${TEST_SCRIPTS[@]}"; do
    logfile="logs/${script%.py}.log"
    echo "🔹 Running $script → logging to $logfile..."
    python "$script" > "$logfile" 2>&1
    echo "✅ $script completed."
done

# -------------------------------
# 4️⃣  Pipeline Complete
# -------------------------------
echo "🎉 All tasks completed! You can review logs in the logs/ folder."