#!/bin/bash

set -e  # stop on any error

# 0️⃣ Timestamp & Result Folder
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="json/run_$TIMESTAMP"
LOG_DIR="$RESULT_DIR/logs"

mkdir -p "$LOG_DIR"
echo "📂 Results directory created at $RESULT_DIR"

# 1️⃣ GPU Detection
if command -v nvidia-smi &> /dev/null
then
    GPU_AVAILABLE=true
    echo "🖥 GPU detected. Training scripts will use GPU."
else
    GPU_AVAILABLE=false
    echo "⚠️ No GPU detected. Training will run on CPU."
fi

export USE_GPU=$GPU_AVAILABLE

# 2️⃣ Activate Virtual Environment
echo "🛠 Setting up Python virtual environment..."
if [ -d ".venv" ]; then
    echo "✅ Activating existing .venv..."
    source .venv/bin/activate
else
    echo "⚠️ Creating new .venv..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "✅ Virtual environment created and activated."
fi

pip install --upgrade pip > /dev/null
pip install -r requirements.txt > /dev/null
echo "📦 Dependencies installed."

# 3️⃣ EXCLUDED NEGATIVE (Baseline)
echo "=================================================="
echo "🔥 Running EXCLUDED NEGATIVE scripts (Baseline)"
echo "=================================================="

BASELINE_SCRIPTS=(
    "NBaseline/train_cross.py"
    "NBaseline/train_ed.py"
)

for script in "${BASELINE_SCRIPTS[@]}"; do
    name=$(basename "$script" .py)
    logfile="$LOG_DIR/${name}.log"
    echo "▶ Running $script → logging to $logfile"
    python "$script" --result_dir "$RESULT_DIR" > "$logfile" 2>&1
    echo "✅ $script completed."
done

# 4️⃣ INCLUDED NEGATIVE
echo "=================================================="
echo "🔥 Running INCLUDED NEGATIVE scripts"
echo "=================================================="

INCLUDED_SCRIPTS=(
    "train_cross.py"
    "train_ed.py"
    "train_edB.py"
)

for script in "${INCLUDED_SCRIPTS[@]}"; do
    name=$(basename "$script" .py)
    logfile="$LOG_DIR/${name}.log"
    echo "▶ Running $script → logging to $logfile"
    python "$script" --result_dir "$RESULT_DIR" > "$logfile" 2>&1
    echo "✅ $script completed."
done

# 5️⃣ Testing / Evaluation

echo "=================================================="
echo "🧪 Running Test Scripts & Evaluations"
echo "=================================================="

TEST_SCRIPTS=(
    "test/test_cross.py"
    "test/test_ed.py"
)

for script in "${TEST_SCRIPTS[@]}"; do
    name=$(basename "$script" .py)
    logfile="$LOG_DIR/${name}.log"
    echo "▶ Running $script → logging to $logfile"
    python "$script" --result_dir "$RESULT_DIR" > "$logfile" 2>&1
    echo "✅ $script completed."
done

# -------------------------------
# 6️⃣ Compute BLEU / BERTScore
# -------------------------------
echo "=================================================="
echo "📊 Computing BLEU and BERTScore metrics"
echo "=================================================="

METRIC_FILE="$RESULT_DIR/metrics.json"
python - <<END
import json, glob
from evaluation import compute_bleu, compute_bertscore

results = {}
for f in glob.glob("$RESULT_DIR/logs/*.log"):
    name = f.split("/")[-1].replace(".log", "")
    results[name] = {}
    results[name]["BLEU"] = compute_bleu(f)
    results[name]["BERTScore"] = compute_bertscore(f)

with open("$METRIC_FILE", "w") as out:
    json.dump(results, out, indent=2)

print(f"✅ Metrics saved to {METRIC_FILE}")
END

# 7️⃣ Save Run Metadata
echo "📝 Saving run metadata"

cat <<EOF > "$RESULT_DIR/run_details.json"
{
  "timestamp": "$TIMESTAMP",
  "gpu_available": "$GPU_AVAILABLE",
  "baseline_scripts": ["NBaseline/train_cross.py", "NBaseline/train_ed.py"],
  "included_negative_scripts": ["train_cross.py", "train_ed.py", "train_edB.py"],
  "test_scripts": ["test/test_cross.py", "test/test_ed.py"],
  "logs_directory": "$LOG_DIR",
  "metrics_file": "$METRIC_FILE"
}
EOF

echo "✅ Run metadata saved."

# 8️⃣ Launch UI
echo "=================================================="
echo "🌐 Launching Inference UI"
echo "=================================================="

python inference/app.py --result_dir "$RESULT_DIR"

echo "🎉 Full Pipeline Completed Successfully!"