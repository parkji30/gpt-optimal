#!/bin/bash
# Benchmark script for Python vs Rust GPT-2 training
# Runs both pipelines and stores timing + evaluation metrics

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$RESULTS_DIR/benchmark_$TIMESTAMP.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create results directory
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  GPT-2 Training Benchmark: Python vs Rust  ${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""

# Function to extract metrics from training output
extract_metrics() {
    local output="$1"
    local train_loss=$(echo "$output" | grep -E "^Final:" | sed -E 's/.*train loss ([0-9.]+).*/\1/')
    local val_loss=$(echo "$output" | grep -E "^Final:" | sed -E 's/.*val loss ([0-9.]+).*/\1/')
    local training_time=$(echo "$output" | grep -E "Total training time:" | sed -E 's/.*: ([0-9.]+)s.*/\1/')
    local n_params=$(echo "$output" | grep -E "Model parameters:" | sed -E 's/.*: ([0-9,]+).*/\1/' | tr -d ',')

    echo "$train_loss|$val_loss|$training_time|$n_params"
}

# Initialize JSON results
echo "{" > "$RESULTS_FILE"
echo "  \"timestamp\": \"$TIMESTAMP\"," >> "$RESULTS_FILE"
echo "  \"config\": \"$SCRIPT_DIR/hyperparams/config.json\"," >> "$RESULTS_FILE"

# ===========================
# Run Python Training
# ===========================
echo -e "${GREEN}[1/2] Running Python (PyTorch) Training...${NC}"
echo ""

cd "$SCRIPT_DIR/gpt2-python"

# Check for virtual environment and activate if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate 2>/dev/null || true
fi

# Capture Python output
PYTHON_OUTPUT=$(python gpt2.py 2>&1)
echo "$PYTHON_OUTPUT"

# Extract Python metrics
PYTHON_METRICS=$(extract_metrics "$PYTHON_OUTPUT")
IFS='|' read -r PY_TRAIN_LOSS PY_VAL_LOSS PY_TIME PY_PARAMS <<< "$PYTHON_METRICS"

# Get device info
PY_DEVICE=$(echo "$PYTHON_OUTPUT" | grep -E "^Using device:" | sed 's/Using device: //')

echo ""
echo -e "${GREEN}Python training completed!${NC}"
echo ""

# ===========================
# Run Rust Training
# ===========================
echo -e "${YELLOW}[2/2] Running Rust (Burn) Training...${NC}"
echo ""

cd "$SCRIPT_DIR/gpt2-rust"

# Build and run in release mode
RUST_OUTPUT=$(cargo run --release 2>&1)
echo "$RUST_OUTPUT"

# Extract Rust metrics
RUST_METRICS=$(extract_metrics "$RUST_OUTPUT")
IFS='|' read -r RS_TRAIN_LOSS RS_VAL_LOSS RS_TIME RS_PARAMS <<< "$RUST_METRICS"

echo ""
echo -e "${YELLOW}Rust training completed!${NC}"
echo ""

# ===========================
# Write Results to JSON
# ===========================
cat >> "$RESULTS_FILE" << EOF
  "python": {
    "framework": "PyTorch",
    "device": "$PY_DEVICE",
    "training_time_seconds": $PY_TIME,
    "final_train_loss": $PY_TRAIN_LOSS,
    "final_val_loss": $PY_VAL_LOSS,
    "parameters": $PY_PARAMS
  },
  "rust": {
    "framework": "Burn (CUDA)",
    "device": "CUDA",
    "training_time_seconds": $RS_TIME,
    "final_train_loss": $RS_TRAIN_LOSS,
    "final_val_loss": $RS_VAL_LOSS,
    "parameters": $RS_PARAMS
  },
  "comparison": {
    "time_ratio_rust_vs_python": $(echo "scale=4; $RS_TIME / $PY_TIME" | bc),
    "faster_framework": "$(if (( $(echo "$PY_TIME < $RS_TIME" | bc -l) )); then echo "python"; else echo "rust"; fi)",
    "time_difference_seconds": $(echo "scale=2; if ($PY_TIME < $RS_TIME) $RS_TIME - $PY_TIME else $PY_TIME - $RS_TIME" | bc)
  }
}
EOF

# ===========================
# Print Summary
# ===========================
echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}              BENCHMARK RESULTS              ${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""
printf "%-20s %15s %15s\n" "Metric" "Python" "Rust"
printf "%-20s %15s %15s\n" "--------------------" "---------------" "---------------"
printf "%-20s %15s %15s\n" "Framework" "PyTorch" "Burn (CUDA)"
printf "%-20s %15s %15s\n" "Device" "$PY_DEVICE" "CUDA"
printf "%-20s %15ss %15ss\n" "Training Time" "$PY_TIME" "$RS_TIME"
printf "%-20s %15s %15s\n" "Final Train Loss" "$PY_TRAIN_LOSS" "$RS_TRAIN_LOSS"
printf "%-20s %15s %15s\n" "Final Val Loss" "$PY_VAL_LOSS" "$RS_VAL_LOSS"
printf "%-20s %15s %15s\n" "Parameters" "$PY_PARAMS" "$RS_PARAMS"
echo ""

# Determine winner
if (( $(echo "$PY_TIME < $RS_TIME" | bc -l) )); then
    SPEEDUP=$(echo "scale=2; $RS_TIME / $PY_TIME" | bc)
    echo -e "${GREEN}🏆 Winner: Python (PyTorch) - ${SPEEDUP}x faster${NC}"
else
    SPEEDUP=$(echo "scale=2; $PY_TIME / $RS_TIME" | bc)
    echo -e "${YELLOW}🏆 Winner: Rust (Burn) - ${SPEEDUP}x faster${NC}"
fi

echo ""
echo -e "Results saved to: ${BLUE}$RESULTS_FILE${NC}"
echo ""

# Also save a summary CSV for easy tracking across runs
CSV_FILE="$RESULTS_DIR/benchmark_history.csv"
if [ ! -f "$CSV_FILE" ]; then
    echo "timestamp,py_time,py_train_loss,py_val_loss,rs_time,rs_train_loss,rs_val_loss,winner" > "$CSV_FILE"
fi
WINNER=$(if (( $(echo "$PY_TIME < $RS_TIME" | bc -l) )); then echo "python"; else echo "rust"; fi)
echo "$TIMESTAMP,$PY_TIME,$PY_TRAIN_LOSS,$PY_VAL_LOSS,$RS_TIME,$RS_TRAIN_LOSS,$RS_VAL_LOSS,$WINNER" >> "$CSV_FILE"

echo -e "History appended to: ${BLUE}$CSV_FILE${NC}"
