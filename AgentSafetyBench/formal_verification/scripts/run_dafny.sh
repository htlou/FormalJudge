#!/bin/bash
# Run formal verification with Dafny

TIMESTAMP=$(date +"%m%d_%H%M%S")
# Default values
DATA_PATH="~/FormalJudge/AgentSafetyBench/data/released_data.json"
MODEL_NAME="claude-opus-4-5-20251101"
NUM_WORKERS=128
AGENT_OUTPUTS=(
    # PATH TO AGENT OUTPUTS
    )
OUTPUT_DIRS=(
    # PATH TO OUTPUT DIRS
    )

# Change to script directory
cd "$(dirname "$0")/../.."

# Kill any active Ray processes first
echo "Checking for active Ray processes..."
if command -v ray &> /dev/null; then
    # Try graceful shutdown first
    ray stop --force 2>/dev/null || true
    # Kill any remaining Ray processes
    pkill -f "ray.*gcs_server" 2>/dev/null || true
    pkill -f "ray.*raylet" 2>/dev/null || true
    pkill -f "ray.*dashboard" 2>/dev/null || true
    sleep 1
fi

# Generate unique timestamped cache directory for Ray
RAY_TMPDIR="~/cache/ray_${TIMESTAMP}"
mkdir -p "$RAY_TMPDIR"
export RAY_TMPDIR

for i in "${!AGENT_OUTPUTS[@]}"; do
    AGENT_OUTPUT="${AGENT_OUTPUTS[$i]}"
    OUTPUT_DIR="${OUTPUT_DIRS[$i]}"
    python formal_verification/run_verification.py \
        --data_path "$DATA_PATH" \
        --agent_output "$AGENT_OUTPUT" \
        --output_dir "$OUTPUT_DIR" \
        --language dafny \
        --model_name "$MODEL_NAME" \
        --num_workers "$NUM_WORKERS" \
        --dafny_workers 128
done