#!/bin/bash
# Run iterative refinement for local models using vLLM
# Requires vLLM server to be running

set -e

# Configuration
MODEL=${1:-"qwen-2.5-7b"}  # Default to Qwen 7B
VLLM_URL=${2:-"http://localhost:8000/v1"}
MAX_ITERATIONS=${3:-3}
LANGUAGE=${4:-"dafny"}
DEBUG_SAMPLES=${5:-""}  # Empty for full run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Setup Ray temp directory for verification workers
export RAY_TMPDIR="${SCRIPT_DIR}/ray_tmp"
mkdir -p "$RAY_TMPDIR"

# Output directory
OUTPUT_DIR="./refinement_results"
mkdir -p "$OUTPUT_DIR"

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${OUTPUT_DIR}/run_${MODEL}_vllm_${TIMESTAMP}.log"

echo "======================================"
echo "Iterative Refinement - vLLM Models"
echo "======================================"
echo "Model: $MODEL"
echo "vLLM URL: $VLLM_URL"
echo "Max Iterations: $MAX_ITERATIONS"
echo "Language: $LANGUAGE"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "======================================"

# Check if vLLM server is running
echo "Checking vLLM server..."
if ! curl -s "${VLLM_URL}/models" > /dev/null 2>&1; then
    echo "Error: vLLM server not reachable at $VLLM_URL"
    echo "Please start the vLLM server first:"
    echo "  ./start_vllm_server.sh $MODEL"
    exit 1
fi
echo "vLLM server is running."

# Build command
CMD="python run_refinement.py \
    --model $MODEL \
    --vllm \
    --vllm_url $VLLM_URL \
    --data_path ../data/released_data.json \
    --output_dir $OUTPUT_DIR \
    --max_iterations $MAX_ITERATIONS \
    --language $LANGUAGE"

if [ -n "$DEBUG_SAMPLES" ]; then
    CMD="$CMD --debug $DEBUG_SAMPLES"
    echo "Debug Mode: $DEBUG_SAMPLES samples"
fi

echo ""
echo "Running: $CMD"
echo ""

# Run with logging
$CMD 2>&1 | tee "$LOG_FILE"

echo ""
echo "Done! Results saved to $OUTPUT_DIR"
echo "Log saved to $LOG_FILE"
