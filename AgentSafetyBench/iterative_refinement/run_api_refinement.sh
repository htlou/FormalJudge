#!/bin/bash
# Run iterative refinement for API models (Claude, GPT-5)
# Uses Ray for parallel API calls
#
# Usage:
#   ./run_api_refinement.sh [MODEL] [MAX_ITERATIONS] [NUM_WORKERS] [LANGUAGE] [DEBUG_SAMPLES] [SHIELD_MODEL]
#
# Language options:
#   - dafny: Formal verification with Dafny
#   - python: Python-based verification
#   - natural_language: Natural language verification
#   - direct: No verification (baseline - always passes)
#   - baseline: Shield model evaluation (safety classifier)
#   - llm_cot: LLM + Chain of Thought (single round with CoT prompting)
#   - llm_fewshot: LLM + Few-Shot (single round with few-shot examples)
#
# Pipeline behavior:
#   - PASSED/FAILED is determined by Shield model (not verification)
#   - All samples go through ALL iterations (not just failed ones)
#   - Alignment rate = agreement between verification and shield judgments
#
# Examples:
#   ./run_api_refinement.sh gemini-3-pro-preview 3 64 dafny        # Formal verification
#   ./run_api_refinement.sh gpt-5 1 32 direct                       # Direct (no verification)
#   ./run_api_refinement.sh claude-opus-4-5 1 32 baseline           # Shield model baseline
#   ./run_api_refinement.sh claude-opus-4-5 1 32 llm_cot            # LLM + Chain of Thought
#   ./run_api_refinement.sh claude-opus-4-5 1 32 llm_fewshot        # LLM + Few-Shot
#   ./run_api_refinement.sh gemini-3-pro-preview 3 64 dafny 10      # Debug with 10 samples
#   ./run_api_refinement.sh claude-opus-4-5 3 64 dafny "" /path/to/ShieldAgent  # Custom shield model

set -e

# Configuration
MODEL=${1:-"claude-opus-4-5-20251101"}  # Default to Claude
MAX_ITERATIONS=${2:-3}
NUM_WORKERS=${3:-64}
LANGUAGE=${4:-"dafny"}  # dafny, python, natural_language, direct, baseline, llm_cot, llm_fewshot
DEBUG_SAMPLES=${5:-""}  # Empty for full run
SHIELD_MODEL=${6:-"~/models/ShieldAgent"}  # Shield model path
SHIELD_BATCH_SIZE=${7:-4}  # Batch size for shield model (reduce if OOM)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Setup Ray temp directory to avoid /tmp issues
export RAY_TMPDIR="~/cache"
mkdir -p "$RAY_TMPDIR"

# Set GPU for shield model (change as needed, e.g., "0", "0,1", "2,3")
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

# Output directory
OUTPUT_DIR="./refinement_results"
mkdir -p "$OUTPUT_DIR"

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${OUTPUT_DIR}/run_${MODEL}_${TIMESTAMP}.log"

echo "======================================"
echo "Iterative Refinement - API Models"
echo "======================================"
echo "Model: $MODEL"
echo "Max Iterations: $MAX_ITERATIONS"
echo "Workers: $NUM_WORKERS"
echo "Language: $LANGUAGE"
echo "Shield Model: $SHIELD_MODEL"
echo "Shield Batch Size: $SHIELD_BATCH_SIZE"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "======================================"

# Build command
CMD="python run_refinement.py \
    --model $MODEL \
    --data_path ../data/released_data.json \
    --output_dir $OUTPUT_DIR \
    --max_iterations $MAX_ITERATIONS \
    --num_workers $NUM_WORKERS \
    --language $LANGUAGE \
    --shield_model $SHIELD_MODEL \
    --shield_batch_size $SHIELD_BATCH_SIZE"

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
