#!/bin/bash
# Run direct judge (pure NL, single round) with Claude Opus

# Default values
TIMESTAMP=$(date +"%m%d_%H%M%S")
DATA_PATH="~/FormalJudge/AgentSafetyBench/data/released_data.json"
MODEL_NAME="claude-opus-4-5-20251101"
NUM_WORKERS=128
AGENT_OUTPUT="~/FormalJudge/AgentSafetyBench/evaluation/evaluation_results/tot-claude-opus-4-5-20251101/gen_res.json"
OUTPUT_PATH="~/FormalJudge/AgentSafetyBench/formal_verification/verification_results/direct_judge_claude-opus-4-5-20251101_${TIMESTAMP}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --agent_output)
            AGENT_OUTPUT="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$AGENT_OUTPUT" ]; then
    echo "Error: --agent_output is required"
    echo "Usage: $0 --agent_output <path> --output_path <path> [--model_name <model>] [--num_workers <n>]"
    exit 1
fi

if [ -z "$OUTPUT_PATH" ]; then
    echo "Error: --output_path is required"
    echo "Usage: $0 --agent_output <path> --output_path <path> [--model_name <model>] [--num_workers <n>]"
    exit 1
fi

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
RAY_TMPDIR="/data/hantao/cache/ray_${TIMESTAMP}"
mkdir -p "$RAY_TMPDIR"
export RAY_TMPDIR

# Run direct judge
python formal_verification/run_direct_judge.py \
    --data_path "$DATA_PATH" \
    --agent_output "$AGENT_OUTPUT" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --num_workers "$NUM_WORKERS"

