#!/bin/bash
# Run evaluation using a vLLM server
#
# This script:
# 1. Starts a vLLM server in the background
# 2. Waits for it to be ready
# 3. Runs evaluation using eval_vllm_client.py
# 4. Shuts down the server when done
#
# Usage:
#   ./run_vllm_eval.sh [MODEL_PATH] [OPTIONS]
#
# Examples:
#   ./run_vllm_eval.sh /data/share/models/qwen2.5/Qwen2.5-32B-Instruct
#   ./run_vllm_eval.sh /data/share/models/qwen2.5/Qwen2.5-32B-Instruct --num_workers 100
#   TENSOR_PARALLEL_SIZE=2 ./run_vllm_eval.sh /data/share/models/qwen2.5/Qwen2.5-32B-Instruct

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
MODEL_PATH="${1:-/data/share/models/qwen2.5/Qwen2.5-32B-Instruct}"
PORT="${PORT:-8899}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
NUM_WORKERS="${NUM_WORKERS:-50}"
GREEDY="${GREEDY:-0}"

# Shift first arg if it's a model path
EXTRA_ARGS=""
if [[ -n "$1" && ! "$1" =~ ^-- ]]; then
    shift
fi
EXTRA_ARGS="$@"

# Create output directory
OUTPUT_DIR="${SCRIPT_DIR}/vllm_server_info"
mkdir -p "${OUTPUT_DIR}"

MODEL_NAME=$(basename "${MODEL_PATH}")
CONNECTION_FILE="${OUTPUT_DIR}/${MODEL_NAME}_connection.json"
SERVER_LOG="${OUTPUT_DIR}/${MODEL_NAME}_server.log"
SERVER_PID_FILE="${OUTPUT_DIR}/${MODEL_NAME}_server.pid"

# Cleanup function
cleanup() {
    echo "Cleaning up..."

    # Kill vLLM server and all workers for this specific model
    if [[ -n "${MODEL_PATH}" ]]; then
        echo "Stopping all vLLM processes for model: ${MODEL_NAME}..."

        # Kill the main vLLM API server process (this will also terminate workers)
        pkill -TERM -f "vllm.entrypoints.openai.api_server.*${MODEL_PATH}" 2>/dev/null || true
        sleep 3

        # Force kill if still running
        pkill -KILL -f "vllm.entrypoints.openai.api_server.*${MODEL_PATH}" 2>/dev/null || true
    fi

    # Also kill via PID file (the launcher script)
    if [[ -f "${SERVER_PID_FILE}" ]]; then
        SERVER_PID=$(cat "${SERVER_PID_FILE}")
        if ps -p "${SERVER_PID}" > /dev/null 2>&1; then
            echo "Killing server launcher process (PID: ${SERVER_PID})..."
            kill -KILL "${SERVER_PID}" 2>/dev/null || true
        fi
        rm -f "${SERVER_PID_FILE}"
    fi

    # Final check: kill any orphaned vLLM workers on our port
    pkill -KILL -f "vllm.*--port ${PORT}" 2>/dev/null || true

    echo "Cleanup complete."
}

trap cleanup EXIT INT TERM

echo "=========================================="
echo "vLLM Evaluation Pipeline"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Port: ${PORT}"
echo "Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"
echo "GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Max Model Length: ${MAX_MODEL_LEN}"
echo "Tool Call Parser: ${TOOL_CALL_PARSER}"
echo "Num Workers: ${NUM_WORKERS}"
echo "Greedy: ${GREEDY}"
echo "=========================================="

# Step 1: Start vLLM server in background
echo ""
echo "Step 1: Starting vLLM server..."

# Remove stale connection file from previous runs
if [[ -f "${CONNECTION_FILE}" ]]; then
    echo "Removing stale connection file..."
    rm -f "${CONNECTION_FILE}"
fi

python "${SCRIPT_DIR}/vllm_server.py" \
    --model_path "${MODEL_PATH}" \
    --port "${PORT}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --tool_call_parser "${TOOL_CALL_PARSER}" \
    --output_file "${CONNECTION_FILE}" \
    --log_file "${SERVER_LOG}" \
    --wait &

SERVER_PID=$!
echo "${SERVER_PID}" > "${SERVER_PID_FILE}"
echo "Server starting with PID: ${SERVER_PID}"

# Wait for connection file to be created (server is ready)
echo "Waiting for server to be ready..."
MAX_WAIT=600  # 10 minutes
WAITED=0
while [[ ! -f "${CONNECTION_FILE}" && ${WAITED} -lt ${MAX_WAIT} ]]; do
    sleep 5
    WAITED=$((WAITED + 5))
    echo "  Waited ${WAITED}s..."

    # Check if server process died
    if ! ps -p "${SERVER_PID}" > /dev/null 2>&1; then
        echo "ERROR: Server process died. Check logs: ${SERVER_LOG}"
        exit 1
    fi
done

if [[ ! -f "${CONNECTION_FILE}" ]]; then
    echo "ERROR: Server did not start within ${MAX_WAIT} seconds"
    exit 1
fi

# Verify server is actually responding (not just file exists)
SERVER_URL=$(python3 -c "import json; print(json.load(open('${CONNECTION_FILE}'))['url'].replace('/v1', ''))")
echo "Verifying server health at ${SERVER_URL}..."
HEALTH_WAITED=0
while [[ ${HEALTH_WAITED} -lt 60 ]]; do
    if curl -s --max-time 5 "${SERVER_URL}/health" > /dev/null 2>&1; then
        echo "Server health check passed!"
        break
    fi
    sleep 2
    HEALTH_WAITED=$((HEALTH_WAITED + 2))
done

if [[ ${HEALTH_WAITED} -ge 60 ]]; then
    echo "ERROR: Server not responding to health checks"
    exit 1
fi

echo "Server is ready!"
echo "Connection info:"
cat "${CONNECTION_FILE}"

# Step 2: Run evaluation
echo ""
echo "Step 2: Running evaluation..."

python "${SCRIPT_DIR}/eval_vllm_client.py" \
    --connection_file "${CONNECTION_FILE}" \
    --num_workers "${NUM_WORKERS}" \
    --greedy "${GREEDY}" \
    ${EXTRA_ARGS}

echo ""
echo "Evaluation complete!"
echo "Server logs: ${SERVER_LOG}"
