#!/bin/bash
# Start vLLM server and save connection info
#
# Usage:
#   ./start_vllm_server.sh [MODEL_PATH] [OPTIONS]
#
# Examples:
#   ./start_vllm_server.sh /data/share/models/qwen2.5/Qwen2.5-32B-Instruct
#   ./start_vllm_server.sh /data/share/models/qwen2.5/Qwen2.5-32B-Instruct --tensor_parallel_size 2

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
MODEL_PATH="${1:-/data/share/models/qwen2.5/Qwen2.5-32B-Instruct}"
PORT="${PORT:-8888}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"

# Shift first arg if it's a model path
if [[ -n "$1" && ! "$1" =~ ^-- ]]; then
    shift
fi

# Create output directory for connection info
OUTPUT_DIR="${SCRIPT_DIR}/vllm_server_info"
mkdir -p "${OUTPUT_DIR}"

MODEL_NAME=$(basename "${MODEL_PATH}")
CONNECTION_FILE="${OUTPUT_DIR}/${MODEL_NAME}_connection.json"
LOG_FILE="${OUTPUT_DIR}/${MODEL_NAME}_server.log"

echo "=========================================="
echo "Starting vLLM Server"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Port: ${PORT}"
echo "Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"
echo "GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Max Model Length: ${MAX_MODEL_LEN}"
echo "Tool Call Parser: ${TOOL_CALL_PARSER}"
echo "Connection info will be saved to: ${CONNECTION_FILE}"
echo "Server logs: ${LOG_FILE}"
echo "=========================================="

# Start the server
python "${SCRIPT_DIR}/vllm_server.py" \
    --model_path "${MODEL_PATH}" \
    --port "${PORT}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --tool_call_parser "${TOOL_CALL_PARSER}" \
    --output_file "${CONNECTION_FILE}" \
    --log_file "${LOG_FILE}" \
    --wait \
    "$@"
