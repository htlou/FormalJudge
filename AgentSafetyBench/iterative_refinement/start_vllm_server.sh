#!/bin/bash
# Start vLLM server for local model inference

set -e

MODEL=${1:-"qwen-2.5-7b"}
PORT=${2:-8000}
TENSOR_PARALLEL=${3:-1}

# Model path mapping
declare -A MODEL_PATHS
MODEL_PATHS["qwen-2.5-7b"]="Qwen/Qwen2.5-7B-Instruct"
MODEL_PATHS["qwen-2.5-14b"]="Qwen/Qwen2.5-14B-Instruct"
MODEL_PATHS["qwen-2.5-32b"]="Qwen/Qwen2.5-32B-Instruct"
MODEL_PATHS["qwen-2.5-72b"]="Qwen/Qwen2.5-72B-Instruct"

# Tensor parallel size mapping (based on model size)
declare -A TP_SIZES
TP_SIZES["qwen-2.5-7b"]=1
TP_SIZES["qwen-2.5-14b"]=1
TP_SIZES["qwen-2.5-32b"]=2
TP_SIZES["qwen-2.5-72b"]=4

MODEL_PATH=${MODEL_PATHS[$MODEL]}
if [ -z "$MODEL_PATH" ]; then
    echo "Error: Unknown model $MODEL"
    echo "Available models: ${!MODEL_PATHS[@]}"
    exit 1
fi

# Use default TP size if not specified
if [ "$TENSOR_PARALLEL" == "1" ] && [ -n "${TP_SIZES[$MODEL]}" ]; then
    TENSOR_PARALLEL=${TP_SIZES[$MODEL]}
fi

echo "======================================"
echo "Starting vLLM Server"
echo "======================================"
echo "Model: $MODEL"
echo "Model Path: $MODEL_PATH"
echo "Port: $PORT"
echo "Tensor Parallel: $TENSOR_PARALLEL"
echo "======================================"

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --port "$PORT" \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9
