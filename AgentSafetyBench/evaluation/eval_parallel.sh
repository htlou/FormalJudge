#!/bin/bash

# Parallel evaluation script using Ray
# Usage: ./eval_parallel.sh [num_workers] [other_args...]
# Example: ./eval_parallel.sh 8 --model_name gpt-5 --regen_exceed 1 --extra_info ""


# CUDA_VISIBLE_DEVICES=0 python -u eval_parallel.py --num_workers 50 --model_name gpt-5 --regen_exceed 1 --extra_info ""
# CUDA_VISIBLE_DEVICES=0 python -u eval_parallel.py --num_workers 25 --model_name gemini-3-pro-preview --greedy 0 --regen_exceed 1 --extra_info ""
# for more models, please see eval_parallel.py