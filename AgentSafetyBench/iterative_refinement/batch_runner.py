#!/usr/bin/env python3
"""
Batch runner for parallel iterative refinement across multiple models.

This script allows running multiple models simultaneously using separate processes.
Useful for maximizing throughput when running on a multi-GPU cluster.

Usage:
    # Run all API models in parallel
    python batch_runner.py --models claude-opus-4-5 gpt-5 --parallel

    # Run specific models
    python batch_runner.py --models claude-opus-4-5 qwen-2.5-7b --max_iterations 3
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, Any, List
import multiprocessing as mp


def run_model(args):
    """Run refinement for a single model."""
    model_name, config = args

    cmd = [
        sys.executable, "run_refinement.py",
        "--model", model_name,
        "--data_path", config["data_path"],
        "--output_dir", config["output_dir"],
        "--max_iterations", str(config["max_iterations"]),
        "--language", config["language"]
    ]

    # Add vLLM args if needed
    if config.get("use_vllm"):
        cmd.extend(["--vllm", "--vllm_url", config.get("vllm_url", "http://localhost:8000/v1")])
    else:
        cmd.extend(["--num_workers", str(config.get("num_workers", 8))])

    if config.get("debug_samples"):
        cmd.extend(["--debug", str(config["debug_samples"])])

    print(f"[{model_name}] Starting...")
    print(f"[{model_name}] Command: {' '.join(cmd)}")

    # Run subprocess
    log_file = os.path.join(config["output_dir"], f"batch_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    os.makedirs(config["output_dir"], exist_ok=True)

    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        process.wait()

    print(f"[{model_name}] Completed with return code {process.returncode}")
    print(f"[{model_name}] Log saved to: {log_file}")

    return model_name, process.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for iterative refinement"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["claude-opus-4-5", "gpt-5"],
        help="Models to run"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/released_data.json",
        help="Path to benchmark data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./refinement_results",
        help="Output directory"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=3,
        help="Maximum iterations"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Workers per model for API models"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="dafny",
        choices=["dafny", "python", "natural_language"]
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run models in parallel"
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=None,
        help="Debug mode: process N samples"
    )
    # vLLM URL mapping for local models
    parser.add_argument(
        "--vllm_urls",
        type=str,
        default="{}",
        help="JSON mapping of model to vLLM URL (e.g., '{\"qwen-2.5-7b\": \"http://localhost:8001/v1\"}')"
    )

    args = parser.parse_args()

    vllm_urls = json.loads(args.vllm_urls)

    # Determine which models use vLLM
    vllm_models = {"qwen-2.5-7b", "qwen-2.5-14b", "qwen-2.5-32b", "qwen-2.5-72b"}

    # Prepare configs for each model
    model_configs = []
    for model in args.models:
        config = {
            "data_path": args.data_path,
            "output_dir": args.output_dir,
            "max_iterations": args.max_iterations,
            "num_workers": args.num_workers,
            "language": args.language,
            "debug_samples": args.debug
        }

        if model in vllm_models:
            config["use_vllm"] = True
            config["vllm_url"] = vllm_urls.get(model, f"http://localhost:8000/v1")

        model_configs.append((model, config))

    print("=" * 60)
    print("Batch Iterative Refinement Runner")
    print("=" * 60)
    print(f"Models: {args.models}")
    print(f"Max Iterations: {args.max_iterations}")
    print(f"Language: {args.language}")
    print(f"Parallel: {args.parallel}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    if args.parallel:
        # Run in parallel using multiprocessing
        with mp.Pool(len(model_configs)) as pool:
            results = pool.map(run_model, model_configs)
    else:
        # Run sequentially
        results = [run_model(cfg) for cfg in model_configs]

    # Print summary
    print("\n" + "=" * 60)
    print("Batch Run Complete")
    print("=" * 60)
    for model, returncode in results:
        status = "SUCCESS" if returncode == 0 else f"FAILED (code {returncode})"
        print(f"  {model}: {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
