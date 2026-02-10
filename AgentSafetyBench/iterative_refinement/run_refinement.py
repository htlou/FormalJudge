#!/usr/bin/env python3
"""
Main entry point for iterative refinement pipeline.

Usage:
    # Run with API model (Claude, GPT-5) using Ray
    python run_refinement.py --model claude-opus-4-5 --max_iterations 3

    # Run with local vLLM model
    python run_refinement.py --model qwen-2.5-7b --vllm --vllm_url http://localhost:8000/v1

    # Debug mode (few samples)
    python run_refinement.py --model gpt-5 --debug 10

    # Custom shield model path
    python run_refinement.py --model claude-opus-4-5 --shield_model /path/to/ShieldAgent
"""
import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iterative_refinement import (
    TARGET_MODELS,
    RefinementConfig,
    IterativeRefinementPipeline,
    JUDGE_MODEL
)
from iterative_refinement.vllm_pipeline import run_vllm_refinement
import json


def main():
    parser = argparse.ArgumentParser(
        description="Run iterative refinement with formal verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  API models (use Ray):
    - claude-opus-4-5
    - gpt-5
    - gemini-3-pro-preview

  Local models (use vLLM):
    - qwen-2.5-7b
    - qwen-2.5-14b
    - qwen-2.5-32b
    - qwen-2.5-72b

Examples:
    # Run Claude with Dafny verification
    python run_refinement.py --model claude-opus-4-5 --language dafny

    # Run local Qwen model with vLLM server
    python run_refinement.py --model qwen-2.5-7b --vllm --vllm_url http://localhost:8000/v1

    # Debug mode with 10 samples
    python run_refinement.py --model gpt-5 --debug 10 --max_iterations 2

    # Custom shield model
    python run_refinement.py --model claude-opus-4-5 --shield_model /path/to/ShieldAgent
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(TARGET_MODELS.keys()),
        help="Target model name"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/released_data.json",
        help="Path to benchmark data JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./refinement_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=3,
        help="Maximum refinement iterations (default: 3)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel workers for API calls (default: 8)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="dafny",
        choices=["dafny", "python", "natural_language", "direct", "baseline", "llm_cot", "llm_fewshot"],
        help="Verification language: dafny, python, natural_language, direct (no verification), baseline (shield model), llm_cot (LLM+CoT), llm_fewshot (LLM+Few-Shot)"
    )
    parser.add_argument(
        "--shield_model",
        type=str,
        default="/data/hantao/models/ShieldAgent",
        help="Path to ShieldAgent model for safety evaluation (default: /data/hantao/models/ShieldAgent)"
    )
    parser.add_argument(
        "--shield_batch_size",
        type=int,
        default=16,
        help="Batch size for shield model evaluation (default: 16)"
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Use vLLM server for inference (for local models)"
    )
    parser.add_argument(
        "--vllm_url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM server URL (default: http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=None,
        metavar="N",
        help="Debug mode: process only first N samples"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from previous run (path to verification_format/full_traces.json)"
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.data_path):
        print(f"Error: Data path does not exist: {args.data_path}")
        sys.exit(1)

    # Determine backend
    model_config = TARGET_MODELS[args.model]
    use_vllm = args.vllm or model_config.backend == "vllm"

    print(f"\n{'='*60}")
    print(f"Iterative Refinement Pipeline")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Backend: {'vLLM' if use_vllm else 'Ray (API)'}")
    print(f"Verification: {args.language}")
    print(f"Shield Model: {args.shield_model}")
    print(f"Max Iterations: {args.max_iterations}")
    print(f"Data Path: {args.data_path}")
    print(f"Output Directory: {args.output_dir}")
    if args.debug:
        print(f"Debug Mode: {args.debug} samples")
    print(f"{'='*60}\n")

    # Load samples
    with open(args.data_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # Create config with shield model settings
    config = RefinementConfig(
        max_iterations=args.max_iterations,
        num_workers=args.num_workers,
        verification_language=args.language,
        debug_samples=args.debug,
        data_path=args.data_path,
        output_base_dir=args.output_dir,
        shield_model_path=args.shield_model,
        shield_batch_size=args.shield_batch_size
    )

    # Run pipeline
    if use_vllm:
        summary = run_vllm_refinement(
            model_name=args.model,
            vllm_base_url=args.vllm_url,
            data_path=args.data_path,
            output_dir=args.output_dir,
            max_iterations=args.max_iterations,
            verification_language=args.language,
            debug_samples=args.debug
        )
    else:
        # Create pipeline with full config
        pipeline = IterativeRefinementPipeline(
            model_config=model_config,
            judge_config=JUDGE_MODEL,
            refinement_config=config
        )
        summary = pipeline.run_batch_refinement(
            samples=samples,
            output_dir=args.output_dir,
            resume_from=args.resume
        )

    print(f"\nResults saved to: {summary['output_dir']}")
    return summary


if __name__ == "__main__":
    main()
