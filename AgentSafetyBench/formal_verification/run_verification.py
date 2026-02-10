#!/usr/bin/env python3
"""
Main entry point for formal verification pipeline.

Usage:
    python run_verification.py --data_path ../data/released_data.json \
                               --agent_output evaluation/evaluation_results/tot-gpt4o/gen_res.json \
                               --output_dir ./verification_results/gpt4o_dafny \
                               --language dafny \
                               --model_name gpt-4o \
                               --num_workers 4

Ablation settings:
    --language dafny          # Full formal verification with Dafny
    --language python         # Python verification without formal proof
    --language natural_language  # Direct LLM judgment (boolean output)
"""
import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formal_verification.pipeline import ParallelVerificationRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run formal verification pipeline on agent outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with Dafny (default)
    python run_verification.py --data_path ../data/released_data.json \\
                               --agent_output ../evaluation/evaluation_results/tot-gpt4o/gen_res.json \\
                               --output_dir ./verification_results/gpt4o_dafny

    # Run with Python (no formal proof)
    python run_verification.py --data_path ../data/released_data.json \\
                               --agent_output ../evaluation/evaluation_results/tot-gpt4o/gen_res.json \\
                               --output_dir ./verification_results/gpt4o_python \\
                               --language python

    # Run with natural language (direct LLM judgment)
    python run_verification.py --data_path ../data/released_data.json \\
                               --agent_output ../evaluation/evaluation_results/tot-gpt4o/gen_res.json \\
                               --output_dir ./verification_results/gpt4o_nl \\
                               --language natural_language
        """
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/released_data.json",
        help="Path to benchmark data JSON"
    )
    parser.add_argument(
        "--agent_output",
        type=str,
        required=True,
        help="Path to agent output JSON (from evaluation)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save verification results"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="dafny",
        choices=["dafny", "python", "natural_language", "llm_cot", "llm_fewshot"],
        help="Verification language (default: dafny)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o",
        help="LLM model name for verification agents"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default=None,
        help="API base URL (defaults to env var API_BASE_URL)"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key (defaults to env var OPENAI_API_KEY or XHUB_API_KEY)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0.0)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Max tokens for LLM response"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache intermediate Dafny files for debugging (defaults to output_dir/cache)"
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=None,
        metavar="N",
        help="Debug mode: only process first N samples (e.g., --debug 10)"
    )
    parser.add_argument(
        "--dafny_timeout",
        type=int,
        default=120,
        help="Timeout in seconds for Dafny execution (default: 120)"
    )
    parser.add_argument(
        "--dafny_workers",
        type=int,
        default=4,
        help="Number of parallel workers for Dafny execution (default: 4, keep low to avoid CPU contention)"
    )

    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.data_path):
        print(f"Error: Data path does not exist: {args.data_path}")
        sys.exit(1)
    
    if not os.path.exists(args.agent_output):
        print(f"Error: Agent output path does not exist: {args.agent_output}")
        sys.exit(1)
    
    # Create runner
    generation_config = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens
    }

    # Set cache_dir to output_dir/cache by default
    cache_dir = args.cache_dir or os.path.join(args.output_dir, "cache")

    runner = ParallelVerificationRunner(
        model_name=args.model_name,
        language=args.language,
        num_workers=args.num_workers,
        api_base=args.api_base,
        api_key=args.api_key,
        generation_config=generation_config,
        cache_dir=cache_dir,
        max_samples=args.debug,
        dafny_timeout=args.dafny_timeout,
        dafny_workers=args.dafny_workers
    )
    
    # Run verification
    summary = runner.run_full_verification(
        data_path=args.data_path,
        agent_output_path=args.agent_output,
        output_dir=args.output_dir
    )
    
    print(f"\nResults saved to: {args.output_dir}")
    return summary


if __name__ == "__main__":
    main()

