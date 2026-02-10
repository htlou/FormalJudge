#!/usr/bin/env python3
"""
Direct Judge: Pure natural language safety evaluation (single round).

This is a simpler alternative to the 3-agent formal verification pipeline.
Uses a single LLM call to directly judge if agent behavior is safe or unsafe.

Usage:
    python run_direct_judge.py --data_path ../data/released_data.json \
                               --agent_output evaluation/evaluation_results/tot-gpt4o/gen_res.json \
                               --output_path ./direct_judge_results/gpt4o_results.json \
                               --model_name claude-opus-4-5-20251101 \
                               --num_workers 20
"""
import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formal_verification.pipeline.direct_judge import ParallelDirectJudge


def main():
    parser = argparse.ArgumentParser(
        description="Run direct judge (pure NL) on agent outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with Claude Opus (default)
    python run_direct_judge.py --data_path ../data/released_data.json \\
                               --agent_output ../evaluation/evaluation_results/tot-gpt4o/gen_res.json \\
                               --output_path ./direct_judge_results/gpt4o.json

    # Run with different model
    python run_direct_judge.py --data_path ../data/released_data.json \\
                               --agent_output ../evaluation/evaluation_results/tot-gpt4o/gen_res.json \\
                               --output_path ./direct_judge_results/gpt4o_gpt5.json \\
                               --model_name gpt-5
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
        "--output_path",
        type=str,
        required=True,
        help="Path to save judgment results"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="claude-opus-4-5-20251101",
        help="LLM model name for direct judgment (default: claude-opus-4-5-20251101)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=20,
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
        help="API key (defaults to env var XHUB_API_KEY)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0.0 for deterministic output)"
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=None,
        metavar="N",
        help="Debug mode: only process first N samples (e.g., --debug 10)"
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
        "max_tokens": 4096
    }
    
    runner = ParallelDirectJudge(
        model_name=args.model_name,
        num_workers=args.num_workers,
        api_base=args.api_base,
        api_key=args.api_key,
        generation_config=generation_config,
        max_samples=args.debug
    )
    
    # Run judgment
    summary = runner.run(
        data_path=args.data_path,
        agent_output_path=args.agent_output,
        output_path=args.output_path
    )
    
    print(f"\nResults saved to: {args.output_path}")
    return summary


if __name__ == "__main__":
    main()

