#!/usr/bin/env python3
"""
Analyze and compare iterative refinement results across models.

Usage:
    python analyze_results.py --results_dir ./refinement_results

    # Compare specific runs
    python analyze_results.py --runs claude-opus-4-5_20260128_120000 gpt-5_20260128_120000
"""
import argparse
import json
import os
import glob
from typing import Dict, Any, List
from collections import defaultdict


def load_summary(run_dir: str) -> Dict[str, Any]:
    """Load summary from a run directory."""
    summary_path = os.path.join(run_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            return json.load(f)
    return None


def load_traces(run_dir: str) -> List[Dict[str, Any]]:
    """Load full traces from a run directory."""
    traces_path = os.path.join(run_dir, "verification_format", "full_traces.json")
    if os.path.exists(traces_path):
        with open(traces_path, 'r') as f:
            return json.load(f)
    return []


def analyze_run(run_dir: str) -> Dict[str, Any]:
    """Analyze a single run."""
    summary = load_summary(run_dir)
    traces = load_traces(run_dir)

    if not summary:
        return None

    # Compute detailed statistics
    iterations_to_pass = []
    failure_reasons = defaultdict(int)

    for trace in traces:
        if trace.get("final_passed"):
            iterations_to_pass.append(trace["total_iterations"])
        else:
            # Analyze failure reason from last iteration
            iterations = trace.get("iterations", [])
            if iterations:
                last_iter = iterations[-1]
                verification = last_iter.get("verification", {})
                explanation = verification.get("explanation", "")

                # Categorize failure
                if "parse error" in explanation.lower():
                    failure_reasons["Parse Error"] += 1
                elif "timeout" in explanation.lower():
                    failure_reasons["Timeout"] += 1
                elif "safety" in explanation.lower():
                    failure_reasons["Safety Violation"] += 1
                else:
                    failure_reasons["Other"] += 1

    # Compute statistics
    stats = {
        "model_name": summary.get("model_name", "unknown"),
        "run_dir": run_dir,
        "total_samples": summary.get("total_samples", 0),
        "passed_samples": summary.get("passed_samples", 0),
        "pass_rate": summary.get("pass_rate", 0),
        "avg_iterations": summary.get("avg_iterations", 0),
        "iterations_distribution": summary.get("iterations_distribution", {}),
    }

    if iterations_to_pass:
        stats["avg_iterations_to_pass"] = sum(iterations_to_pass) / len(iterations_to_pass)
        stats["min_iterations_to_pass"] = min(iterations_to_pass)
        stats["max_iterations_to_pass"] = max(iterations_to_pass)
    else:
        stats["avg_iterations_to_pass"] = None

    stats["failure_reasons"] = dict(failure_reasons)

    return stats


def compare_runs(results: List[Dict[str, Any]]) -> None:
    """Print comparison table for multiple runs."""
    if not results:
        print("No results to compare.")
        return

    print("\n" + "=" * 100)
    print("Comparison of Iterative Refinement Results")
    print("=" * 100)

    # Header
    headers = ["Model", "Samples", "Passed", "Pass Rate", "Avg Iters", "Avg to Pass"]
    widths = [25, 10, 10, 12, 12, 12]

    header_line = " | ".join(h.center(w) for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))

    # Rows
    for result in sorted(results, key=lambda x: x.get("pass_rate", 0), reverse=True):
        row = [
            result.get("model_name", "unknown")[:24],
            str(result.get("total_samples", 0)),
            str(result.get("passed_samples", 0)),
            f"{result.get('pass_rate', 0):.2%}",
            f"{result.get('avg_iterations', 0):.2f}",
            f"{result.get('avg_iterations_to_pass', 'N/A'):.2f}" if result.get('avg_iterations_to_pass') else "N/A"
        ]
        row_line = " | ".join(str(v).center(w) for v, w in zip(row, widths))
        print(row_line)

    print("=" * 100)

    # Failure analysis
    print("\nFailure Reason Analysis:")
    print("-" * 50)
    for result in results:
        print(f"\n{result.get('model_name', 'unknown')}:")
        failure_reasons = result.get("failure_reasons", {})
        if failure_reasons:
            for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {reason}: {count}")
        else:
            print("  - No failures")


def find_runs(results_dir: str) -> List[str]:
    """Find all run directories."""
    runs = []
    for entry in os.listdir(results_dir):
        entry_path = os.path.join(results_dir, entry)
        if os.path.isdir(entry_path):
            summary_path = os.path.join(entry_path, "summary.json")
            if os.path.exists(summary_path):
                runs.append(entry_path)
    return sorted(runs)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze iterative refinement results"
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="./refinement_results",
        help="Directory containing results"
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        help="Specific run directories to analyze"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for analysis (JSON)"
    )

    args = parser.parse_args()

    # Find runs to analyze
    if args.runs:
        run_dirs = [os.path.join(args.results_dir, r) if not os.path.isabs(r) else r
                    for r in args.runs]
    else:
        run_dirs = find_runs(args.results_dir)

    if not run_dirs:
        print(f"No runs found in {args.results_dir}")
        return

    print(f"Found {len(run_dirs)} runs to analyze")

    # Analyze each run
    results = []
    for run_dir in run_dirs:
        print(f"Analyzing: {run_dir}")
        analysis = analyze_run(run_dir)
        if analysis:
            results.append(analysis)

    # Print comparison
    compare_runs(results)

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nAnalysis saved to: {args.output}")


if __name__ == "__main__":
    main()
