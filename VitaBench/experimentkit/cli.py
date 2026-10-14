"""Shared CLI for the three evaluation workflows."""
import argparse
import json
from pathlib import Path

from .design import load_design
from .runner import execute, plan
from .summary import summarize_directory


def main(default_config: Path):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run", "summarize"])
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Explicitly enable model/solver calls")
    parser.add_argument("--max-api-calls", type=int)
    parser.add_argument("--dafny", default="dafny")
    args = parser.parse_args()
    try:
        design = load_design(args.config)
        if args.command == "run":
            if not args.execute or args.max_api_calls is None:
                raise ValueError("run requires --execute and --max-api-calls")
            execute(design, args.output, max_api_calls=args.max_api_calls, dafny=args.dafny)
            print(json.dumps(summarize_directory(design, args.output), indent=2))
        elif args.command == "plan":
            m = plan(design, args.output)
            print(json.dumps({"jobs": len(m["jobs"]), "max_api_calls": m["max_api_calls"], "model_calls_made": 0}, indent=2))
        else:
            print(json.dumps(summarize_directory(design, args.output), indent=2))
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"Experiment stopped: {exc}\n")
