"""Render prompts, call a user-configured model, or verify a supplied Dafny file."""
import argparse
import json
from pathlib import Path

from promptkit.client import complete
from promptkit.formal import verify
from promptkit.records import extract_dafny, normalize_aggregation, normalize_semantic, parse_json_response
from promptkit.render import render


def read_json(path):
    return json.loads(path.read_text()) if path else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("render", "infer"):
        p = sub.add_parser(name)
        p.add_argument("--stage", choices=["formal", "semantic", "aggregation"], required=True)
        p.add_argument("--input", type=Path, required=True)
        p.add_argument("--semantic", type=Path)
        p.add_argument("--formal", type=Path)
        p.add_argument("--output", type=Path, required=True)
        if name == "infer":
            p.add_argument("--dafny-output", type=Path, help="Optional extracted .dfy output for the formal stage")
    p = sub.add_parser("verify")
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--dafny", default="dafny")
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        extra = getattr(args, "dafny_output", None)
        outputs = [args.output] + ([extra] if extra else [])
        if extra and args.stage != "formal":
            raise ValueError("--dafny-output is only valid for the formal stage")
        if len({p.resolve() for p in outputs}) != len(outputs):
            raise ValueError("Output paths must be distinct")
        if any(p.exists() for p in outputs):
            raise ValueError("Output already exists; choose a new output path")
        if args.command == "verify":
            result = verify(args.code, args.dafny, args.timeout)
        else:
            raw = read_json(args.input)
            semantic = read_json(args.semantic)
            formal = read_json(args.formal)
            prompt = render(args.stage, raw, semantic=semantic, formal=formal)
            if args.command == "render":
                result = prompt
            else:
                response, model = complete(prompt)
                if args.stage == "formal":
                    result = {"dafny_code": extract_dafny(response), "verification": "NOT_RUN"}
                elif args.stage == "semantic":
                    result = normalize_semantic(parse_json_response(response))
                else:
                    result = normalize_aggregation(parse_json_response(response))
                result.update({"origin": "model_response", "model": model, "stage": args.stage})
                result["input_origin"] = raw.get("example_origin", "user_supplied")
                if args.stage == "aggregation":
                    result["evidence_origins"] = {
                        "semantic": semantic.get("origin", "user_supplied"),
                        "formal": formal.get("origin", "user_supplied"),
                    }
        for path in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            f.write(result if isinstance(result, str) else json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        if extra:
            with extra.open("x") as f:
                f.write(result["dafny_code"])
        print(f"Wrote {args.output}")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Sample tool stopped: {exc}\n")


if __name__ == "__main__":
    main()
