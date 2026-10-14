"""Sequential opt-in execution; no scheduler, retries, or model provisioning."""
import json
import os
from pathlib import Path

from promptkit.client import complete
from promptkit.formal import verify
from promptkit.records import extract_dafny, normalize_aggregation, normalize_semantic, parse_json_response
from promptkit.render import render


def write_json(path: Path, value):
    with path.open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def manifest(design):
    return {"experiment": design.experiment, "group_by": design.dimension,
            "label_origin": design.label_origin, "fingerprints": design.fingerprints,
            "max_api_calls": design.max_calls, "jobs": design.jobs()}


def plan(design, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    samples = {s["sample_id"]: s for s in design.samples}
    m = manifest(design)
    m["mode"] = "plan_only"
    write_json(output / "manifest.json", m)
    for job in design.jobs():
        folder = output / job["job_id"]
        folder.mkdir()
        action = samples[job["sample_id"]]["action"]
        (folder / "semantic_prompt.txt").write_text(render("semantic", action))
        if job["method"] == "hybrid":
            (folder / "formal_prompt.txt").write_text(render("formal", action))
    return m


def execute(design, output: Path, *, max_api_calls: int, dafny="dafny", caller=None, solver=None):
    if max_api_calls < design.max_calls:
        raise ValueError(f"Plan requires at most {design.max_calls} API calls; budget is smaller")
    caller = caller or complete
    solver = solver or verify
    profiles = {p["id"]: p for p in design.profiles}
    samples = {s["sample_id"]: s for s in design.samples}
    for p in profiles.values():
        if not os.environ.get(p["base_url_env"]) or not os.environ.get(p["model_env"]):
            raise ValueError(f"Configure model and endpoint environment variables for {p['id']}")
    if design.experiment == "cross_verifier":
        resolved = {(os.environ[p["base_url_env"]].rstrip("/"), os.environ[p["model_env"]]) for p in profiles.values()}
        if len(resolved) != len(profiles):
            raise ValueError("Cross-verifier profiles resolve to the same model and endpoint")
    output.mkdir(parents=True, exist_ok=False)
    m = manifest(design)
    m["mode"] = "execute"
    write_json(output / "manifest.json", m)
    calls_used = 0
    rows = []
    with (output / "results.jsonl").open("x") as stream:
        for job in design.jobs():
            folder = output / job["job_id"]
            folder.mkdir()
            action = samples[job["sample_id"]]["action"]
            profile = profiles[job["verifier_id"]]
            row = {**job, "status": "error", "input_origin": action.get("example_origin", "user_supplied")}
            stage = "semantic"
            used_before = calls_used

            def request(stage_name, **evidence):
                nonlocal calls_used
                prompt = render(stage_name, action, **evidence)
                (folder / f"{stage_name}_prompt.txt").write_text(prompt)
                calls_used += 1
                return caller(prompt, base_url=os.environ[profile["base_url_env"]],
                              model=os.environ[profile["model_env"]], api_key_env=profile["api_key_env"])

            try:
                response, model = request("semantic")
                semantic = normalize_semantic(parse_json_response(response))
                semantic.update({"origin": "model_response", "model": model})
                write_json(folder / "semantic.json", semantic)
                if job["method"] == "semantic_only":
                    decision = "PASS" if semantic["overall_score"] >= 60 else "BLOCK"
                else:
                    stage = "formal_generation"
                    response, _ = request("formal")
                    code = extract_dafny(response)
                    path = folder / "action.dfy"
                    path.write_text(code)
                    stage = "formal_verification"
                    formal = solver(path, executable=dafny)
                    write_json(folder / "formal.json", formal)
                    if formal.get("error_type"):
                        raise RuntimeError("Formal execution did not complete")
                    stage = "aggregation"
                    response, _ = request("aggregation", semantic=semantic, formal=formal)
                    final = normalize_aggregation(parse_json_response(response))
                    write_json(folder / "aggregation.json", final)
                    decision = final["decision"]
                row.update({"status": "complete", "prediction": decision, "model": model})
            except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
                row.update({"failed_stage": stage, "error_kind": type(exc).__name__})
            row["api_call_attempts"] = calls_used - used_before
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            rows.append(row)
    write_json(output / "run_status.json", {"jobs": len(rows), "api_call_attempts": calls_used,
               "complete": sum(r["status"] == "complete" for r in rows),
               "errors": sum(r["status"] == "error" for r in rows)})
    return rows
