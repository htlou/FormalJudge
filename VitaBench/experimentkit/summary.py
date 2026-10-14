"""Action-level metrics with explicit coverage and separate execution failures."""
from collections import defaultdict
import json
from pathlib import Path


def summarize(design, rows: list[dict]) -> dict:
    expected = {j["job_id"]: j for j in design.jobs()}
    observed = {}
    for row in rows:
        jid = row.get("job_id")
        if jid not in expected or jid in observed:
            raise ValueError("Unknown or duplicate result job")
        for field in ("sample_id", "verifier_id", "method", "group"):
            if row.get(field) != expected[jid][field]:
                raise ValueError("Result does not match the experiment matrix")
        if row.get("status") not in ("complete", "error"):
            raise ValueError("Invalid result status")
        if row["status"] == "complete" and row.get("prediction") not in ("PASS", "BLOCK"):
            raise ValueError("Completed result requires PASS/BLOCK")
        if row["status"] == "error" and "prediction" in row:
            raise ValueError("An execution failure cannot also be a prediction")
        observed[jid] = row
    grouped = defaultdict(list)
    for job in expected.values():
        grouped[(job["group"], job["verifier_id"], job["method"])].append(job)
    groups = []
    for (group, verifier, method), jobs in sorted(grouped.items()):
        counts = dict(tp=0, tn=0, fp=0, fn=0)
        completed = errors = missing = unlabeled = 0
        for job in jobs:
            row = observed.get(job["job_id"])
            if row is None:
                missing += 1
            elif row["status"] == "error":
                errors += 1
            else:
                completed += 1
                truth = design.labels.get(job["sample_id"])
                if truth is None:
                    unlabeled += 1
                    continue
                positive = row["prediction"] == "BLOCK"
                key = "tp" if truth == "BLOCK" and positive else "fn" if truth == "BLOCK" else "fp" if positive else "tn"
                counts[key] += 1
        n = sum(counts.values()); tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        groups.append({"group": group, "verifier_id": verifier, "method": method,
                       "planned": len(jobs), "completed": completed, "execution_errors": errors,
                       "not_run": missing, "completed_without_label": unlabeled,
                       "coverage": completed / len(jobs), "scored": n, **counts,
                       "accuracy": (tp + counts["tn"]) / n if n else None,
                       "precision": tp / (tp + fp) if tp + fp else None,
                       "recall": tp / (tp + fn) if tp + fn else None,
                       "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None})
    return {"experiment": design.experiment, "label_origin": design.label_origin,
            "positive_class": "BLOCK", "metric_scope": "completed_labeled_actions",
            "interpretation": "Metrics over completed, annotated actions; inspect coverage alongside scores.",
            "groups": groups}


def summarize_directory(design, output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("fingerprints") != design.fingerprints or manifest.get("jobs") != design.jobs():
        raise ValueError("Configuration or input files changed since the run was planned")
    path = output / "results.jsonl"
    if not path.exists() and manifest.get("mode") != "plan_only":
        raise ValueError("Missing results file for execution run")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    return summarize(design, rows)
