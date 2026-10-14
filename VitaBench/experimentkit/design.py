"""Load a fixed experiment matrix. Labels are kept outside model inputs."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from reference.schemas import ActionContext

PROFILE_KEYS = {"id", "base_url_env", "model_env", "api_key_env"}
ACTION_KEYS = {"task_id", "user_intent", "content", "tool_calls", "history",
               "task_scenario", "tool_definitions", "system_prompt", "tool_results",
               "previous_tool_calls", "completed_steps", "example_origin"}
METHODS = {"hybrid": 3, "semantic_only": 1}


@dataclass
class Design:
    experiment: str
    dimension: str
    samples: list[dict]
    profiles: list[dict]
    methods: list[str]
    labels: dict
    label_origin: str
    fingerprints: dict

    def jobs(self):
        return [
            {"job_id": f"job_{n:04d}", "sample_id": sample["sample_id"],
             "verifier_id": profile["id"], "method": method,
             "group": profile["id"] if self.dimension == "verifier_id" else sample[self.dimension]}
            for n, (sample, profile, method) in enumerate(
                ((s, p, m) for s in self.samples for p in self.profiles for m in self.methods), 1)
        ]

    @property
    def max_calls(self):
        return len(self.samples) * len(self.profiles) * sum(METHODS[m] for m in self.methods)


def load_design(path: Path) -> Design:
    raw = path.read_bytes()
    config = json.loads(raw)
    if config.get("experiment") not in ("multi_agent", "adversarial", "cross_verifier"):
        raise ValueError("Unknown experiment")
    dimension = {"multi_agent": "source_agent", "adversarial": "attack_level", "cross_verifier": "verifier_id"}[config["experiment"]]
    methods = config.get("methods")
    if not isinstance(methods, list) or not methods or len(set(methods)) != len(methods) or not set(methods) <= set(METHODS):
        raise ValueError("Methods must be unique hybrid / semantic_only entries")
    profiles = config.get("verifiers")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("At least one verifier profile is required")
    ids = set()
    for p in profiles:
        if not isinstance(p, dict) or set(p) != PROFILE_KEYS:
            raise ValueError("Profiles contain only id and environment-variable names")
        if not isinstance(p["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", p["id"]) or p["id"] in ids:
            raise ValueError("Invalid or duplicate verifier id")
        ids.add(p["id"])
        if any(not isinstance(p[k], str) or not re.fullmatch(r"[A-Z][A-Z0-9_]+", p[k]) for k in PROFILE_KEYS - {"id"}):
            raise ValueError("Profile values must name environment variables")
    sample_path = path.parent / config["samples"]
    sample_bytes = sample_path.read_bytes()
    if len(sample_bytes) > 5_000_000:
        raise ValueError("Sample file exceeds configured driver size limit")
    samples = [json.loads(line) for line in sample_bytes.decode().splitlines() if line.strip()]
    if not samples or len(samples) > 1000:
        raise ValueError("Experiment driver requires 1–1000 sample actions")
    seen = set()
    for sample in samples:
        sid = sample.get("sample_id")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError("Sample ids must be nonempty and unique")
        seen.add(sid)
        action = sample.get("action")
        if not isinstance(action, dict) or not set(action) <= ACTION_KEYS:
            raise ValueError("Action contains unsupported fields; keep labels and metadata outside action")
        ActionContext.from_dict(action)
        if dimension != "verifier_id" and (not isinstance(sample.get(dimension), str) or not sample[dimension]):
            raise ValueError(f"Each sample requires {dimension}")
        if dimension == "attack_level" and sample[dimension] not in ("L1", "L2", "L3", "L4"):
            raise ValueError("Attack level must be L1–L4")
    fingerprints = {"config_sha256": hashlib.sha256(raw).hexdigest(), "samples_sha256": hashlib.sha256(sample_bytes).hexdigest()}
    labels = {}; origin = "not_supplied"
    if config.get("labels"):
        label_bytes = (path.parent / config["labels"]).read_bytes()
        table = json.loads(label_bytes)
        labels = table.get("labels"); origin = table.get("origin")
        if not isinstance(labels, dict) or not set(labels) <= seen or any(x not in ("PASS", "BLOCK") for x in labels.values()):
            raise ValueError("Label table must map known sample ids to PASS/BLOCK")
        if not isinstance(origin, str) or not origin:
            raise ValueError("Label source must be specified")
        fingerprints["labels_sha256"] = hashlib.sha256(label_bytes).hexdigest()
    if config["experiment"] == "cross_verifier" and len(profiles) < 2:
        raise ValueError("Cross-verifier comparison needs at least two profiles")
    return Design(config["experiment"], dimension, samples, profiles, methods, labels, origin, fingerprints)
