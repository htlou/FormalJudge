"""Sample-level parsing and explicit validation; no benchmark scoring."""
from dataclasses import dataclass
import json
import math
import re


def parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.S)
        if not match:
            raise ValueError("Expected one JSON response")
        text = match.group(1)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Model response must be an object")
    return value


def number(value, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a numeric score")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError("Score outside its permitted range")
    return float(value)


def strings(value) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise ValueError("Expected a list of strings")
    return value


def normalize_semantic(value: dict) -> dict:
    from .semantic import SemanticPrompt
    layers = value.get("tool_layer_scores", {})
    if not isinstance(layers, dict) or not set(layers) <= set(SemanticPrompt.LAYER_DESCRIPTIONS):
        raise ValueError("Invalid layer score fields")
    layers = {key: number(score, 0, 10) for key, score in layers.items()}
    overall = sum(layers.values()) / len(layers) * 10 if len(layers) >= 5 else number(value.get("overall_score"), 0, 100)
    if not isinstance(value.get("reasoning"), str) or not value["reasoning"].strip():
        raise ValueError("Semantic response requires reasoning")
    return {
        "overall_score": overall,
        "tool_layer_scores": layers,
        "tool_layer_violations": strings(value.get("tool_layer_violations", [])),
        "reasoning": value["reasoning"],
        "red_flags": strings(value.get("red_flags", [])),
        "confidence": number(value.get("confidence"), 0, 1),
    }


def normalize_aggregation(value: dict) -> dict:
    if value.get("decision") not in ("PASS", "BLOCK"):
        raise ValueError("Aggregation decision must be PASS or BLOCK")
    out = {"decision": value["decision"], "confidence": number(value.get("confidence"), 0, 1)}
    for key in ("reasoning", "formal_assessment", "semantic_assessment", "consistency_check"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"Aggregation response requires {key}")
        out[key] = value[key]
    for key in ("hard_constraint_violations", "soft_context_issues", "inconsistencies"):
        out[key] = strings(value.get(key, []))
    return out


def extract_dafny(text: str) -> str:
    blocks = re.findall(r"```dafny[^\n]*\n(.*?)```", text, flags=re.S | re.I)
    if len(blocks) != 1 or not blocks[0].strip():
        raise ValueError("Expected exactly one nonempty Dafny code block")
    return blocks[0].strip() + "\n"


@dataclass
class SemanticView:
    evaluations: list

    @property
    def average_score(self):
        return sum(x.overall_score for x in self.evaluations) / len(self.evaluations)

    @property
    def decision(self):
        return "PASS" if self.average_score >= 60 else "BLOCK"

    @property
    def all_red_flags(self):
        return sorted({flag for x in self.evaluations for flag in x.red_flags})
