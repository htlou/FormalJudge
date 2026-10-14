"""Portable adapters around extracted prompt builders."""
import json
from pathlib import Path
from types import SimpleNamespace

from reference.schemas import ActionContext
from .semantic import SemanticPrompt
from .aggregation import AggregationPrompt
from .records import SemanticView, normalize_semantic

ROOT = Path(__file__).resolve().parents[1]


def render(stage: str, raw: dict, *, semantic=None, formal=None) -> str:
    action = ActionContext.from_dict(raw)
    calls = [{"name": x.name, "arguments": x.arguments} for x in action.tool_calls]
    history = [SimpleNamespace(**x) for x in action.history]
    text = action.content or json.dumps(calls, ensure_ascii=False)
    scenario = raw.get("task_scenario")
    if scenario is not None and not isinstance(scenario, str):
        raise ValueError("task_scenario must be a string")
    context = SemanticPrompt()._format_conversation_context(history)
    if stage == "semantic":
        return SemanticPrompt()._build_evaluation_prompt(
            text_content=text, conversation_context=history,
            user_intent=action.user_intent, task_scenario=scenario,
            tool_calls=calls or None, tool_results=raw.get("tool_results"),
            previous_tool_calls=raw.get("previous_tool_calls"),
            verified_facts=None, tool_name=calls[0]["name"] if calls else None,
        )
    if stage == "formal":
        template = (ROOT / "prompts/formal_generation.txt").read_text()
        # The extracted template contains an extra escape layer in its final
        # Dafny examples. Normalize only the template, before inserting input:
        # never interpret or rewrite braces in user-supplied task/history text.
        template = template.replace("{{{{", "{{").replace("}}}}", "}}")
        return template.format(
            layer_definitions=(ROOT / "prompts/layers.txt").read_text(),
            system_prompt=raw.get("system_prompt") or "No system prompt",
            conversation_history=json.dumps(list(action.history), ensure_ascii=False, indent=2),
            current_user_message=action.user_intent,
            agent_output=text,
            interaction_count=len(action.history), tool_call_count=len(calls),
            completed_steps=raw.get("completed_steps", "None"),
            tools_definitions=json.dumps(raw.get("tool_definitions", []), ensure_ascii=False, indent=2),
            task_context=action.user_intent + ("\n" + scenario if scenario else ""),
            historical_rules="**First Round**: Generate complete Dafny program from scratch based on layer analysis.",
        )
    if stage == "aggregation":
        if not isinstance(semantic, dict) or not isinstance(formal, dict):
            raise ValueError("Aggregation requires semantic and formal records")
        normalized = normalize_semantic(semantic)
        status = formal.get("status")
        if status not in ("VERIFIED", "UNPROVEN", "VIOLATION"):
            raise ValueError("Formal record is not a completed verification record")
        if formal.get("error_type"):
            raise ValueError("Formal execution did not complete; no aggregation performed")
        if not isinstance(formal.get("reasoning"), str):
            raise ValueError("Formal record requires reasoning")
        violations = formal.get("violations", [])
        if not isinstance(violations, list) or any(not isinstance(x, str) for x in violations):
            raise ValueError("Formal violations must be a list of strings")
        view = SimpleNamespace(status=status, reasoning=formal["reasoning"],
                               violations=violations, error_type=None)
        return AggregationPrompt()._build_aggregation_prompt(
            formal_result=view,
            semantic_verdict=SemanticView([SimpleNamespace(**normalized)]),
            agent_text=action.content, agent_tool_calls=calls,
            conversation_context=context, user_intent=action.user_intent,
            task_scenario=scenario, accumulated_facts=None,
        )
    raise ValueError("Unknown prompt stage")
