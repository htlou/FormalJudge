"""Input data structures for action verification."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ActionContext:
    task_id: str
    user_intent: str
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    history: tuple[dict[str, Any], ...] = ()
    # The preview never interprets these fields as established facts.
    supplied_context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ActionContext":
        if not isinstance(raw, dict):
            raise ValueError("Input must be an object")
        for key in ("task_id", "user_intent"):
            if not isinstance(raw.get(key), str) or not raw[key].strip():
                raise ValueError(f"{key} must be a nonempty string")
        content = raw.get("content", "")
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        calls = raw.get("tool_calls", [])
        if not isinstance(calls, list):
            raise ValueError("tool_calls must be an array")
        parsed = []
        for call in calls:
            if not isinstance(call, dict):
                raise ValueError("Each tool call must be an object")
            if not isinstance(call.get("name"), str) or not call["name"].strip():
                raise ValueError("Tool name must be a nonempty string")
            if not isinstance(call.get("arguments"), dict):
                raise ValueError("Tool arguments must be an object")
            parsed.append(ToolCall(call["name"], call["arguments"]))
        history = raw.get("history", [])
        if not isinstance(history, list) or any(not isinstance(x, dict) for x in history):
            raise ValueError("history must be an array of objects")
        context = raw.get("supplied_context", {})
        if not isinstance(context, dict):
            raise ValueError("supplied_context must be an object")
        return cls(raw["task_id"], raw["user_intent"], content,
                   tuple(parsed), tuple(history), context)
