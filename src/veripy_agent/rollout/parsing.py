from __future__ import annotations

import json
import re

from .runner import AgentAction

_THINK_PREFIX = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def parse_tool_call(text: str) -> AgentAction:
    """Parse the Qwen tool marker or the environment's canonical JSON action."""

    candidate = _THINK_PREFIX.sub("", text, count=1).strip()
    if candidate.endswith("<|im_end|>"):
        candidate = candidate[: -len("<|im_end|>")].rstrip()
    if candidate.startswith("<tool_call>"):
        if not candidate.endswith("</tool_call>"):
            raise ValueError("model output contains an incomplete tool_call block")
        payload = candidate[len("<tool_call>") : -len("</tool_call>")].strip()
    else:
        payload = candidate
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("tool call payload must be a JSON object")
    keys = set(value)
    if keys == {"name", "arguments"}:
        value = {"tool": value["name"], "arguments": value["arguments"]}
    elif keys != {"tool", "arguments"}:
        raise ValueError("tool call must contain only tool/name and arguments")
    return AgentAction.model_validate(value)
