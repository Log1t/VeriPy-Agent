"""Canonical protocol receipt and hidden-oracle leakage checks."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .messages import MESSAGE_PROTOCOL, SYSTEM_PROMPT

HIDDEN_KEYS = frozenset(
    {
        "answer_key",
        "expected",
        "gold_answer",
        "hidden_answer",
        "oracle",
        "verifier_diagnostics",
    }
)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def tool_schema_sha256(tools: Sequence[Mapping[str, Any]]) -> str:
    return canonical_json_sha256(list(tools))


def protocol_manifest(
    tools: Sequence[Mapping[str, Any]],
    *,
    tokenizer_revision: str,
    chat_template_sha256: str,
    max_turns: int = 6,
    enable_thinking: bool = False,
    stop_policy: str = "eos_or_complete_tool_call",
) -> dict[str, Any]:
    if max_turns < 1:
        raise ValueError("max_turns must be positive")
    manifest = {
        "schema": "veripy.agent.protocol",
        "message_protocol": MESSAGE_PROTOCOL,
        "system_prompt_sha256": canonical_json_sha256(SYSTEM_PROMPT),
        "tools_sha256": tool_schema_sha256(tools),
        "tokenizer_revision": tokenizer_revision,
        "chat_template_sha256": chat_template_sha256,
        "enable_thinking": enable_thinking,
        "add_generation_prompt": True,
        "stop_policy": stop_policy,
        "max_turns": max_turns,
        "roles": ["system", "user", "assistant.tool_calls", "tool"],
        "recovery_roles": ["assistant.raw", "user.recovery"],
    }
    manifest["protocol_sha256"] = canonical_json_sha256(manifest)
    return manifest


def protocol_manifest_from_resources(
    tools: Sequence[Mapping[str, Any]],
    *,
    resources_lock: str | Path,
    tokenizer_path: str | Path | None = None,
    max_turns: int = 6,
) -> dict[str, Any]:
    lock = json.loads(Path(resources_lock).read_text(encoding="utf-8"))
    model = next(
        (item for item in lock.get("resources", []) if item.get("type") == "model"),
        None,
    )
    if model is None or not model.get("revision"):
        raise ValueError("resource lock is missing a pinned model revision")
    configured_path = tokenizer_path or os.environ.get(str(model.get("local_path_env", "")))
    configured_path = configured_path or model.get("local_path")
    root = Path(str(configured_path)).resolve()
    template_path = root / "chat_template.jinja"
    if template_path.is_file():
        template = template_path.read_text(encoding="utf-8")
    else:
        tokenizer_config = json.loads((root / "tokenizer_config.json").read_text(encoding="utf-8"))
        template = tokenizer_config.get("chat_template")
        if not isinstance(template, str) or not template:
            raise ValueError(f"tokenizer chat template is missing: {root}")
    return protocol_manifest(
        tools,
        tokenizer_revision=str(model["revision"]),
        chat_template_sha256=hashlib.sha256(template.encode()).hexdigest(),
        max_turns=max_turns,
    )


def hidden_leak_paths(value: Any, path: str = "$") -> list[str]:
    leaks: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            if str(key).lower() in HIDDEN_KEYS:
                leaks.append(child)
            leaks.extend(hidden_leak_paths(item, child))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            leaks.extend(hidden_leak_paths(item, f"{path}[{index}]"))
    return leaks


def assert_no_hidden_leakage(value: Any) -> None:
    leaks = hidden_leak_paths(value)
    if leaks:
        raise ValueError(f"hidden verifier data leaked into public payload: {leaks}")
