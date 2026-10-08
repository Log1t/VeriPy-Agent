from __future__ import annotations

import json
from typing import Any

from veripy_agent.contracts import TaskSpec

SYSTEM_PROMPT = (
    "You are a verifiable data-analysis agent. Use tools, recover from errors, "
    "and submit structured evidence."
)
MESSAGE_PROTOCOL = "veripy.agent.messages"


def _public_schema(value: Any) -> Any:
    # Parquet widens heterogeneous schema structs with null fields. Those
    # fields were absent in the source task and must not change model prompts.
    if isinstance(value, dict):
        return {key: _public_schema(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_public_schema(item) for item in value]
    return value


def task_message_content(task: TaskSpec) -> str:
    return json.dumps(
        {
            "instruction": task.instruction,
            "files": task.files,
            "answer_schema": _public_schema(task.answer_schema),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def initial_agent_messages(task: TaskSpec) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": task_message_content(task),
        },
    ]


def append_tool_exchange(
    messages: list[dict[str, Any]],
    *,
    name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
) -> None:
    messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "content": json.dumps(
                result, ensure_ascii=False, separators=(",", ":")
            ),
        }
    )


def append_rejected_action(
    messages: list[dict[str, Any]],
    *,
    raw_output: str,
    result: dict[str, Any],
) -> None:
    messages.append({"role": "assistant", "content": raw_output})
    messages.append(
        {
            "role": "user",
            "content": (
                "The previous tool call was rejected. "
                f"Observation: {json.dumps(result, ensure_ascii=False)}\n"
                "Return one complete tool call with valid JSON."
            ),
        }
    )


def encode_message_observation(messages: list[dict[str, Any]]) -> str:
    return json.dumps(
        {"protocol": MESSAGE_PROTOCOL, "messages": messages},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def decode_message_observation(content: str) -> list[dict[str, Any]] | None:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or payload.get("protocol") != MESSAGE_PROTOCOL:
        return None
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("VeriPy message observation requires a non-empty messages list")
    _validate_message_sequence(messages)
    return messages


def _validate_message_sequence(messages: list[Any]) -> None:
    if len(messages) < 2:
        raise ValueError("VeriPy message observation is missing the initial task")
    if messages[0] != {"role": "system", "content": SYSTEM_PROMPT}:
        raise ValueError("VeriPy message observation has an invalid system prompt")
    initial_user = messages[1]
    if (
        not isinstance(initial_user, dict)
        or set(initial_user) != {"role", "content"}
        or initial_user.get("role") != "user"
        or not isinstance(initial_user.get("content"), str)
    ):
        raise ValueError("VeriPy message observation has an invalid task message")
    index = 2
    while index < len(messages):
        assistant = messages[index]
        if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
            raise ValueError("VeriPy message history must alternate from assistant")
        if "tool_calls" in assistant:
            if set(assistant) != {"role", "tool_calls"}:
                raise ValueError("VeriPy assistant tool message has unexpected fields")
            calls = assistant["tool_calls"]
            if (
                not isinstance(calls, list)
                or len(calls) != 1
                or not isinstance(calls[0], dict)
                or calls[0].get("type") != "function"
                or not isinstance(calls[0].get("function"), dict)
                or not isinstance(calls[0]["function"].get("name"), str)
                or not isinstance(calls[0]["function"].get("arguments"), str)
            ):
                raise ValueError("VeriPy assistant tool call is invalid")
            expected_role = "tool"
        else:
            if set(assistant) != {"role", "content"} or not isinstance(
                assistant.get("content"), str
            ):
                raise ValueError("VeriPy rejected assistant message is invalid")
            expected_role = "user"
        if index + 1 >= len(messages):
            raise ValueError("VeriPy message history ends without an observation")
        observation = messages[index + 1]
        if (
            not isinstance(observation, dict)
            or set(observation) != {"role", "content"}
            or observation.get("role") != expected_role
            or not isinstance(observation.get("content"), str)
        ):
            raise ValueError("VeriPy message history contains an invalid observation")
        index += 2
