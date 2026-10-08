from __future__ import annotations

import json
from typing import Any

from veripy_agent.contracts import StepRecord, TaskSpec
from veripy_agent.rollout import (
    AgentAction,
    append_rejected_action,
    append_tool_exchange,
    initial_agent_messages,
    parse_tool_call,
)
from veripy_agent.training.sft import TOOLS


def openai_tools() -> list[dict[str, Any]]:
    return [{"type": "function", "function": item} for item in TOOLS]


def episode_messages(task: TaskSpec, steps: list[StepRecord]) -> list[dict[str, Any]]:
    messages = initial_agent_messages(task)
    for step in steps:
        if step.tool_call is None or step.tool_result is None:
            raw_output = (
                step.assistant_action.get("raw_output")
                if isinstance(step.assistant_action, dict)
                else None
            )
            if raw_output and step.tool_result is not None:
                feedback = step.next_observation or step.tool_result.model_dump(mode="json")
                append_rejected_action(
                    messages,
                    raw_output=raw_output,
                    result=feedback,
                )
            continue
        append_tool_exchange(
            messages,
            name=step.tool_call["name"],
            arguments=step.tool_call["arguments"],
            result=step.tool_result.model_dump(mode="json", exclude_none=True),
        )
    return messages


class TransformersToolAgent:
    def __init__(self, model: Any, tokenizer: Any, *, max_new_tokens: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.outputs: list[str] = []

    def __call__(
        self, task: TaskSpec, *, observation: Any, steps: list[StepRecord]
    ) -> AgentAction | dict[str, str]:
        del observation
        encoded = self.tokenizer.apply_chat_template(
            episode_messages(task, steps),
            tools=openai_tools(),
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        output = self.model.generate(
            **encoded,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        prompt_length = encoded["input_ids"].shape[-1]
        text = self.tokenizer.decode(output[0, prompt_length:], skip_special_tokens=False)
        self.outputs.append(text)
        try:
            return parse_tool_call(text)
        except (json.JSONDecodeError, ValueError) as exc:
            return {"raw_output": text, "parse_error": str(exc)}
