from __future__ import annotations

from functools import wraps
from typing import Any

from veripy_agent.rollout import decode_message_observation
from veripy_agent.rollout.messages import SYSTEM_PROMPT


def protocol_messages(messages: Any, tools: list[dict[str, Any]]) -> tuple[Any, dict[str, Any]]:
    """Shared renderer arguments for collection and exact context budgeting."""
    chat = list(messages)
    if len(chat) == 1 and chat[0].get("role") == "user":
        content = chat[0].get("content")
        decoded = decode_message_observation(content) if isinstance(content, str) else None
        if decoded is not None:
            chat = decoded
        else:
            chat.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        return chat, {"tools": tools, "enable_thinking": False}
    return chat, {}


def install_agent_prompt_protocol_patch(
    trajectory_collector_class: Any, tools: list[dict[str, Any]]
) -> Any:
    """Make the legacy collector render prompts with the SFT tool protocol."""

    installed = getattr(trajectory_collector_class, "_veripy_prompt_protocol_patch", None)
    if installed is not None:
        return installed

    original = trajectory_collector_class.preprocess_single_sample

    @wraps(original)
    def patched(self: Any, *args: Any, **kwargs: Any) -> Any:
        apply_chat_template = self.tokenizer.apply_chat_template

        def render(messages: Any, *render_args: Any, **render_kwargs: Any) -> Any:
            chat, defaults = protocol_messages(messages, tools)
            for key, value in defaults.items():
                render_kwargs.setdefault(key, value)
            return apply_chat_template(chat, *render_args, **render_kwargs)

        self.tokenizer.apply_chat_template = render
        try:
            return original(self, *args, **kwargs)
        finally:
            self.tokenizer.apply_chat_template = apply_chat_template

    trajectory_collector_class.preprocess_single_sample = patched
    trajectory_collector_class._veripy_prompt_protocol_patch = original
    return original
