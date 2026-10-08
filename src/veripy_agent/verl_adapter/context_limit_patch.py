"""Opt-in context-budget termination at the real collector/environment seam."""

from __future__ import annotations

from functools import wraps
from typing import Any

from .prompt_patch import protocol_messages


def observation_tokens(tokenizer: Any, observation: str, tools: list[dict[str, Any]]) -> int:
    messages, options = protocol_messages([{"role": "user", "content": observation}], tools)
    rendered = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, **options
    )
    # Matches verl_F.tokenize_and_postprocess_data, including special tokens.
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


class ContextBudgetEnvironment:
    def __init__(self, manager: Any, tokenizer: Any, tools: list[dict[str, Any]], limit: int):
        self.manager = manager
        self.tokenizer = tokenizer
        self.tools = tools
        self.limit = limit

    def __getattr__(self, name: str) -> Any:
        return getattr(self.manager, name)

    def reset(self, **kwargs: Any) -> Any:
        observations, infos = self.manager.reset(**kwargs)
        for text in observations["text"]:
            if observation_tokens(self.tokenizer, text, self.tools) > self.limit:
                self.manager.envs.close()
                raise ValueError("context limit at bootstrap: no policy action to train on")
        return observations, infos

    def step(self, actions: list[str]) -> Any:
        observations, rewards, dones, infos = self.manager.step(actions)
        # Only collector-local inactive placeholders; real observations and
        # environment histories are retained, not truncated or rewritten.
        observations = dict(observations)
        observations["text"] = list(observations["text"])
        dones = dones.copy()
        infos = [dict(info) for info in infos]
        for i, text in enumerate(observations["text"]):
            if not dones[i]:
                count = observation_tokens(self.tokenizer, text, self.tools)
                if count > self.limit:
                    self.manager.envs.terminate_context_limit(
                        i, prompt_tokens=count, limit=self.limit
                    )
                    dones[i] = True
                    infos[i].update(
                        context_limit=True,
                        context_prompt_tokens=count,
                        context_prompt_limit=self.limit,
                    )
            if dones[i]:
                observations["text"][i] = "episode already terminated"
        return observations, rewards, dones, infos


def install_context_limit_patch(collector_class: Any, tools: list[dict[str, Any]]) -> Any:
    installed = getattr(collector_class, "_veripy_context_limit_patch", None)
    if installed is not None:
        return installed
    original = collector_class.vanilla_multi_turn_loop

    @wraps(original)
    def wrapped(self: Any, gen_batch: Any, actor_rollout_wg: Any, envs: Any) -> Any:
        limit = min(
            int(self.config.data.max_prompt_length),
            int(self.config.actor_rollout_ref.rollout.max_model_len)
            - int(self.config.data.max_response_length),
        )
        if limit <= 0:
            raise ValueError("context budget must reserve response tokens")
        return original(
            self,
            gen_batch,
            actor_rollout_wg,
            ContextBudgetEnvironment(envs, self.tokenizer, tools, limit),
        )

    collector_class.vanilla_multi_turn_loop = wrapped
    collector_class._veripy_context_limit_patch = original
    return original
