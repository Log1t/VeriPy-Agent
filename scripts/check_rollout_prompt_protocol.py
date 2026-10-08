#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from veripy_agent.data import generate_tasks
from veripy_agent.rollout import (
    append_tool_exchange,
    encode_message_observation,
    initial_agent_messages,
)
from veripy_agent.training.sft import TOOLS
from veripy_agent.verl_adapter import (
    install_agent_prompt_protocol_patch,
    install_auto_tokenizer_compat_patch,
)


def verify_verl_agent_revision(root: Path, resources_lock: Path) -> str:
    lock = json.loads(resources_lock.read_text(encoding="utf-8"))
    expected = next(
        item["revision"]
        for item in lock["resources"]
        if item.get("name") == "verl-agent"
    )
    actual = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected:
        raise RuntimeError(
            f"verl-agent revision mismatch: expected {expected}, found {actual}"
        )
    return actual


def load_trajectory_collector(root: Path) -> type[Any]:
    checkout = root.resolve()
    if not (checkout / "agent_system" / "multi_turn_rollout" / "rollout_loop.py").is_file():
        raise RuntimeError(f"verl-agent checkout is incomplete: {checkout}")
    checkout_text = str(checkout)
    if checkout_text not in sys.path:
        sys.path.insert(0, checkout_text)
    module = importlib.import_module("agent_system.multi_turn_rollout.rollout_loop")
    return cast(type[Any], module.TrajectoryCollector)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--verl-agent-root", type=Path, default=Path("external/verl-agent"))
    parser.add_argument("--resources-lock", type=Path, default=Path("resources.lock.json"))
    args = parser.parse_args()
    if args.turns < 0:
        parser.error("--turns must be non-negative")
    if args.max_prompt_length <= 0:
        parser.error("--max-prompt-length must be positive")
    verl_agent_revision = verify_verl_agent_revision(
        args.verl_agent_root, args.resources_lock
    )
    trajectory_collector = load_trajectory_collector(args.verl_agent_root)

    install_auto_tokenizer_compat_patch()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tools = [{"type": "function", "function": item} for item in TOOLS]
    task = generate_tasks("datetime_timeseries", seed=41000, count=1, split="validation")[0].task_spec
    messages = initial_agent_messages(task)
    for turn in range(args.turns):
        append_tool_exchange(
            messages,
            name="execute_python",
            arguments={"code": f"print('turn {turn + 1}')"},
            result={
                "status": "ok",
                "stdout": f"turn {turn + 1}\n",
                "stderr": "",
                "artifacts": [],
                "duration": 0.01,
                "resource_usage": {"cpu_seconds": 0.01},
            },
        )
    observation = encode_message_observation(messages)

    collector = cast(Any, object.__new__(trajectory_collector))
    collector.tokenizer = tokenizer
    collector.processor = None
    collector.config = OmegaConf.create(
        {
            "data": {
                "max_prompt_length": args.max_prompt_length,
                "truncation": "error",
                "return_raw_chat": True,
            }
        }
    )
    install_agent_prompt_protocol_patch(trajectory_collector, tools)
    batch = SimpleNamespace(
        non_tensor_batch={
            "raw_prompt": [[{"role": "user", "content": "unused"}]],
            "data_source": ["veripy_bench"],
        }
    )
    row = collector.preprocess_single_sample(
        item=0,
        gen_batch=batch,
        obs={"text": [observation], "image": None, "anchor": None},
    )

    rendered = tokenizer.apply_chat_template(
        messages,
        tools=cast(list[dict[Any, Any] | Callable[..., Any]], tools),
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )
    expected_ids = tokenizer.encode(rendered, add_special_tokens=False)
    if row["raw_prompt_ids"] != expected_ids:
        raise RuntimeError("collector prompt differs from the canonical SFT message prompt")
    if len(expected_ids) > args.max_prompt_length:
        raise RuntimeError(
            f"prompt uses {len(expected_ids)} tokens, exceeding "
            f"max_prompt_length={args.max_prompt_length}"
        )
    print(
        json.dumps(
            {
                "collector_prompt_matches": True,
                "max_prompt_length": args.max_prompt_length,
                "prompt_tokens": len(expected_ids),
                "roles": [message["role"] for message in messages],
                "turns": args.turns,
                "verl_agent_revision": verl_agent_revision,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
