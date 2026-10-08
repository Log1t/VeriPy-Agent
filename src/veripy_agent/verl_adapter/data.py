from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from veripy_agent.contracts import TaskSpec
from veripy_agent.data import validate_task_bundle
from veripy_agent.rollout.messages import (
    decode_message_observation,
    encode_message_observation,
    initial_agent_messages,
)


def build_rollout_row(task: TaskSpec, *, input_root: str) -> dict[str, Any]:
    # Keep the RL first observation aligned with the SFT canonical protocol.
    initial_observation = encode_message_observation(
        initial_agent_messages(task)
    )
    row = {
        "data_source": "veripy_bench",
        "prompt": [{"role": "user", "content": initial_observation}],
        "ability": task.family,
        "reward_model": {"style": "episode"},
        "extra_info": {"task_id": task.task_id, "split": task.split},
        "metadata": {"difficulty": task.difficulty},
        "env_kwargs": {
            "task": task.model_dump(mode="json"),
            "input_root": input_root,
        },
    }
    validate_rollout_row(row)
    return row


def validate_rollout_row(row: dict[str, Any]) -> None:
    required = {
        "data_source",
        "prompt",
        "ability",
        "reward_model",
        "extra_info",
        "metadata",
        "env_kwargs",
    }
    missing = required - row.keys()
    if missing:
        raise ValueError(f"rollout row missing fields: {sorted(missing)}")
    if not isinstance(row["prompt"], list) or not row["prompt"]:
        raise ValueError("prompt must be a non-empty message list")
    if not row["env_kwargs"].get("input_root"):
        raise ValueError("env_kwargs.input_root must be explicit")
    task = TaskSpec.model_validate(row["env_kwargs"]["task"])
    prompt = row["prompt"]
    if len(prompt) != 1 or prompt[0].get("role") != "user":
        raise ValueError("rollout prompt does not match the public task contract")
    messages = decode_message_observation(prompt[0]["content"])
    canonical = initial_agent_messages(task)
    if messages != canonical:
        raise ValueError("rollout prompt does not match the public task contract")


def build_rollout_rows_from_corpus(
    corpus_path: str | Path,
    split: str,
    *,
    expected_sha256: str,
    dataset_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    corpus_path = Path(corpus_path)
    resolved_dataset_root = Path(dataset_root).resolve() if dataset_root is not None else None
    actual_corpus_hash = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    if actual_corpus_hash != expected_sha256:
        raise ValueError("corpus manifest hash mismatch")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    config_path = Path(corpus["config"])
    if not config_path.is_file():
        raise ValueError(f"corpus generator config is unavailable: {config_path}")
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != corpus["config_sha256"]:
        raise ValueError("corpus generator config hash mismatch")
    rows = []
    for bundle in corpus["bundles"]:
        if bundle["split"] != split:
            continue
        manifest_path = Path(bundle["manifest"])
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != bundle["manifest_sha256"]:
            raise ValueError(f"bundle manifest hash mismatch: {manifest_path}")
        manifest = validate_task_bundle(manifest_path)
        expected_metadata = {
            "split": manifest["split"],
            "family": manifest["families"][0],
            "count": manifest["task_count"],
            "seed": manifest["seed"],
        }
        if any(bundle.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError(f"corpus bundle metadata mismatch: {manifest_path}")
        for record in manifest["_records"]:
            task = TaskSpec.model_validate(record["task_spec"])
            input_files = [manifest_path.parent / relative for relative in task.files]
            input_roots = {path.parent.resolve() for path in input_files}
            if len(input_roots) != 1:
                raise ValueError("all files for an episode must share one input directory")
            input_root = input_roots.pop()
            if resolved_dataset_root is not None:
                try:
                    input_root_reference = input_root.relative_to(resolved_dataset_root)
                except ValueError as exc:
                    raise ValueError(
                        f"episode input root escapes configured dataset root: {input_root}"
                    ) from exc
            else:
                input_root_reference = input_root
            episode_task = task.model_copy(
                update={"files": [path.name for path in input_files]}
            )
            rows.append(
                build_rollout_row(episode_task, input_root=str(input_root_reference))
            )
    if not rows:
        raise ValueError(f"no tasks found for split {split}")
    return rows
