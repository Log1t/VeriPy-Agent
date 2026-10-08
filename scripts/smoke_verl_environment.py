#!/usr/bin/env python3
"""Run a CPU-only VeriPy vector-environment integration smoke."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from veripy_agent.environment import HardenedProcessWorker
from veripy_agent.training.public_replay import load_public_bundle
from veripy_agent.verifiers import verify_public_submission
from veripy_agent.verl_adapter.vector_env import VeriPyVectorEnv


def _load_public_episodes(bundle: Path, actions_path: Path, count: int):
    records = load_public_bundle(bundle)
    actions = {
        item["task_id"]: item["actions"]
        for line in actions_path.read_text().splitlines()
        if line
        for item in [json.loads(line)]
    }
    selected = []
    for record in records:
        if record.task.task_id not in actions:
            continue
        selected.append((record, actions[record.task.task_id]))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("public bundle does not contain enough admitted episodes")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("artifacts/public_finqa_program_freeze/finqa/train"),
    )
    parser.add_argument(
        "--actions",
        type=Path,
        default=Path("artifacts/public_finqa_program_freeze/actions.jsonl"),
    )
    parser.add_argument("--count", type=int, default=2)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    actions_path = args.actions.resolve()
    episodes = _load_public_episodes(bundle, actions_path, args.count)
    expected_by_id = {item.task.task_id: item.expected for item, _ in episodes}

    def verifier_factory(task):
        expected = expected_by_id[task.task_id]
        return lambda current, submission: verify_public_submission(
            current, submission, expected
        )
    with tempfile.TemporaryDirectory(prefix="veripy-vector-smoke-", dir="/tmp") as tmp:
        root = Path(tmp)
        root.chmod(0o755)
        tasks = [
            (
                record.task.model_dump(mode="json"),
                record.input_root,
                action_list,
            )
            for record, action_list in episodes
        ]
        environment = VeriPyVectorEnv(
            worker_factory=lambda paths: HardenedProcessWorker(paths, timeout=5),
            dataset_root=bundle,
            scratch_root=root / "scratch",
            max_turns=args.max_turns,
            submission_verifier_factory=verifier_factory,
        )
        try:
            environment.reset(
                kwargs=[
                    {"task": task, "input_root": str(input_root)}
                    for task, input_root, _ in tasks
                ]
            )
            batches = [["not-json"] * len(tasks)]
            for action_index in range(2):
                batches.append(
                    [
                        json.dumps(action_list[action_index], sort_keys=True)
                        for _, _, action_list in tasks
                    ]
                )
            receipts = []
            for step, batch in enumerate(batches, start=1):
                _, rewards, dones, infos = environment.step(batch)
                receipts.append(
                    {
                        "step": step,
                        "rewards": list(rewards),
                        "dones": list(dones),
                        "infos": infos,
                    }
                )
            result = {
                "invalid_action_recovered": (
                    receipts[0]["infos"][0]["is_action_valid"] is False
                    and receipts[1]["infos"][0]["is_action_valid"]
                ),
                "both_won": all(item["won"] for item in receipts[-1]["infos"]),
                "both_done": receipts[-1]["dones"] == [True, True],
                "final_rewards": receipts[-1]["rewards"],
                "turn_counts": [item["turn_count"] for item in receipts[-1]["infos"]],
                "tool_call_counts": [item["tool_call_count"] for item in receipts[-1]["infos"]],
                "source": "FinQA public replay-admitted bundle",
            }
            print(json.dumps(result, sort_keys=True))
            return int(not all(result[key] for key in ("invalid_action_recovered", "both_won", "both_done")))
        finally:
            environment.close()


if __name__ == "__main__":
    raise SystemExit(main())
