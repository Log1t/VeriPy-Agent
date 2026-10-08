#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from veripy_agent.contracts import Submission, TaskSpec, VerifyResult
from veripy_agent.environment import (
    DockerSandboxWorker,
    EpisodeEnvironment,
    HardenedProcessWorker,
    SandboxPaths,
)
from veripy_agent.evaluation.model_agent import TransformersToolAgent, episode_messages
from veripy_agent.rollout import assert_no_hidden_leakage, protocol_manifest_from_resources
from veripy_agent.training.public_replay import PublicBundleTask, load_public_bundle
from veripy_agent.training.sft import TOOLS
from veripy_agent.verifiers import verify_public_submission, verify_task_submission
from veripy_agent.verl_adapter import (
    build_rollout_rows_from_corpus,
    install_auto_tokenizer_compat_patch,
)


def _family_sample(rows: list[dict], count: int) -> list[dict]:
    selected = []
    seen = set()
    for row in rows:
        if row["ability"] in seen:
            continue
        selected.append(row)
        seen.add(row["ability"])
        if len(selected) == count:
            break
    return selected


def _select_rows(rows: list[dict], *, count: int, families: list[str]) -> list[dict]:
    if not families:
        return _family_sample(rows, count)
    requested = set(families)
    selected = _family_sample(
        [row for row in rows if row["ability"] in requested], len(requested)
    )
    found = {row["ability"] for row in selected}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"requested families missing from corpus: {', '.join(missing)}")
    return selected


def _select_manifest_rows(
    rows: list[dict], manifest_path: Path, *, corpus_sha256: str
) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "veripy.paired_validation_selection":
        raise ValueError("invalid validation selection schema")
    if manifest.get("split") != "validation" or manifest.get("corpus_sha256") != corpus_sha256:
        raise ValueError("validation selection does not match locked corpus")
    task_ids = manifest.get("task_ids")
    if not isinstance(task_ids, list) or not task_ids or any(not isinstance(x, str) for x in task_ids):
        raise ValueError("validation selection must contain task IDs")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate validation task IDs")
    digest = hashlib.sha256(json.dumps(task_ids, separators=(",", ":")).encode()).hexdigest()
    if manifest.get("count") != len(task_ids) or manifest.get("selection_sha256") != digest:
        raise ValueError("validation selection count or hash mismatch")
    by_id = {row["extra_info"]["task_id"]: row for row in rows}
    missing = sorted(set(task_ids) - by_id.keys())
    if missing:
        raise ValueError(f"validation task IDs missing from corpus: {missing}")
    return [by_id[task_id] for task_id in task_ids]


def _select_public_tasks(
    bundle: Path, *, count: int, task_ids: list[str]
) -> list[PublicBundleTask]:
    tasks = load_public_bundle(bundle)
    by_id = {item.task.task_id: item for item in tasks}
    if task_ids:
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("duplicate public task IDs")
        missing = sorted(set(task_ids) - by_id.keys())
        if missing:
            raise ValueError(f"public task IDs missing from bundle: {missing}")
        return [by_id[task_id] for task_id in task_ids]
    if count < 1:
        raise ValueError("public evaluation count must be positive")
    return tasks[:count]


def _validate_adapter_base(
    adapter: Path, model: Path, *, source_alias: str | None = None
) -> None:
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    configured_base = str(config["base_model_name_or_path"])
    if Path(configured_base).resolve() != model.resolve() and configured_base != source_alias:
        raise ValueError(
            f"adapter base model mismatch: expected {model.resolve()}, "
            f"found {configured_base}"
        )


def _public_verifier_for(
    task_id: str, expected: Any
) -> Callable[[TaskSpec, Submission], tuple[VerifyResult, bool]]:
    def verifier(candidate: TaskSpec, submission: Submission) -> tuple[VerifyResult, bool]:
        if candidate.task_id != task_id:
            raise ValueError("public evaluation task identity changed")
        return verify_public_submission(candidate, submission, expected)

    return verifier


def _artifact_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"artifact directory contains no files: {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--adapter-base-alias", type=str)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resources-lock", type=Path, default=Path("resources.lock.json"))
    parser.add_argument("--scratch-root", type=Path, default=Path("artifacts/evaluation/agent"))
    parser.add_argument(
        "--sandbox-backend", choices=("docker", "hardened_process"), default="docker"
    )
    parser.add_argument("--sandbox-user", default="veripy-sandbox")
    parser.add_argument("--corpus", type=Path, default=Path("data/manifests/task_corpus.json"))
    parser.add_argument("--public-bundle", type=Path)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument(
        "--corpus-lock", type=Path, default=Path("configs/data/task_corpus.lock.json")
    )
    parser.add_argument(
        "--sandbox-config", type=Path, default=Path("configs/rl/veripy_agent_grpo.yaml")
    )
    args = parser.parse_args()

    if args.public_bundle is not None:
        if args.selection_manifest is not None:
            raise ValueError("--selection-manifest is only for generated validation")
        if args.family:
            raise ValueError("--family cannot be combined with --public-bundle")
        public_tasks = _select_public_tasks(
            args.public_bundle, count=args.count, task_ids=args.task_id
        )
        selected = [
            (item.task, item.input_root, item.expected) for item in public_tasks
        ]
        source_path = args.public_bundle / "manifest.json"
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        source_kind = "public_bundle"
    else:
        if args.task_id:
            raise ValueError("--task-id requires --public-bundle")
        if args.selection_manifest is not None and args.family:
            raise ValueError("--family cannot be combined with --selection-manifest")
        lock = json.loads(args.corpus_lock.read_text(encoding="utf-8"))
        rows = build_rollout_rows_from_corpus(
            args.corpus, "validation", expected_sha256=lock["sha256"]
        )
        chosen_rows = (
            _select_manifest_rows(rows, args.selection_manifest, corpus_sha256=lock["sha256"])
            if args.selection_manifest is not None
            else _select_rows(rows, count=args.count, families=args.family)
        )
        if args.selection_manifest is not None:
            if args.count > len(chosen_rows):
                raise ValueError("--count exceeds validation selection size")
            chosen_rows = chosen_rows[:args.count]
        selected = [
            (
                TaskSpec.model_validate(row["env_kwargs"]["task"]),
                Path(row["env_kwargs"]["input_root"]),
                None,
            )
            for row in chosen_rows
        ]
        source_path = args.corpus
        source_hash = lock["sha256"]
        source_kind = "generated_validation"
    sandbox = yaml.safe_load(args.sandbox_config.read_text(encoding="utf-8"))["env"]
    if args.sandbox_backend == "hardened_process":
        report = HardenedProcessWorker.capability_report(user=args.sandbox_user)
        if not report.available:
            raise RuntimeError(f"hardened process sandbox unavailable: {report.reason}")
    if args.adapter is not None:
        _validate_adapter_base(args.adapter, args.model, source_alias=args.adapter_base_alias)
    if args.dry_run:
        protocol = protocol_manifest_from_resources(
            TOOLS,
            resources_lock=args.resources_lock,
            tokenizer_path=args.model,
            max_turns=6,
        )
        print(
            json.dumps(
                {
                    "source_kind": source_kind,
                    "source_sha256": source_hash,
                    "task_ids": [task.task_id for task, _, _ in selected],
                    "model": str(args.model),
                    "adapter": str(args.adapter) if args.adapter is not None else None,
                    "adapter_base_alias": args.adapter_base_alias,
                    "selection_manifest_sha256": (
                        hashlib.sha256(args.selection_manifest.read_bytes()).hexdigest()
                        if args.selection_manifest is not None else None
                    ),
                    "protocol_sha256": protocol["protocol_sha256"],
                    "sandbox_backend": args.sandbox_backend,
                },
                sort_keys=True,
            )
        )
        return
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    install_auto_tokenizer_compat_patch()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model: Any = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    records: list[dict[str, Any]] = []
    for task, input_root, expected in selected:
        paths = SandboxPaths(
            input_root,
            args.scratch_root / task.task_id,
        )
        worker = (
            DockerSandboxWorker(paths, image=sandbox["sandbox_image"])
            if args.sandbox_backend == "docker"
            else HardenedProcessWorker(paths, user=args.sandbox_user)
        )
        environment = EpisodeEnvironment(
            task,
            paths,
            worker=worker,
            submission_verifier=(
                _public_verifier_for(task.task_id, expected)
                if source_kind == "public_bundle"
                else verify_task_submission
            ),
        )
        agent = TransformersToolAgent(model, tokenizer, max_new_tokens=args.max_new_tokens)
        result, reward = environment.run(agent)
        valid_calls = sum(step.tool_call is not None for step in result.steps)
        records.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "turns": len(result.steps),
                "valid_calls": valid_calls,
                "submitted": result.submitted,
                "termination_reason": result.termination_reason,
                "correct": reward.components.get("correctness") == 1.0,
                "evidence_consistent": reward.components.get("evidence") == 0.2,
                "reward": reward.model_dump(mode="json"),
                "outputs": agent.outputs,
                "steps": [step.model_dump(mode="json") for step in result.steps],
                "messages": episode_messages(task, result.steps),
            }
        )
    task_ids = [item["task_id"] for item in records]
    selection_sha256 = hashlib.sha256(
        json.dumps(task_ids, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "model": str(args.model),
        "model_sha256": _artifact_sha256(args.model),
        "adapter": str(args.adapter) if args.adapter is not None else None,
        "adapter_base_alias": args.adapter_base_alias,
        "adapter_sha256": (
            _artifact_sha256(args.adapter) if args.adapter is not None else None
        ),
        "sandbox_backend": args.sandbox_backend,
        "source_kind": source_kind,
        "source_path": str(source_path),
        "corpus_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "expected_corpus_sha256": source_hash,
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "seed": 0,
        },
        "task_ids": task_ids,
        "selection_sha256": selection_sha256,
        "selection_manifest_sha256": (
            hashlib.sha256(args.selection_manifest.read_bytes()).hexdigest()
            if args.selection_manifest is not None else None
        ),
        "count": len(records),
        "tool_call_parse_rate": sum(item["valid_calls"] for item in records)
        / max(1, sum(item["turns"] for item in records)),
        "submission_schema_rate": sum(item["submitted"] for item in records) / max(1, len(records)),
        "active_termination_rate": sum(
            item["termination_reason"] == "submitted" for item in records
        )
        / max(1, len(records)),
        "task_success_rate": sum(item["correct"] for item in records) / max(1, len(records)),
        "evidence_consistency_rate": sum(item["evidence_consistent"] for item in records)
        / max(1, len(records)),
        "protocol": protocol_manifest_from_resources(
            TOOLS,
            resources_lock=args.resources_lock,
            tokenizer_path=args.model,
            max_turns=6,
        ),
    }
    assert_no_hidden_leakage({"summary": summary, "records": records})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
