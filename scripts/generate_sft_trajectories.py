#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import yaml

from veripy_agent.contracts import TaskSpec
from veripy_agent.environment import DockerSandboxWorker, EpisodeEnvironment, SandboxPaths
from veripy_agent.rollout import protocol_manifest_from_resources
from veripy_agent.training.oracle_teacher import (
    bind_sandbox_input_paths,
    oracle_smoke_actions,
    oracle_smoke_provenance,
)
from veripy_agent.training.sft import (
    TOOLS,
    TeacherConfig,
    TeacherResponseError,
    admit_trajectory,
    append_jsonl_record,
    assign_trajectory_kinds,
    load_admitted_progress,
    load_teacher_secret_file,
    provenance_for,
    request_teacher_actions,
    write_admitted_trajectories,
    write_dataset_info,
    write_trajectory_manifest,
)
from veripy_agent.verl_adapter import build_rollout_rows_from_corpus


def _load_rows(corpus: Path, lock_path: Path, count: int) -> list[dict]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    rows = build_rollout_rows_from_corpus(corpus, "sft_train", expected_sha256=lock["sha256"])
    selected = []
    seen_families = set()
    for row in rows:
        family = row["ability"]
        if family not in seen_families:
            selected.append(row)
            seen_families.add(family)
        if len(selected) == min(count, 8):
            break
    if count > len(selected):
        selected_ids = {row["env_kwargs"]["task"]["task_id"] for row in selected}
        selected.extend(
            row for row in rows if row["env_kwargs"]["task"]["task_id"] not in selected_ids
        )
    return selected[:count]


def _teacher_input_bundle(task: TaskSpec, input_root: Path) -> dict[str, object]:
    root = input_root.resolve()
    bundle: dict[str, object] = {}
    for relative in task.files:
        source = (root / relative).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError(f"invalid task input path: {relative}")
        if source.stat().st_size > 1_000_000:
            raise ValueError(f"teacher input exceeds 1 MB: {relative}")
        text = source.read_text(encoding="utf-8")
        try:
            bundle[relative] = json.loads(text)
        except json.JSONDecodeError:
            bundle[relative] = text
    return bundle


def _teacher_prompt(task: TaskSpec, kind: str, input_root: Path) -> dict[str, object]:
    requirements = {
        "max_actions": 6,
        "final_tool": "submit_answer",
        "must_execute_python_successfully": True,
        "trajectory_kind": kind,
        "kind_behavior": {
            "direct_success": "Do not include any failing tool call.",
            "recoverable_execution_error": (
                "Include one execute_python call that raises an error, then recover with a "
                "successful execute_python call."
            ),
            "plan_correction": (
                "First inspect a nonexistent input path, observe the failure, then inspect the "
                "real file and recover."
            ),
        }[kind],
    }
    return {
        "task": task.model_dump(mode="json"),
        "input_bundle": _teacher_input_bundle(task, input_root),
        "requirements": requirements,
        "tools": [
            "inspect_files()",
            "inspect_table(path, rows=5)",
            "execute_python(code)",
            "submit_answer(payload)",
        ],
        "python_filesystem": (
            "inspect_table uses task-relative paths such as input.csv; execute_python runs in a "
            "container and must open the same file as /inputs/input.csv."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("oracle_smoke", "openai_compatible"), required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--corpus", type=Path, default=Path("data/manifests/task_corpus.json"))
    parser.add_argument(
        "--corpus-lock", type=Path, default=Path("configs/data/task_corpus.lock.json")
    )
    parser.add_argument("--teacher-config", type=Path)
    parser.add_argument("--teacher-secret-file", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/sft"))
    parser.add_argument("--dataset-name", default="veripy_sft_smoke")
    parser.add_argument("--dataset-file", default="veripy_sft_smoke.jsonl")
    parser.add_argument("--audit-file", default="veripy_sft_smoke.audit.jsonl")
    parser.add_argument("--progress-file")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--resources-lock", type=Path, default=Path("resources.lock.json"))
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument(
        "--sandbox-config", type=Path, default=Path("configs/rl/veripy_agent_grpo.yaml")
    )
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")
    if args.provider == "oracle_smoke" and args.count > 8:
        raise ValueError("oracle_smoke is limited to the eight-sample engineering smoke")
    if args.formal and (
        args.provider != "openai_compatible"
        or args.count != 3000
        or args.dataset_name != "veripy_sft"
        or args.dataset_file != "trajectories.jsonl"
    ):
        raise ValueError(
            "formal SFT requires openai_compatible, count=3000, "
            "dataset_name=veripy_sft, and dataset_file=trajectories.jsonl"
        )

    rows = _load_rows(args.corpus, args.corpus_lock, args.count)
    tasks = [TaskSpec.model_validate(row["env_kwargs"]["task"]) for row in rows]
    kinds = assign_trajectory_kinds(task.task_id for task in tasks)
    sandbox_config = yaml.safe_load(args.sandbox_config.read_text(encoding="utf-8"))["env"]
    teacher = None
    teacher_environ = None
    if args.provider == "openai_compatible":
        if (args.teacher_config is None) == (args.teacher_secret_file is None):
            raise ValueError("provide exactly one of --teacher-config or --teacher-secret-file")
        if args.teacher_secret_file is not None:
            teacher, teacher_environ = load_teacher_secret_file(args.teacher_secret_file)
        else:
            assert args.teacher_config is not None
            teacher = TeacherConfig.model_validate(
                yaml.safe_load(args.teacher_config.read_text(encoding="utf-8"))
            )
        print(
            json.dumps(
                {
                    "teacher_provider": teacher.provider,
                    "teacher_model": teacher.model,
                    "teacher_base_url_host": urlparse(teacher.base_url or "").netloc,
                    "teacher_api_key_present": bool(teacher.resolve_api_key(teacher_environ)),
                },
                sort_keys=True,
            )
        )

    progress_name = args.progress_file or (args.audit_file + ".progress.jsonl")
    progress_path = args.dataset_dir / progress_name
    rejection_path = args.dataset_dir / (args.audit_file + ".rejected.jsonl")
    if not args.resume and (progress_path.exists() or rejection_path.exists()):
        raise RuntimeError("progress exists; use --resume or choose fresh output names")
    admitted = load_admitted_progress(progress_path) if args.resume else []
    accepted_task_ids = {item.task.task_id for item in admitted}
    for row, task in zip(rows, tasks, strict=True):
        if task.task_id in accepted_task_ids:
            continue
        kind = kinds[task.task_id]
        input_root = Path(row["env_kwargs"]["input_root"])
        paths = SandboxPaths(input_root, args.dataset_dir / "scratch" / task.task_id)
        environment = EpisodeEnvironment(
            task,
            paths,
            worker=DockerSandboxWorker(paths, sandbox_config["sandbox_image"]),
        )
        raw_response_path = None
        try:
            if args.provider == "oracle_smoke":
                actions = oracle_smoke_actions(task, kind)
                provenance = oracle_smoke_provenance(task, actions)
            else:
                assert teacher is not None
                prompt = _teacher_prompt(task, kind.value, input_root)
                actions, request_raw, response_raw = request_teacher_actions(
                    teacher, prompt, environ=teacher_environ
                )
                response_sha256 = hashlib.sha256(response_raw).hexdigest()
                raw_response_path = (
                    args.dataset_dir / "raw_teacher" / f"{task.task_id}-{response_sha256[:12]}.json"
                )
                raw_response_path.parent.mkdir(parents=True, exist_ok=True)
                raw_response_path.write_bytes(response_raw)
                actions = bind_sandbox_input_paths(task, actions)
                provenance = provenance_for(
                    teacher,
                    request_raw,
                    response_raw,
                    raw_response_path=str(raw_response_path),
                )
            trajectory = admit_trajectory(
                task=task,
                kind=kind,
                actions=actions,
                environment=environment,
                provenance=provenance,
            )
            append_jsonl_record(progress_path, trajectory)
            admitted.append(trajectory)
            accepted_task_ids.add(task.task_id)
        except Exception as exc:
            if isinstance(exc, TeacherResponseError) and exc.response_raw:
                response_sha256 = hashlib.sha256(exc.response_raw).hexdigest()
                raw_response_path = (
                    args.dataset_dir
                    / "raw_teacher"
                    / f"{task.task_id}-{response_sha256[:12]}-invalid.json"
                )
                raw_response_path.parent.mkdir(parents=True, exist_ok=True)
                raw_response_path.write_bytes(exc.response_raw)
            append_jsonl_record(
                rejection_path,
                {
                    "task_id": task.task_id,
                    "kind": kind.value,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "raw_response_path": str(raw_response_path) if raw_response_path else None,
                },
            )
    if len(admitted) != args.count:
        raise RuntimeError(
            f"admitted {len(admitted)} of {args.count} trajectories; rerun with --resume"
        )
    write_admitted_trajectories(
        admitted,
        audit_path=args.dataset_dir / args.audit_file,
        dataset_path=args.dataset_dir / args.dataset_file,
    )
    write_dataset_info(
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset_name,
        dataset_file=args.dataset_file,
    )
    write_trajectory_manifest(
        admitted,
        path=args.dataset_dir / (args.dataset_file + ".manifest.json"),
        dataset_path=args.dataset_dir / args.dataset_file,
        audit_path=args.dataset_dir / args.audit_file,
        expected_provider=args.provider,
        protocol_receipt=protocol_manifest_from_resources(
            TOOLS,
            resources_lock=args.resources_lock,
            tokenizer_path=args.tokenizer_path,
        ),
    )
    print(
        json.dumps(
            {
                "accepted": len(admitted),
                "dataset": str(args.dataset_dir / args.dataset_file),
                "provider": args.provider,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
