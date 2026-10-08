"""Fail-closed loading and sandbox replay for public training trajectories."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from veripy_agent.contracts import Submission, TaskSpec
from veripy_agent.environment import EpisodeEnvironment, SandboxPaths
from veripy_agent.environment.worker import PythonWorker
from veripy_agent.rollout import AgentAction
from veripy_agent.verifiers import verify_public_submission

from .sft import AdmittedTrajectory, TeacherProvenance, TrajectoryKind, admit_trajectory


@dataclass(frozen=True)
class PublicBundleTask:
    task: TaskSpec
    input_root: Path
    expected: Any
    answer_type: str
    program: str | None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def load_public_bundle(root: str | Path) -> list[PublicBundleTask]:
    bundle_root = Path(root).resolve()
    manifest_path = bundle_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "veripy.public_task_bundle":
        raise ValueError("unsupported public task bundle schema")
    for relative, expected_hash in manifest.get("files", {}).items():
        path = (bundle_root / relative).resolve()
        if not path.is_relative_to(bundle_root) or not path.is_file():
            raise ValueError(f"public bundle file is missing or escapes root: {relative}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"public bundle hash mismatch: {relative}")

    public_rows = _jsonl(bundle_root / str(manifest["public_records"]))
    oracle_rows = _jsonl(bundle_root / str(manifest["private_oracle"]))
    oracle_by_id: dict[str, dict[str, Any]] = {}
    for row in oracle_rows:
        task_id = str(row["task_id"])
        if task_id in oracle_by_id:
            raise ValueError(f"duplicate private oracle task: {task_id}")
        oracle_by_id[task_id] = row

    tasks: list[PublicBundleTask] = []
    seen: set[str] = set()
    for row in public_rows:
        task = TaskSpec.model_validate(row["task_spec"])
        if task.task_id != row.get("task_id") or task.task_id in seen:
            raise ValueError(f"public task identity mismatch or duplicate: {task.task_id}")
        seen.add(task.task_id)
        oracle = oracle_by_id.pop(task.task_id, None)
        if oracle is None:
            raise ValueError(f"missing private oracle for task: {task.task_id}")
        input_root = (bundle_root / str(row["input_root"])).resolve()
        if not input_root.is_relative_to(bundle_root) or not input_root.is_dir():
            raise ValueError(f"public task input root escapes bundle: {task.task_id}")
        for relative in task.files:
            candidate = (input_root / relative).resolve()
            if not candidate.is_relative_to(input_root) or not candidate.is_file():
                raise ValueError(f"public task input is missing or escapes root: {relative}")
        tasks.append(
            PublicBundleTask(
                task=task,
                input_root=input_root,
                expected=oracle["answer"],
                answer_type=str(oracle["answer_type"]),
                program=oracle.get("program"),
            )
        )
    if oracle_by_id:
        raise ValueError(f"orphan private oracle tasks: {sorted(oracle_by_id)}")
    if len(tasks) != manifest.get("task_count") or [item.task.task_id for item in tasks] != manifest.get(
        "task_ids"
    ):
        raise ValueError("public bundle manifest task identity mismatch")
    return tasks


def admit_public_trajectory(
    *,
    bundle_task: PublicBundleTask,
    actions: list[AgentAction],
    scratch_root: str | Path,
    worker_factory: Callable[[SandboxPaths], PythonWorker],
    provenance: TeacherProvenance,
) -> AdmittedTrajectory:
    def verifier(task: TaskSpec, submission: Submission):
        if task.task_id != bundle_task.task.task_id:
            raise ValueError("public replay task identity changed")
        return verify_public_submission(task, submission, bundle_task.expected)

    paths = SandboxPaths(
        bundle_task.input_root,
        Path(scratch_root) / bundle_task.task.task_id,
    )
    environment = EpisodeEnvironment(
        bundle_task.task,
        paths,
        worker=worker_factory(paths),
        submission_verifier=verifier,
    )
    return admit_trajectory(
        task=bundle_task.task,
        kind=TrajectoryKind.DIRECT_SUCCESS,
        actions=actions,
        environment=environment,
        provenance=provenance,
    )
