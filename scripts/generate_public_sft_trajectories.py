#!/usr/bin/env python3
"""Admit teacher-produced public trajectories after local sandbox replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from veripy_agent.environment import HardenedProcessWorker
from veripy_agent.rollout import AgentAction, protocol_manifest_from_resources
from veripy_agent.training.public_replay import admit_public_trajectory, load_public_bundle
from veripy_agent.training.sft import (
    TOOLS,
    AdmittedTrajectory,
    TeacherProvenance,
    write_admitted_trajectories,
    write_dataset_info,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_actions(path: Path) -> dict[str, tuple[list[AgentAction], TeacherProvenance]]:
    records: dict[str, tuple[list[AgentAction], TeacherProvenance]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or "task_id" not in value:
            raise ValueError(f"invalid action record at {path}:{line_number}")
        task_id = str(value["task_id"])
        if task_id in records:
            raise ValueError(f"duplicate public action task: {task_id}")
        actions = [AgentAction.model_validate(item) for item in value["actions"]]
        provenance = TeacherProvenance.model_validate(value["provenance"])
        records[task_id] = actions, provenance
    return records


def _bundle_roots(freeze_root: Path, manifest: dict[str, Any]) -> list[Path]:
    roots: list[Path] = []
    for source in manifest.get("sources", []):
        for split in source.get("splits", {}).values():
            relative = Path(str(split["bundle"]))
            bundle_root = (freeze_root / relative).parent
            if not (bundle_root / "manifest.json").is_file():
                raise ValueError(f"public bundle is missing: {bundle_root}")
            roots.append(bundle_root)
    return roots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--actions-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resources-lock", type=Path, default=Path("resources.lock.json"))
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--sandbox-user", default="veripy-sandbox")
    parser.add_argument("--dataset-name", default="veripy_public_sft")
    parser.add_argument("--limit", type=int, default=0, help="limit bundle tasks visited for a bounded replay smoke")
    args = parser.parse_args()

    freeze_manifest = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    if freeze_manifest.get("schema") != "veripy.public_corpus.freeze":
        raise ValueError("unsupported public freeze manifest schema")
    if freeze_manifest.get("status") != "passed":
        raise ValueError("public freeze manifest is not admitted")
    actions = _load_actions(args.actions_jsonl)
    admitted: list[AdmittedTrajectory] = []
    rejected: list[dict[str, Any]] = []
    scratch_root = args.output_dir / "scratch"
    visited = 0
    for bundle_root in _bundle_roots(args.freeze_manifest.parent, freeze_manifest):
        for bundle_task in load_public_bundle(bundle_root):
            if args.limit and visited >= args.limit:
                break
            visited += 1
            record = actions.get(bundle_task.task.task_id)
            if record is None:
                rejected.append({"task_id": bundle_task.task.task_id, "error": "missing_actions"})
                continue
            task_actions, provenance = record
            try:
                trajectory = admit_public_trajectory(
                    bundle_task=bundle_task,
                    actions=task_actions,
                    scratch_root=scratch_root,
                    worker_factory=lambda paths: HardenedProcessWorker(
                        paths, user=args.sandbox_user
                    ),
                    provenance=provenance,
                )
            except Exception as exc:
                rejected.append(
                    {
                        "task_id": bundle_task.task.task_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                continue
            admitted.append(trajectory)
        if args.limit and visited >= args.limit:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "trajectories.audit.jsonl"
    dataset_path = args.output_dir / "trajectories.jsonl"
    write_admitted_trajectories(admitted, audit_path=audit_path, dataset_path=dataset_path)
    write_dataset_info(
        dataset_dir=args.output_dir,
        dataset_name=args.dataset_name,
        dataset_file=dataset_path.name,
    )
    rejected_path = args.output_dir / "trajectories.rejected.jsonl"
    rejected_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in rejected), encoding="utf-8"
    )
    protocol = protocol_manifest_from_resources(
        TOOLS,
        resources_lock=args.resources_lock,
        tokenizer_path=args.tokenizer_path,
    )
    manifest = {
        "schema": "veripy.public_sft.trajectory_manifest",
        "schema_version": 1,
        "count": len(admitted),
        "trajectory_kinds": {"direct_success": len(admitted)},
        "freeze_manifest": str(args.freeze_manifest),
        "freeze_manifest_sha256": _sha256(args.freeze_manifest),
        "actions": str(args.actions_jsonl),
        "actions_sha256": _sha256(args.actions_jsonl),
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "audit": str(audit_path),
        "audit_sha256": _sha256(audit_path),
        "rejected": str(rejected_path),
        "rejected_count": len(rejected),
        "visited_count": visited,
        "protocol": protocol,
    }
    (args.output_dir / "trajectories.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"accepted": len(admitted), "rejected": len(rejected)}, sort_keys=True))


if __name__ == "__main__":
    main()
