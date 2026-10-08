"""Explicit zero-update student source admission; never fabricates a GRPO launch."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from veripy_agent.contracts import TaskSpec


def _sha(path: Path) -> str:
    resolved = path.resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size <= 0
        or resolved.is_relative_to(Path("/root/veripy-secrets"))
        or path.suffix == ".env"
    ):
        raise ValueError("source file missing, empty, secret or symlink")
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("source changed during hashing")
    return h.hexdigest()


def bind_sampling_source(root: Path) -> dict:
    """Revalidate immutable sampling/model/train bytes and complete four-way roster.

    The review is an attributed agent artifact, not human approval or evidence of
    learning. Completion's exit0 is explicitly self-recorded, not an OS receipt.
    No oracle, probe input or review prose enters the returned teacher binding.
    """
    root = root.resolve()
    plan_path = root / "plan.json"
    plan_hash = _sha(plan_path)
    plan = json.loads(plan_path.read_text())
    if (
        plan.get("schema") != "veripy.repaired_parent_sampling.plan.v1"
        or type(plan.get("optimizer_updates")) is not int
        or plan["optimizer_updates"] != 0
        or type(plan.get("tasks_count")) is not int
        or plan["tasks_count"] != 32
        or type(plan.get("samples_per_task")) is not int
        or plan["samples_per_task"] != 4
        or plan.get("max_turns") != 6
    ):
        raise ValueError("invalid zero-update sampling plan")
    for name in ["model_bindings", "data_bindings"]:
        if not isinstance(plan.get(name), dict) or not plan[name]:
            raise ValueError("sampling source requires byte bindings")
        for filename, expected in plan[name].items():
            if _sha(Path(filename)) != expected:
                raise ValueError("sampling source artifact changed")
    base, adapter = Path(plan["base"]), Path(plan["adapter"])
    for p in [
        base / "model.safetensors",
        adapter / "adapter_model.safetensors",
        adapter / "adapter_config.json",
    ]:
        if str(p) not in plan["model_bindings"]:
            raise ValueError("sampling model weight/config identity unbound")
    config = json.loads((adapter / "adapter_config.json").read_text())
    if Path(config["base_model_name_or_path"]).resolve() != base.resolve():
        raise ValueError("sampling adapter base mismatch")
    tasks_path, train_path = Path(plan["tasks"]), Path(plan["train_parquet"])
    if any(str(p) not in plan["data_bindings"] for p in [tasks_path, train_path]):
        raise ValueError("sampling train identity unbound")
    tasks = [TaskSpec.model_validate(r["task"]) for r in json.loads(tasks_path.read_text())]
    specs = {t.task_id: t for t in tasks}
    if len(tasks) != 32 or len(specs) != 32 or any(t.split != "rl_train" for t in tasks):
        raise ValueError("invalid sampling task roster")
    completion_path = root / "completion.json"
    completion_hash = _sha(completion_path)
    completion = json.loads(completion_path.read_text())
    if (
        completion.get("status") != "completed_review_pending"
        or completion.get("source") != "no_update_student_sampling"
        or type(completion.get("exit_code")) is not int
        or completion["exit_code"] != 0
        or type(completion.get("optimizer_updates")) is not int
        or completion["optimizer_updates"] != 0
        or completion.get("learning_verified") is not False
        or completion.get("plan_sha256") != plan_hash
        or completion.get("episodes") != 128
    ):
        raise ValueError("sampling completion identity invalid")
    counts: Counter = Counter()
    inventory = {}
    aggregate = hashlib.sha256()
    for path in sorted((root / "receipts").glob("*.json")):
        checksum = _sha(path)
        raw = path.read_bytes()
        aggregate.update(path.name.encode() + b"\0" + raw)
        r = json.loads(raw)
        tid = r.get("task_id")
        if (
            r.get("schema") != "veripy.student_sampling_receipt.v1"
            or r.get("source") != "no_update_student_sampling"
            or r.get("split") != "rl_train"
            or type(r.get("optimizer_updates")) is not int
            or r["optimizer_updates"] != 0
            or tid not in specs
        ):
            raise ValueError("sampling receipt source invalid")
        spec = specs[tid]
        if (
            r.get("family") != spec.family
            or r.get("structure") != spec.verifier_spec.get("structure", {})
            or type(r.get("won")) is not bool
            or type(r.get("submitted")) is not bool
            or r["won"] != (r.get("failure_code") == "success")
            or r["won"]
            and not r["submitted"]
            or not isinstance(r.get("episode_reward"), (float, int))
            or not math.isfinite(r["episode_reward"])
            or "security_violation" in r.get("step_failure_codes", [])
        ):
            raise ValueError("sampling receipt outcome invalid")
        steps = r.get("steps")
        if (
            not isinstance(steps, list)
            or not 1 <= len(steps) <= 6
            or any(
                s.get("turn") != i + 1 or not isinstance(s.get("action"), str)
                for i, s in enumerate(steps)
            )
        ):
            raise ValueError("sampling trace invalid")
        counts[tid] += 1
        inventory[path.name] = checksum
    if (
        set(counts) != set(specs)
        or any(n != 4 for n in counts.values())
        or dict(counts) != completion.get("group_counts")
        or inventory != completion.get("receipt_hashes")
        or aggregate.hexdigest() != completion.get("receipts_sha256")
    ):
        raise ValueError("sampling receipt roster/hash mismatch")
    verification_path = root / "sampling_receipt_verification.json"
    review_path = root / "main_trajectory_review.json"
    verification_hash, review_hash = _sha(verification_path), _sha(review_path)
    verification = json.loads(verification_path.read_text())
    review = json.loads(review_path.read_text())
    for document in [verification, review]:
        if (
            document.get("plan_sha256") != plan_hash
            or document.get("completion_sha256") != completion_hash
        ):
            raise ValueError("sampling review/verification identity changed")
    reviewed = {Path(e["receipt"]).name: e["sha256"] for e in review.get("entries", [])}
    if (
        verification.get("receipts_sha256") != aggregate.hexdigest()
        or review.get("schema") != "veripy.agent_sampling_review.v1"
        or review.get("coverage", {}).get("full_action_feedback_trajectories") != 128
        or reviewed != inventory
        or len(review.get("entries", [])) != 128
    ):
        raise ValueError("sampling review coverage/receipt binding invalid")
    return {
        "source_kind": "no_update_student_sampling",
        "source_receipt_schema": "veripy.student_sampling_receipt.v1",
        "source_sampling_root": str(root),
        "source_plan_sha256": plan_hash,
        "source_completion_sha256": completion_hash,
        "source_sampling_verification_sha256": verification_hash,
        "source_trajectory_review_sha256": review_hash,
        "source_tasks_sha256": _sha(tasks_path),
        "source_train_sha256": _sha(train_path),
        "source_receipts_sha256": aggregate.hexdigest(),
        "source_model_base": str(base.resolve()),
        "source_model_adapter": str(adapter.resolve()),
        "source_model_bindings": dict(plan["model_bindings"]),
        "optimizer_updates": 0,
        "purpose": "train_course_design",
        "exit_status_provenance": "self_recorded_completion_not_independent_os_receipt",
    }
