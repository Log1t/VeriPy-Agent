"""Bounded offline teacher evidence. No model execution, oracle, or reward changes."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from veripy_agent.contracts import TaskSpec
from veripy_agent.data.behavior_curriculum import episode_features
from veripy_agent.data.generator import generate_tasks


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def bind_source_checkpoint(launch_path: Path) -> dict:
    """Snapshot complete source checkpoint bytes without loading tensors or using GPU."""
    launch = json.loads(launch_path.read_text())
    if (launch.get("status") != "completed" or launch.get("exit_code") != 0
            or type(launch.get("steps")) is not int or launch["steps"] < 1):
        raise ValueError("source checkpoint requires a completed run")
    root = launch_path.parent / f"global_step_{launch['steps']}"
    required = ["actor/model_world_size_1_rank_0.pt", "actor/optim_world_size_1_rank_0.pt",
                "actor/extra_state_world_size_1_rank_0.pt", "data.pt"]
    artifacts = {}
    for name in required:
        path = root / name
        if (path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                or not path.is_file() or path.stat().st_size <= 0):
            raise ValueError("source checkpoint file missing, empty or escaping")
        before = path.stat()
        checksum = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                checksum.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("source checkpoint changed during hashing")
        artifacts[name] = {"bytes": after.st_size, "sha256": checksum.hexdigest()}
    return {"schema": "veripy.teacher_harness.checkpoint_snapshot.v1", "path": str(root.resolve()),
            "artifacts": artifacts, "snapshot_sha256": digest(artifacts),
            "scope": "nonempty_completed_source_checkpoint_bytes_not_tensor_semantics_or_learning"}


def _hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def attach_harness(
    context: dict, train_rows: list[dict], receipt_root: Path, *, binding: dict,
    rollouts_per_task: int = 4, max_context_bytes: int = 256_000,
) -> dict:
    """Return an allowlisted, fully bound packet; refuse truncation or incomplete sources."""
    if (context.get("source_split") != "rl_train" or not context.get("behavior_targeted")
            or context.get("public_contract_version") != "explicit-v1"):
        raise ValueError("harness requires explicit behavior train evidence")
    sampling = binding.get("source_kind") == "no_update_student_sampling"
    if sampling:
        from veripy_agent.data.sampling_source import bind_sampling_source

        source_root = Path(binding["source_sampling_root"])
        if (bind_sampling_source(source_root) != binding
                or context.get("source_plan_sha256") != binding["source_plan_sha256"]
                or receipt_root.resolve() != (source_root / "receipts").resolve()):
            raise ValueError("invalid sampling source binding")
        source_plan = json.loads((source_root / "plan.json").read_text())
        source_tasks = {r["task"]["task_id"]: TaskSpec.model_validate(r["task"])
                        for r in json.loads(Path(source_plan["tasks"]).read_text())}
        supplied_tasks = {r["extra_info"]["task_id"]: TaskSpec.model_validate(r["env_kwargs"]["task"])
                          for r in train_rows}
        if supplied_tasks != source_tasks or len(supplied_tasks) != len(train_rows):
            raise ValueError("sampling train roster changed")
    elif binding.get("source_kind") not in {None, "train_rollout"}:
        raise ValueError("unknown teacher source kind")
    elif (not _hash(binding.get("source_launch_sha256"))
            or not _hash(binding.get("source_train_sha256"))
            or not _hash(binding.get("source_receipts_sha256"))
            or context.get("source_launch_sha256") != binding["source_launch_sha256"]
            or not binding.get("source_checkpoint")
            or binding.get("purpose") not in {"historical_cpu_integration", "train_course_design"}):
        raise ValueError("invalid source binding")
    if type(rollouts_per_task) is not int or rollouts_per_task != 4:
        raise ValueError("harness requires complete four-way source groups")
    if type(max_context_bytes) is not int or not 1 <= max_context_bytes <= 512_000:
        raise ValueError("context budget outside bounded range")
    specs = {r["extra_info"]["task_id"]: TaskSpec.model_validate(r["env_kwargs"]["task"]) for r in train_rows}
    if len(specs) != len(train_rows) or any(k != s.task_id or s.split != "rl_train" for k, s in specs.items()):
        raise ValueError("train roster identity mismatch")
    receipts = {}
    counts: Counter = Counter()
    inventory = {}
    aggregate = hashlib.sha256()
    for path in sorted(receipt_root.glob("*.json")):
        raw = path.read_bytes()
        aggregate.update(path.name.encode() + b"\0" + raw)
        key = hashlib.sha256(raw).hexdigest()
        r = json.loads(raw)
        tid = r.get("task_id")
        if (path.is_symlink() or not path.resolve().is_relative_to(receipt_root.resolve())
                or r.get("schema") != ("veripy.student_sampling_receipt.v1" if sampling else "veripy.train_rollout_receipt")
                or r.get("source") != ("no_update_student_sampling" if sampling else "train_rollout") or r.get("split") != "rl_train"
                or tid not in specs):
            raise ValueError("non-training or escaping receipt")
        spec = specs[tid]
        if (r.get("family") != spec.family or r.get("structure") != spec.verifier_spec.get("structure", {})
                or type(r.get("won")) is not bool or type(r.get("submitted")) is not bool
                or r["won"] != (r.get("failure_code") == "success")
                or r["won"] and not r["submitted"]
                or not isinstance(r.get("episode_reward"), (int, float))
                or not math.isfinite(r["episode_reward"])
                or "security_violation" in r.get("step_failure_codes", [])):
            raise ValueError("receipt identity or outcome invalid")
        steps = r.get("steps")
        if (not isinstance(steps, list) or not 1 <= len(steps) <= 6
                or any(type(s.get("turn")) is not int or s["turn"] != i + 1
                       or not isinstance(s.get("action"), str) for i, s in enumerate(steps))):
            raise ValueError("invalid bounded trace")
        receipts[key] = r
        counts[tid] += 1
        inventory[path.name] = key
    if set(counts) != set(specs) or any(n != rollouts_per_task for n in counts.values()):
        raise ValueError("incomplete source roster")
    if aggregate.hexdigest() != binding["source_receipts_sha256"]:
        raise ValueError("source receipt aggregate changed")
    result = copy.deepcopy(context)
    episodes = {}
    facts = {}
    for cluster in result["clusters"]:
        for example in cluster["real_failure_examples"]:
            pair = [example, *example["successful_siblings"]]
            for pos, ref in enumerate(pair):
                key = ref["receipt_sha256"]
                if key not in receipts:
                    raise ValueError("cited receipt hash changed or unavailable")
                r = receipts[key]
                if r["task_id"] != example["task_id"] or bool(r["won"]) != (pos > 0):
                    raise ValueError("failure/sibling task or outcome mismatch")
                public_steps = []
                for step in r["steps"]:
                    observed = step.get("result") or {}
                    public_steps.append({
                        "turn": step["turn"], "action": step["action"],
                        "tool": step.get("tool"), "status": step.get("status"),
                        "stdout": str(observed.get("stdout", "")),
                        "stderr": str(observed.get("stderr", "")),
                        "no_progress": bool(step.get("no_progress", False)),
                        "upstream_output_truncated": bool(observed.get("resource_usage", {}).get("stdout_truncated", False)
                                                          or observed.get("resource_usage", {}).get("stderr_truncated", False)),
                    })
                features = episode_features(r)
                episodes[key] = {
                    "task_id": r["task_id"], "receipt_sha256": key,
                    "steps": public_steps, "features": features, "won": r["won"],
                    "submitted": r["submitted"], "episode_reward": r["episode_reward"],
                    "trace_truncated_by_harness": False,
                }
                spec = specs[r["task_id"]]
                episodes[key]["public_input_contract"] = {
                    "instruction": spec.instruction, "files": spec.files,
                    "required_columns": spec.verifier_spec.get("structure", {}).get("required_columns", []),
                    "answer_schema": spec.answer_schema,
                }
                # One complete canonical copy, not two long duplicates in the API context.
                ref["steps"] = []
                ref["full_trace_ref"] = key
                ref["features"] = features
                ref["truncated_fields"] = []
                for name, value in {"task_correct": r["won"], "submitted": r["submitted"], **features}.items():
                    facts[f"{key}:{name}"] = {"receipt_sha256": key, "task_id": r["task_id"],
                                               "field": name, "value": value}
    packet = {
        "schema": "veripy.teacher_harness.v1", "source_binding": dict(binding),
        "source_receipt_hashes": inventory, "source_roster_sha256": digest(inventory),
        "source_task_ids": sorted(specs),
        "episodes": episodes, "observed_facts": facts,
        "retrieval_catalog": [], "retrieval_pool_sha256": None,
        "student_learning_verified": False,
        "source_checkpoint_identity": ("sampling_base_adapter_byte_snapshot" if sampling else
                                       "byte_snapshot" if binding.get("source_checkpoint_snapshot") else "declared_path_only"),
        "reward_alignment": "Correctness/evidence rewards do not bind execution/stdout. Input conditions cannot force computation.",
        "limits": ["Observed facts are deterministic trace features, not causal explanations.",
                   "Same-task success siblings are alternatives, not proof of a repair.",
                   "No heldout evidence; no online teacher help; episode variance is not GRPO advantages.",
                   "Course legality does not prove student learnability or improvement."],
    }
    result["teacher_harness"] = packet
    if len(json.dumps(result, ensure_ascii=False).encode()) > max_context_bytes:
        raise ValueError("full evidence exceeds context budget; never silently truncate")
    return result


def verify_selected_sources(context: dict, decision: Any, manifest: dict) -> dict:
    """Verify every materialized task ID/source, including exact teacher retrieval IDs."""
    from veripy_agent.data.teacher_curriculum import generate_teacher_candidates

    expected = generate_teacher_candidates(context, decision, base_seed=manifest["generation_seed"])
    actual = {r["task_id"]: r for r in manifest["candidates"]}
    if len(actual) != len(manifest["candidates"]) or set(actual) != {r["task"]["task_id"] for r in expected}:
        raise ValueError("materialized teacher source task roster changed")
    sources = {}
    for row in expected:
        tid = row["task"]["task_id"]
        got = actual[tid]
        source = row["candidate_source"]
        expected_id = row.get("retrieval_task_id")
        # Early v1 manifests predate the explicit retrieval_task_id field. Exact
        # task identity is still verified against the unmodified teacher decision.
        if (got.get("candidate_source") != source
                or got.get("retrieval_task_id", tid if source == "retrieved" else None) != expected_id
                or got["mutation_seed"] != row["mutation_seed"]
                or got["mutation_index"] != row["mutation_index"]):
            raise ValueError("materialized retrieval source or identity changed")
        sources[tid] = {"candidate_source": source, "retrieval_task_id": expected_id}
    return sources


def input_digest(value: Any) -> str:
    return digest(value)


def attach_retrieval(context: dict, pool_rows: list[dict], *, exclusions: dict, limit: int = 32) -> dict:
    """Retrieve by operation/schema/difficulty and actual condition, never text similarity alone.

    Pool and exclusions are independently frozen inputs. This does not relax any
    caller's corpus/seed/content admission rules; only previously unused admissible
    training tasks can appear. The teacher sees no hidden answers or TaskSpec internals.
    """
    from veripy_agent.data.behavior_curriculum import input_properties
    from veripy_agent.data.teacher_curriculum import bucket

    if type(limit) is not int or not 1 <= limit <= 32 or not pool_rows:
        raise ValueError("invalid retrieval budget or empty pool")
    if not all(isinstance(exclusions.get(k), list) for k in ("input_fingerprints", "seeds", "task_ids")):
        raise ValueError("explicit retrieval exclusions required")
    result = copy.deepcopy(context)
    packet = result["teacher_harness"]
    source_ids = set(packet["source_task_ids"])
    forbidden_ids = source_ids | set(exclusions["task_ids"])
    forbidden_inputs = set(exclusions["input_fingerprints"])
    forbidden_seeds = set(exclusions["seeds"])
    catalog = []
    seen = set()
    for row in sorted(pool_rows, key=lambda r: r["extra_info"]["task_id"]):
        spec = TaskSpec.model_validate(row["env_kwargs"]["task"])
        if (spec.split != "rl_train" or spec.task_id != row["extra_info"]["task_id"]
                or spec.seed is None or spec.verifier_spec.get("public_contract_version") != "explicit-v1"):
            raise ValueError("retrieval pool must be explicit-contract train tasks")
        index = int(spec.verifier_spec["generator_index"])
        task = generate_tasks(spec.family, seed=spec.seed, count=index + 1, split="rl_train",
                              public_contract_version="explicit-v1")[index]
        # Paths change on materialization; every semantic/public field must remain identical.
        expected = task.task_spec.model_dump(exclude={"files"})
        if expected != spec.model_dump(exclude={"files"}) or spec.task_id in seen:
            raise ValueError("retrieval task identity changed or duplicate ID")
        seen.add(spec.task_id)
        if (spec.task_id in forbidden_ids or spec.seed in forbidden_seeds
                or input_digest(task.input) in forbidden_inputs):
            continue
        structure = spec.verifier_spec["structure"]
        for c in result["clusters"]:
            if (c["family"] != spec.family or c["operation_sequence"] != structure["operation_sequence"]
                    or c["semantic_signature"] != task.semantic_signature
                    or c["difficulty_bucket"] != bucket(structure["difficulty_score"])):
                continue
            design = next((d for d in c["allowed_designs"] if d["generator_index"] == index), None)
            if design is None:
                continue
            props = input_properties(task.input)
            conditions = [k for k in design["input_conditions"] if props.get(k, False)]
            if not conditions:
                continue
            catalog.append({
                "task_id": task.task_id, "cluster_id": c["cluster_id"], "seed": spec.seed,
                "generator_index": index, "content_signature": task.content_signature,
                "input_fingerprint": input_digest(task.input), "input_conditions": conditions,
                "public_instruction": spec.instruction, "answer_schema": spec.answer_schema,
                "match": "exact_operation_semantic_schema_difficulty_and_actual_condition",
            })
            break
    packet["retrieval_catalog"] = catalog[:limit]
    packet["retrieval_pool_sha256"] = digest(pool_rows)
    packet["retrieval_exclusions_sha256"] = digest(exclusions)
    return result


def summarize_sampling(context: dict, train_rows: list[dict], receipt_root: Path) -> dict:
    """Inspect bound, completed train-source groups, never authorize an optimizer run.

    For a future candidate calibration, the caller must create a new binding to
    that candidate sampling run, not reuse the historical course-design binding.
    Input-sensitive replay and turn-weighted advantage measurements remain unknown
    unless produced by their existing independent execution/audit paths.
    """
    binding = context["teacher_harness"]["source_binding"]
    checked = attach_harness(context, train_rows, receipt_root, binding=binding)
    if checked["teacher_harness"]["source_roster_sha256"] != context["teacher_harness"]["source_roster_sha256"]:
        raise ValueError("sampling source changed")
    groups: dict[str, list[dict]] = {}
    python_count = submitted = correct = 0
    for path in sorted(receipt_root.glob("*.json")):
        r = json.loads(path.read_text())
        groups.setdefault(r["task_id"], []).append(r)
        python_count += episode_features(r)["successful_python_calls"] > 0
        submitted += r["submitted"]
        correct += r["won"]
    per_task = []
    for tid, rows in sorted(groups.items()):
        wins = sum(r["won"] for r in rows)
        variance = statistics.pvariance(r["episode_reward"] for r in rows)
        per_task.append({"task_id": tid, "rollouts": len(rows), "task_correct": wins,
                         "episode_reward_variance": variance, "mixed_success": 0 < wins < 4})
    mixed = sum(r["mixed_success"] for r in per_task)
    varying = sum(r["episode_reward_variance"] > 0 for r in per_task)
    return {
        "schema": "veripy.teacher_harness.sampling_report.v1",
        "source_binding": binding, "source_roster_sha256": context["teacher_harness"]["source_roster_sha256"],
        "task_groups": len(groups), "episodes": len(groups) * 4,
        "task_correct_episodes": correct, "submitted_episodes": submitted,
        "successful_python_episodes": python_count, "mixed_task_success_groups": mixed,
        "all_success_groups": sum(r["task_correct"] == 4 for r in per_task),
        "all_failure_groups": sum(r["task_correct"] == 0 for r in per_task),
        "nonzero_episode_reward_variance_groups": varying, "per_task": per_task,
        "sampling_signal_observed": mixed > 0 and varying > 0,
        "input_sensitive_primary": None, "turn_weighted_grpo_advantages": None,
        "training_authorized": False, "student_learning_verified": False,
        "reason": "Sampling diversity is diagnostic only. Independent replay, current-parent identity and optimizer admission remain required.",
    }
