#!/usr/bin/env python3
"""Build paired GRPO datasets from real training-episode failure receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from veripy_agent.contracts import TaskSpec
from veripy_agent.data import (
    admit_curriculum_candidates,
    build_curriculum_candidates,
    curriculum_manifest,
    generate_tasks,
    validate_task_bundle,
    write_tasks,
)
from veripy_agent.evaluation.failures import cluster_failures
from veripy_agent.verl_adapter import build_rollout_row, validate_rollout_row


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_input_exclusion(candidates, snapshot):
    """Optional study admission: exclude raw input even if prompt/split/seed differ."""
    inputs = set(snapshot["input_fingerprints"])
    seeds = set(snapshot["seeds"])
    ids = set(snapshot["task_ids"])
    for c in candidates:
        key = hashlib.sha256(
            json.dumps(c["task"]["input"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if key in inputs or c["mutation_seed"] in seeds or c["task"]["task_id"] in ids:
            raise ValueError("candidate concrete input, seed or ID overlaps frozen exclusions")
        inputs.add(key)
        seeds.add(c["mutation_seed"])
        ids.add(c["task"]["task_id"])


def _corpus_signatures(
    corpus_path: Path, lock_path: Path
) -> tuple[set[str], set[int], dict[str, Any]]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if Path(lock["manifest"]).resolve() != corpus_path.resolve() or lock["sha256"] != _sha256(
        corpus_path
    ):
        raise ValueError("locked corpus identity mismatch")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    signatures: set[str] = set()
    seeds: set[int] = set()
    for bundle in corpus["bundles"]:
        path = Path(bundle["manifest"])
        if _sha256(path) != bundle["manifest_sha256"]:
            raise ValueError(f"corpus bundle hash mismatch: {path}")
        checked = validate_task_bundle(path)
        signatures.update(checked["content_signatures"])
        seeds.update(checked["seeds"])
    return signatures, seeds, corpus


def _train_failures(
    receipt_root: Path,
    train_rows: list[dict[str, Any]],
    *,
    learnable_only: bool = False,
    sampling_source_root: Path | None = None,
) -> tuple[list[dict[str, Any]], str]:
    def without_arrow_padding(value):
        if isinstance(value, dict):
            return {k: without_arrow_padding(v) for k, v in value.items() if v is not None}
        if isinstance(value, list):
            return [without_arrow_padding(v) for v in value]
        return value

    sampling_binding = None
    if sampling_source_root is not None:
        from veripy_agent.data.sampling_source import bind_sampling_source

        sampling_binding = bind_sampling_source(sampling_source_root)
        if receipt_root.resolve() != (sampling_source_root / "receipts").resolve():
            raise ValueError("sampling receipts must belong to the validated source")
        plan = json.loads((sampling_source_root / "plan.json").read_text())
        expected = {
            x["task"]["task_id"]: TaskSpec.model_validate(x["task"]).model_dump(mode="json")
            for x in json.loads(Path(plan["tasks"]).read_text())
        }
        actual = {
            x["extra_info"]["task_id"]: TaskSpec.model_validate(x["env_kwargs"]["task"]).model_dump(
                mode="json"
            )
            for x in train_rows
        }
        if without_arrow_padding(actual) != without_arrow_padding(expected) or len(actual) != len(
            train_rows
        ):
            raise ValueError("sampling train roster differs from the validated source")
    by_id = {row["extra_info"]["task_id"]: row for row in train_rows}
    if len(by_id) != len(train_rows):
        raise ValueError("base RL train parquet has duplicate task IDs")
    paths = sorted(receipt_root.glob("*.json"))
    if not paths:
        raise ValueError("no real training receipts found")
    digest = hashlib.sha256()
    failures = []
    group_wins: dict[str, list[bool]] = {}
    for path in paths:
        raw = path.read_bytes()
        digest.update(path.name.encode() + b"\0" + raw)
        receipt = json.loads(raw)
        if (
            receipt.get("schema")
            != (
                "veripy.student_sampling_receipt.v1"
                if sampling_binding
                else "veripy.train_rollout_receipt"
            )
            or receipt.get("source")
            != ("no_update_student_sampling" if sampling_binding else "train_rollout")
            or receipt.get("split") != "rl_train"
        ):
            raise ValueError(f"non-training receipt: {path}")
        task_id = receipt["task_id"]
        row = by_id.get(task_id)
        if row is None or row["ability"] != receipt["family"]:
            raise ValueError(f"receipt task identity not in locked train pool: {task_id}")
        task = TaskSpec.model_validate(row["env_kwargs"]["task"])
        expected_structure = task.verifier_spec.get("structure", {})
        if sampling_binding:
            expected_structure = without_arrow_padding(expected_structure)
        if receipt["structure"] != expected_structure:
            raise ValueError(f"receipt task structure mismatch: {task_id}")
        if bool(receipt["won"]) != (receipt["failure_code"] == "success"):
            raise ValueError(f"receipt success/failure mismatch: {task_id}")
        if "security_violation" in receipt.get("step_failure_codes", ()):
            raise ValueError(f"security-violating source episode requires review: {task_id}")
        group_wins.setdefault(task_id, []).append(bool(receipt["won"]))
        if not receipt["won"]:
            failures.append(
                {
                    "source": "synthetic",
                    "task_id": task_id,
                    "family": task.family,
                    "failure_code": receipt["failure_code"],
                    "structure": receipt["structure"],
                    "semantic_signature": task.verifier_spec["semantic_signature"],
                    "receipt_path": str(path),
                    "receipt_sha256": _sha256(path),
                    **(
                        {"student_evidence_source": "no_update_student_sampling"}
                        if sampling_binding
                        else {}
                    ),
                }
            )
    if learnable_only:
        if any(len(wins) != 4 for wins in group_wins.values()):
            raise ValueError("learnability admission requires complete four-way groups")
        mixed = {task_id for task_id, wins in group_wins.items() if 0 < sum(wins) < 4}
        failures = [record for record in failures if record["task_id"] in mixed]
    if not failures:
        raise ValueError("training receipts contain no failure episodes")
    if sampling_binding and digest.hexdigest() != sampling_binding["source_receipts_sha256"]:
        raise ValueError("sampling receipts changed during failure extraction")
    return failures, digest.hexdigest()


def _materialize(
    candidates: list[dict[str, Any]],
    *,
    output_root: Path,
    dataset_root: Path,
    split: str = "rl_train",
) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        spec = TaskSpec.model_validate(candidate["task"]["task_spec"])
        index = int(spec.verifier_spec["generator_index"])
        task = generate_tasks(
            spec.family,
            seed=spec.seed,
            count=index + 1,
            split=split,
            public_contract_version=spec.verifier_spec.get("public_contract_version"),
        )[index]
        if task.task_id != spec.task_id:
            raise ValueError(f"candidate task identity changed: {spec.task_id}")
        output = output_root / "tasks" / f"{task.task_id}.jsonl"
        write_tasks([task], output, split=split, seed=task.seed)
        checked = validate_task_bundle(output.with_suffix(".jsonl.manifest.json"))
        recorded = TaskSpec.model_validate(checked["_records"][0]["task_spec"])
        input_path = output.parent / recorded.files[0]
        input_root = input_path.parent.resolve()
        if not input_root.is_relative_to(dataset_root):
            raise ValueError(f"candidate input escaped dataset root: {input_root}")
        episode_task = recorded.model_copy(update={"files": [input_path.name]})
        row = build_rollout_row(episode_task, input_root=str(input_root.relative_to(dataset_root)))
        validate_rollout_row(row)
        rows.append(row)
    return rows


def _anchors(rows: list[dict[str, Any]], *, count: int, seed: int) -> list[dict[str, Any]]:
    families = sorted({row["ability"] for row in rows})
    if count < len(families) or count % len(families):
        raise ValueError("anchor count must be a positive multiple of family count")
    selected = []
    for family in families:
        family_rows = [row for row in rows if row["ability"] == family]
        family_rows.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}|{row['extra_info']['task_id']}".encode()
            ).hexdigest()
        )
        selected.extend(family_rows[: count // len(families)])
    if len(selected) != count:
        raise ValueError("insufficient frozen training tasks for balanced anchors")
    return selected


def validate_teacher_resume_output(output_root: Path):
    if not output_root.is_dir() or {p.name for p in output_root.iterdir()} != {"teacher"}:
        raise ValueError(
            "teacher resume refuses partial or materialized output outside teacher receipts"
        )
    if not (output_root / "teacher/rejection_3.json").is_file():
        raise ValueError("teacher resume requires a preserved failed request")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-launch-manifest", type=Path, required=True)
    parser.add_argument("--receipts-dir", type=Path, required=True)
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("data/manifests/task_corpus.json"))
    parser.add_argument("--input-exclusion", type=Path, help="locked raw-input exclusion snapshot")
    parser.add_argument(
        "--corpus-lock", type=Path, default=Path("configs/data/task_corpus.lock.json")
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/generated"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--anchor-manifest",
        type=Path,
        help="reuse exactly a pre-frozen common-anchor roster before construction",
    )
    parser.add_argument(
        "--public-contract-version",
        choices=["explicit-v1"],
        help="future teacher candidates only; frozen anchors and validation remain unchanged",
    )
    parser.add_argument(
        "--behavior-targeted",
        action="store_true",
        help="require paired trace hypotheses and realized input-condition decisions",
    )
    parser.add_argument(
        "--behavior-scope",
        type=Path,
        help="pre-frozen single-target measurement/family/operation quotas",
    )
    parser.add_argument(
        "--teacher-harness",
        action="store_true",
        help="v1 full evidence, factual citations and bounded course-source decisions",
    )
    parser.add_argument(
        "--harness-purpose", choices=["historical_cpu_integration", "train_course_design"]
    )
    parser.add_argument(
        "--retrieval-train",
        type=Path,
        help="pre-frozen unused explicit-v1 rl_train candidate parquet",
    )
    parser.add_argument("--retrieval-train-sha256")
    parser.add_argument("--teacher-max-attempts", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument(
        "--resume-teacher-rejections",
        action="store_true",
        help="one explicit interface repair of an exhausted rejected CPU decision",
    )
    parser.add_argument(
        "--exclude-curriculum-manifest",
        action="append",
        type=Path,
        default=[],
        help="exclude fresh tasks from prior CPU courses, not their frozen anchors",
    )
    parser.add_argument(
        "--teacher-secret",
        type=Path,
        help="offline teacher credentials outside repository; enables actual AI task design",
    )
    parser.add_argument(
        "--prior-teacher-decision",
        type=Path,
        help="previous teacher decisions only; never evaluation feedback",
    )
    parser.add_argument(
        "--strategies", nargs="+", choices=("adaptive", "random"), default=["adaptive", "random"]
    )
    parser.add_argument(
        "--exclude-operation",
        action="append",
        default=[],
        help="slash-separated operation sequence pending public-contract review",
    )
    parser.add_argument(
        "--learnable-only",
        action="store_true",
        help="only mutate real training tasks with mixed group success",
    )
    args = parser.parse_args()
    if args.teacher_harness and (
        not args.behavior_targeted
        or not args.harness_purpose
        or args.prior_teacher_decision
        or args.resume_teacher_rejections
    ):
        raise ValueError(
            "harness requires fresh explicit behavior context and purpose; no unvalidated prior decision"
        )
    if (args.retrieval_train or args.retrieval_train_sha256) and (
        not args.teacher_harness
        or not args.retrieval_train
        or not args.retrieval_train_sha256
        or not args.input_exclusion
    ):
        raise ValueError("retrieval requires harness, pinned pool and exclusions")
    if args.public_contract_version and not args.teacher_secret:
        raise ValueError("public contract opt-in requires teacher-designed candidates")
    if args.behavior_targeted and (
        not args.teacher_secret
        or not args.public_contract_version
        or not args.learnable_only
        or args.strategies != ["adaptive"]
    ):
        raise ValueError(
            "behavior targeting requires teacher, explicit contract, adaptive mixed sources"
        )
    scope = json.loads(args.behavior_scope.read_text()) if args.behavior_scope else None
    if scope is not None and (
        not args.behavior_targeted
        or scope.get("required_measurement") != "computation_and_submit"
        or scope.get("family") != "missing_values"
        or sum(scope.get("operation_quotas", {}).values()) != args.count
    ):
        raise ValueError("single-target scope requires frozen missing-value computation quotas")
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    if not output_root.is_relative_to(dataset_root) or output_root == dataset_root:
        raise ValueError("curriculum output must be a child of dataset root")
    if args.resume_teacher_rejections:
        if not args.behavior_targeted:
            raise ValueError("teacher resume cannot overwrite a materialized course")
        validate_teacher_resume_output(output_root)
    elif output_root.exists() and any(output_root.iterdir()):
        raise ValueError("curriculum output must be fresh")
    launch = json.loads(args.source_launch_manifest.read_text(encoding="utf-8"))
    if (
        launch.get("mode") not in {"curriculum_collect", "staged_curriculum", "curriculum"}
        or launch.get("status") != "completed"
        or launch.get("exit_code") != 0
        or launch.get("steps", 0) < 1
    ):
        raise ValueError("source GRPO run is not completed")
    overrides = dict(item.split("=", 1) for item in launch["command"][2:] if "=" in item)
    expected_receipts = (
        int(launch["steps"])
        * int(overrides["data.train_batch_size"])
        * int(overrides["env.rollout.n"])
    )
    actual_receipts = len(list(args.receipts_dir.glob("*.json")))
    if actual_receipts != expected_receipts:
        raise ValueError(f"incomplete training receipts: {actual_receipts} != {expected_receipts}")
    train_artifact = launch["data_artifacts"]["train"]
    if Path(train_artifact["path"]).resolve() != args.base_train.resolve() or train_artifact[
        "sha256"
    ] != _sha256(args.base_train):
        raise ValueError("base train parquet differs from completed source run")
    if not args.receipts_dir.resolve().is_relative_to(args.source_launch_manifest.parent.resolve()):
        raise ValueError("receipts must belong to the source GRPO run")
    existing_signatures, existing_seeds, _ = _corpus_signatures(args.corpus, args.corpus_lock)
    input_exclusion = json.loads(args.input_exclusion.read_text()) if args.input_exclusion else None
    excluded_manifests = {}
    for prior_path in args.exclude_curriculum_manifest:
        prior = json.loads(prior_path.read_text())
        if _sha256(Path(prior["train_parquet"])) != prior["train_parquet_sha256"]:
            raise ValueError("excluded curriculum parquet hash changed")
        for candidate in prior["candidates"]:
            existing_signatures.add(candidate["content_signature"])
            existing_seeds.add(int(candidate["mutation_seed"]))
        excluded_manifests[str(prior_path)] = _sha256(prior_path)
    validation_artifact = launch["data_artifacts"]["val"]
    validation_path = Path(validation_artifact["path"])
    if validation_artifact["sha256"] != _sha256(validation_path):
        raise ValueError("source run frozen validation parquet changed")
    train_rows = pq.read_table(args.base_train).to_pylist()
    # Later rounds must also exclude the source update's newly generated tasks.
    for row in train_rows:
        spec = TaskSpec.model_validate(row["env_kwargs"]["task"])
        if spec.seed is None:
            raise ValueError("source train task seed is required")
        task = generate_tasks(
            spec.family,
            seed=spec.seed,
            count=int(spec.verifier_spec["generator_index"]) + 1,
            split="rl_train",
            public_contract_version=spec.verifier_spec.get("public_contract_version"),
        )[int(spec.verifier_spec["generator_index"])]
        if task.task_id != spec.task_id:
            raise ValueError("source train task regeneration identity changed")
        existing_signatures.add(task.content_signature)
        existing_seeds.add(spec.seed)
    failures, receipts_sha256 = _train_failures(
        args.receipts_dir, train_rows, learnable_only=args.learnable_only
    )
    if args.behavior_targeted:
        from veripy_agent.data.behavior_curriculum import validate_behavior_roster

        if expected_receipts != len(train_rows) * int(overrides["env.rollout.n"]):
            raise ValueError(
                "behavior source requires one complete pass over the frozen train pool"
            )
        validate_behavior_roster(
            [json.loads(p.read_text()) for p in sorted(args.receipts_dir.glob("*.json"))],
            [r["extra_info"]["task_id"] for r in train_rows],
            int(overrides["env.rollout.n"]),
        )
    # Preserve the concrete operation: a failed covariance task must not become
    # a Pearson task merely because their family and difficulty bucket match.
    by_operations = defaultdict(list)
    for failure in failures:
        by_operations[
            (tuple(failure["structure"]["operation_sequence"]), failure["semantic_signature"])
        ].append(failure)
    clusters = []
    for (operations, semantic), records in sorted(by_operations.items()):
        for cluster in cluster_failures(records):
            cluster["operation_sequence"] = list(operations)
            cluster["semantic_signature"] = semantic
            clusters.append(cluster)
    clusters.sort(
        key=lambda row: (
            -row["count"],
            row["family"],
            row["operation_sequence"],
            row["failure_code"],
        )
    )
    excluded_clusters = [
        row for row in clusters if "/".join(row["operation_sequence"]) in args.exclude_operation
    ]
    clusters = [row for row in clusters if row not in excluded_clusters]
    if scope is not None:
        clusters = [row for row in clusters if row["family"] == scope["family"]]
    anchor_rows = _anchors(train_rows, count=args.count, seed=args.seed)
    if args.anchor_manifest:
        anchor_manifest = json.loads(args.anchor_manifest.read_text())
        if (
            _sha256(Path(anchor_manifest["train_parquet"]))
            != anchor_manifest["train_parquet_sha256"]
        ):
            raise ValueError("common anchor source parquet changed")
        ids = anchor_manifest["anchor_task_ids"]
        by_id = {r["extra_info"]["task_id"]: r for r in train_rows}
        if len(ids) != args.count or len(set(ids)) != len(ids) or not set(ids) <= by_id.keys():
            raise ValueError("common anchor roster is not a unique subset of source train")
        anchor_rows = [by_id[tid] for tid in ids]
    frozen_hash = validation_artifact["sha256"]
    output_root.mkdir(parents=True, exist_ok=args.resume_teacher_rejections)
    teacher_context = teacher_decision = None
    if args.teacher_secret:
        if args.strategies != ["adaptive"] or not args.learnable_only:
            raise ValueError("teacher requires adaptive-only mixed-success training sources")
        from veripy_agent.data.teacher_curriculum import (
            generate_teacher_candidates,
            request_curriculum,
        )
        from veripy_agent.data.teacher_curriculum import (
            teacher_context as make_context,
        )

        teacher_context = make_context(
            clusters,
            failures,
            train_rows,
            args.receipts_dir,
            count=args.count,
            source_hash=_sha256(args.source_launch_manifest),
            public_contract_version=args.public_contract_version,
            behavior_targeted=args.behavior_targeted,
        )
        if scope is not None:
            teacher_context.update(
                required_measurement=scope["required_measurement"],
                operation_quotas=scope["operation_quotas"],
                behavior_scope_sha256=_sha256(args.behavior_scope),
            )
            teacher_context["constraints"].extend(
                [
                    "Only computation_and_submit allocations; no recovery or forced errors.",
                    "Operation quotas are a common preregistered arm balance, not a teacher decision. "
                    "Choose cluster, input condition and generator index yourself within them.",
                ]
            )
        if args.prior_teacher_decision:
            teacher_context["previous_teacher_decision"] = json.loads(
                args.prior_teacher_decision.read_text()
            )
            teacher_context["previous_teacher_decision_sha256"] = _sha256(
                args.prior_teacher_decision
            )
        if args.teacher_harness:
            from veripy_agent.data.teacher_harness import (
                attach_harness,
                attach_retrieval,
                bind_source_checkpoint,
            )

            teacher_context = attach_harness(
                teacher_context,
                train_rows,
                args.receipts_dir,
                binding={
                    "source_launch_sha256": _sha256(args.source_launch_manifest),
                    "source_train_sha256": _sha256(args.base_train),
                    "source_receipts_sha256": receipts_sha256,
                    "source_checkpoint": str(
                        args.source_launch_manifest.parent / f"global_step_{launch['steps']}"
                    ),
                    "source_model_base": overrides.get("actor_rollout_ref.model.path"),
                    "source_checkpoint_snapshot": bind_source_checkpoint(
                        args.source_launch_manifest
                    ),
                    "purpose": args.harness_purpose,
                },
            )
            if args.retrieval_train:
                if input_exclusion is None:
                    raise ValueError("retrieval exclusions missing")
                if _sha256(args.retrieval_train) != args.retrieval_train_sha256:
                    raise ValueError("retrieval pool parquet hash changed")
                teacher_context = attach_retrieval(
                    teacher_context,
                    pq.read_table(args.retrieval_train).to_pylist(),
                    exclusions=input_exclusion,
                )
        teacher_decision = request_curriculum(
            teacher_context,
            secret=args.teacher_secret,
            output=output_root / "teacher",
            resume_rejections=args.resume_teacher_rejections,
            max_attempts=args.teacher_max_attempts,
        )
    failure_path = output_root / "train_failures.jsonl"
    with failure_path.open("x", encoding="utf-8") as failure_file:
        failure_file.write("".join(json.dumps(row, sort_keys=True) + "\n" for row in failures))
    summaries = {}
    strategy_rows = {}
    strategy_signatures: set[str] = set()
    strategy_seeds: set[int] = set()
    for strategy in dict.fromkeys(args.strategies):
        generated = (
            generate_teacher_candidates(teacher_context, teacher_decision, base_seed=args.seed)
            if teacher_decision
            else build_curriculum_candidates(
                clusters, count=args.count, base_seed=args.seed, strategy=strategy
            )
        )
        accepted, rejected = admit_curriculum_candidates(
            generated, forbidden_content_signatures=existing_signatures | strategy_signatures
        )
        if rejected or len(accepted) != args.count:
            raise ValueError(f"{strategy} candidate admission failed: {len(rejected)} rejected")
        if input_exclusion is not None:
            validate_input_exclusion(accepted, input_exclusion)
        seeds = {int(row["mutation_seed"]) for row in accepted}
        if seeds & (existing_seeds | strategy_seeds):
            raise ValueError(f"{strategy} candidate seeds overlap frozen corpus or paired arm")
        strategy_seeds.update(seeds)
        strategy_signatures.update(row["admission"]["content_signature"] for row in accepted)
        arm_root = output_root / strategy
        arm_root.mkdir()
        added = _materialize(accepted, output_root=arm_root, dataset_root=dataset_root)
        strategy_rows[strategy] = added
        frame = pd.DataFrame(anchor_rows + added)
        path = arm_root / "train.parquet"
        frame.to_parquet(path, index=False)
        manifest = curriculum_manifest(
            accepted,
            strategy=strategy,
            corpus_sha256=_sha256(args.corpus),
            failure_records_sha256=_sha256(failure_path),
            frozen_validation_sha256=frozen_hash,
        )
        manifest.update(
            {
                "source_launch_manifest_sha256": _sha256(args.source_launch_manifest),
                "source_train_sha256": _sha256(args.base_train),
                "source_receipts_sha256": receipts_sha256,
                "failure_records_path": str(failure_path),
                "source_receipt_count": actual_receipts,
                "source_failure_count": len(failures),
                "learnable_only": args.learnable_only,
                "behavior_targeted": args.behavior_targeted,
                "teacher_harness": args.teacher_harness,
                "harness_purpose": args.harness_purpose,
                "retrieval_train_sha256": args.retrieval_train_sha256,
                "student_sampling_status": "not_run_learning_value_unverified"
                if args.teacher_harness
                else None,
                "training_authorized": False if args.teacher_harness else None,
                "behavior_scope_sha256": _sha256(args.behavior_scope) if scope else None,
                "input_exclusion_sha256": _sha256(args.input_exclusion)
                if args.input_exclusion
                else None,
                "excluded_curriculum_manifests": excluded_manifests,
                "source_failure_clusters": clusters,
                "excluded_failure_clusters": excluded_clusters,
                "exclusion_reason": "ambiguous_public_statistical_convention_pending_review"
                if excluded_clusters
                else None,
                "anchor_task_ids": [row["extra_info"]["task_id"] for row in anchor_rows],
                "added_task_ids": [row["extra_info"]["task_id"] for row in added],
                "candidates": [
                    {
                        "task_id": row["task"]["task_id"],
                        "family": row["task"]["family"],
                        "mutation_seed": row["mutation_seed"],
                        "mutation_index": row["mutation_index"],
                        "parent_failure_family": row["parent_failure_family"],
                        "parent_failure_code": row["parent_failure_code"],
                        "parent_difficulty_bucket": row["parent_difficulty_bucket"],
                        "parent_operation_sequence": row["parent_operation_sequence"],
                        "parent_semantic_signature": row["parent_semantic_signature"],
                        "parent_task_id": row["parent_task_id"],
                        "content_signature": row["admission"]["content_signature"],
                        **(
                            {"behavior_design": row["behavior_design"]}
                            if "behavior_design" in row
                            else {}
                        ),
                        **(
                            {
                                "candidate_source": row["candidate_source"],
                                "retrieval_task_id": row["retrieval_task_id"],
                            }
                            if "candidate_source" in row
                            else {}
                        ),
                    }
                    for row in accepted
                ],
                "generation_seed": args.seed,
                "added_public_contract_version": args.public_contract_version,
                "anchors_public_contract_unchanged": True,
                "anchor_manifest_sha256": _sha256(args.anchor_manifest)
                if args.anchor_manifest
                else None,
                "train_parquet": str(path),
                "train_parquet_sha256": _sha256(path),
                "train_rows": len(frame),
                "rejected_count": 0,
                "generation_controller": "ai_teacher" if teacher_decision else "rule_mutator",
                "teacher_decision_path": str(output_root / "teacher/decision.json")
                if teacher_decision
                else None,
                "teacher_decision_sha256": _sha256(output_root / "teacher/decision.json")
                if teacher_decision
                else None,
                "teacher_provenance_sha256": _sha256(output_root / "teacher/provenance.json")
                if teacher_decision
                else None,
                "teacher_task_designs": [
                    {
                        "task_id": row["task"]["task_id"],
                        "cluster_id": row["teacher_cluster_id"],
                        "generator_index": row["mutation_index"],
                        "reason": row["teacher_reason"],
                        **(
                            {"behavior_design": row["behavior_design"]}
                            if "behavior_design" in row
                            else {}
                        ),
                    }
                    for row in accepted
                ]
                if teacher_decision
                else [],
            }
        )
        (arm_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summaries[strategy] = {
            "rows": len(frame),
            "added": len(added),
            "sha256": manifest["train_parquet_sha256"],
        }
    if set(strategy_rows) == {"adaptive", "random"} and len(strategy_rows["adaptive"]) != len(
        strategy_rows["random"]
    ):
        raise ValueError("adaptive and random arms differ in size")
    print(json.dumps(summaries, sort_keys=True))


if __name__ == "__main__":
    main()
