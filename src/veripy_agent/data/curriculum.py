"""Offline failure-driven curriculum candidate construction."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping
from typing import Any

from veripy_agent.contracts import TaskSpec

from .generator import FAMILIES, generate_tasks


def _seed(base_seed: int, family: str, ordinal: int, strategy: str) -> int:
    value = hashlib.sha256(f"{base_seed}|{family}|{ordinal}|{strategy}".encode()).hexdigest()
    return int(value[:8], 16)


def _difficulty_bucket(score: int) -> str:
    return "low" if score <= 4 else "medium" if score <= 8 else "high"


def build_curriculum_candidates(
    clusters: Iterable[Mapping[str, Any]],
    *,
    count: int,
    base_seed: int,
    strategy: str,
) -> list[dict[str, Any]]:
    if strategy not in {"adaptive", "random"}:
        raise ValueError("strategy must be adaptive or random")
    if count < 1:
        raise ValueError("count must be positive")
    cluster_rows = [row for row in clusters if str(row.get("family")) in FAMILIES]
    if strategy == "adaptive":
        eligible_clusters = [row for row in cluster_rows if row.get("failure_code") != "success"]
        # Proportional quotas, not a prefix of count-expanded sorted clusters.
        total = sum(max(1, int(row.get("count", 1))) for row in eligible_clusters)
        quotas = [count * max(1, int(row.get("count", 1))) // total for row in eligible_clusters] if total else []
        order = sorted(range(len(quotas)), key=lambda i: (
            -(count * max(1, int(eligible_clusters[i].get("count", 1))) % total), i
        ))
        for index in order[:count - sum(quotas)]:
            quotas[index] += 1
        parents = [row for row, quota in zip(eligible_clusters, quotas, strict=True) for _ in range(quota)]
        random.Random(base_seed).shuffle(parents)
        if not parents:
            raise ValueError("adaptive curriculum requires at least one failure cluster")
    else:
        families = list(FAMILIES)
        random.Random(base_seed).shuffle(families)
        parents = [{"family": family} for family in families]
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    ordinal = 0
    while len(candidates) < count:
        parent = parents[ordinal % len(parents)]
        family = str(parent["family"])
        seed = _seed(base_seed, family, ordinal, strategy)
        variants = generate_tasks(family, seed=seed, count=3, split="rl_train")
        target_bucket = parent.get("difficulty_bucket") if strategy == "adaptive" else None
        target_operations = parent.get("operation_sequence") if strategy == "adaptive" else None
        target_semantic = parent.get("semantic_signature") if strategy == "adaptive" else None
        eligible = [
            index for index, variant in enumerate(variants)
            if target_bucket is None or _difficulty_bucket(
                int(variant.task_spec.verifier_spec["structure"]["difficulty_score"])
            ) == target_bucket
            if target_operations is None or variant.task_spec.verifier_spec["structure"]["operation_sequence"] == target_operations
            if target_semantic is None or variant.semantic_signature == target_semantic
        ]
        if not eligible:
            raise ValueError(f"no {target_bucket} mutation available for {family}")
        mutation_index = eligible[ordinal % len(eligible)]
        task = variants[mutation_index]
        ordinal += 1
        if task.content_signature in seen:
            continue
        seen.add(task.content_signature)
        task_payload = task.to_dict()
        # Candidate manifests are public training inputs. The oracle stays in
        # the independent generator/verifier and must never enter a candidate
        # or later model observation.
        task_payload.pop("oracle", None)
        task_payload["verifier_spec"] = {
            key: value
            for key, value in task_payload["verifier_spec"].items()
            if key != "expected"
        }
        task_payload["task_spec"] = task.task_spec.model_copy(
            update={"verifier_spec": task_payload["verifier_spec"]}
        ).model_dump(mode="json")
        candidates.append(
            {
                "task": task_payload,
                "strategy": strategy,
                "parent_failure_family": family if strategy == "adaptive" else None,
                "mutation_seed": seed,
                "mutation_index": mutation_index,
                "parent_failure_code": parent.get("failure_code") if strategy == "adaptive" else None,
                "parent_difficulty_bucket": parent.get("difficulty_bucket") if strategy == "adaptive" else None,
                "parent_operation_sequence": target_operations,
                "parent_semantic_signature": target_semantic,
                "parent_task_id": (
                    parent["task_ids"][ordinal % len(parent["task_ids"])]
                    if strategy == "adaptive" and parent.get("task_ids") else None
                ),
            }
        )
    return candidates


def admit_curriculum_candidates(
    candidates: Iterable[Mapping[str, Any]],
    *,
    forbidden_content_signatures: Iterable[str] = (),
    forbidden_semantic_signatures: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fail-closed CPU admission for generated curriculum candidates."""

    forbidden_content = set(forbidden_content_signatures)
    forbidden_semantic = set(forbidden_semantic_signatures)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_content: set[str] = set()
    for row in candidates:
        candidate = dict(row)
        try:
            task_payload = dict(candidate["task"])
            if "oracle" in task_payload or "expected" in task_payload.get("verifier_spec", {}):
                raise ValueError("oracle leakage")
            task_spec = TaskSpec.model_validate(task_payload["task_spec"])
            if task_spec.seed is None:
                raise ValueError("generated curriculum task requires a seed")
            index = int(task_spec.verifier_spec["generator_index"])
            generated = generate_tasks(
                task_spec.family,
                seed=task_spec.seed,
                count=index + 1,
                split=str(task_spec.split),
                public_contract_version=task_spec.verifier_spec.get("public_contract_version"),
            )[index]
            if generated.task_id != task_spec.task_id:
                raise ValueError("task identity mismatch")
            if generated.content_signature in forbidden_content:
                raise ValueError("content overlap with frozen corpus")
            # Template/operation overlap is allowed within the same adaptive
            # curriculum: it is the skill family being repaired. Concrete
            # content overlap remains the leakage boundary.
            if generated.content_signature in seen_content:
                raise ValueError("duplicate candidate content")
            from veripy_agent.verifiers.dispatcher import verify_answer
            if not verify_answer(task_spec, generated.oracle).passed:
                raise ValueError("independent verifier rejected generated oracle")
            candidate["admission"] = {
                "status": "accepted",
                "content_signature": generated.content_signature,
                "semantic_signature": generated.semantic_signature,
                "semantic_template_overlap": generated.semantic_signature in forbidden_semantic,
                "verifier": "seed_regeneration_consistency_not_independent_calculation",
            }
            accepted.append(candidate)
            seen_content.add(generated.content_signature)
        except Exception as exc:
            candidate["admission"] = {"status": "rejected", "reason": str(exc)}
            rejected.append(candidate)
    return accepted, rejected


def curriculum_manifest(
    candidates: list[dict[str, Any]],
    *,
    strategy: str,
    corpus_sha256: str,
    failure_records_sha256: str,
    frozen_validation_sha256: str,
) -> dict[str, Any]:
    payload = json.dumps(candidates, sort_keys=True, separators=(",", ":"))
    return {
        "schema": "veripy.curriculum.candidates",
        "strategy": strategy,
        "count": len(candidates),
        "candidates_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "corpus_sha256": corpus_sha256,
        "failure_records_sha256": failure_records_sha256,
        "frozen_validation_sha256": frozen_validation_sha256,
        "benchmark_feedback_used": False,
    }
