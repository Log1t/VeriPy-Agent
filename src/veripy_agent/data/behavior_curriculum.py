"""Observed behavior evidence and input opportunities, never policy or answer repair."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from veripy_agent.rollout import ToolName
from veripy_agent.training.agentic import sha

from .generator import generate_tasks


def input_properties(data):
    """Properties of public input only: no oracle, verifier, or policy outcome."""
    props = {"ordinary_input": True}
    op = data.get("operation")
    if op in ("median_imputation_sum", "mean_imputation_sum"):
        observed = [v for v in data["values"] if v is not None]
        fill = (
            statistics.median(observed)
            if data["strategy"] == "median"
            else statistics.mean(observed)
        )
        props["missing_even_observed"] = bool(observed) and len(observed) % 2 == 0
        props["missing_fractional_fill"] = float(fill).is_integer() is False
    if op in ("customer_order_totals", "customer_order_average"):
        amounts = [
            [o["amount"] for o in data["orders"] if o["customer_id"] == c["customer_id"]]
            for c in data["customers"]
        ]
        props["join_empty_customer"] = any(not a for a in amounts)
        if op == "customer_order_average":
            props["join_fractional_average"] = any(a and sum(a) % len(a) != 0 for a in amounts)
    if op in ("group_max_topk", "group_sum_topk"):
        groups: dict[str, list[int]] = {}
        for row in data["rows"]:
            groups.setdefault(row["group"], []).append(row["score"])
        aggregate = max if op == "group_max_topk" else sum
        scores = sorted((aggregate(v) for v in groups.values()), reverse=True)
        # A tie inside the returned top-k or crossing its cutoff affects the output.
        props["topk_relevant_tie"] = any(
            scores[i] == scores[i + 1] for i in range(min(data["k"], len(scores) - 1))
        )
    if op == "median_mad":
        median = statistics.median(data["values"])
        props["mad_nonzero"] = statistics.median([abs(v - median) for v in data["values"]]) > 0
    if op == "population_covariance_direction":
        props["population_covariance"] = len(data["rows"]) > 1
    if op == "quartiles_iqr":
        props["explicit_quartile_halves"] = len(data["values"]) % 2 == 1
    return props


def condition_catalog(task):
    props = input_properties(task.input)
    # Parity is fixed by index, unlike random values: never advertise an impossible condition.
    conditions = [key for key in props if key != "missing_even_observed" or props[key]]
    if (
        task.input.get("operation") == "median_imputation_sum"
        and not props["missing_even_observed"]
    ):
        conditions.remove("missing_fractional_fill")
    return conditions


def realize_condition(family, index, initial_seed, condition, profile):
    for attempt in range(512):
        seed = (
            initial_seed
            if attempt == 0
            else int(
                hashlib.sha256(f"{initial_seed}|condition|{attempt}".encode()).hexdigest()[:8], 16
            )
        )
        task = generate_tasks(
            family, seed=seed, count=index + 1, split="rl_train", public_contract_version=profile
        )[index]
        if condition not in condition_catalog(task):
            raise ValueError("unsupported input condition for template/index")
        if input_properties(task.input)[condition]:
            return task, attempt + 1
    raise ValueError("bounded input-condition seed search exhausted; no fallback course")


def episode_features(receipt):
    steps = receipt["steps"]
    actions = [s["action"] for s in steps]
    marker = any(
        word in "\n".join(actions).lower()
        for word in (
            "intentional",
            "nonexistent",
            "deliberate",
            "raise valueerror",
            "raise runtimeerror",
            "recoverable teacher error",
        )
    )
    errors = [i for i, s in enumerate(steps) if s.get("status") not in ("ok", None)]
    # Recovery evidence concerns repeated failed attempts, not adjacency.
    # A successful calculation between two identical malformed submissions
    # must not hide their repetition; failure then success is not two failures.
    failed_actions = [actions[i] for i in errors]
    python_ok = sum(
        s.get("tool") == ToolName.EXECUTE_PYTHON.value and s.get("status") == "ok" for s in steps
    )
    return {
        "explicit_self_induced_or_raise_marker": marker,
        "successful_python_calls": python_ok,
        "submitted_without_successful_python": python_ok == 0
        and any(s.get("tool") == "submit_answer" for s in steps),
        "first_error_turn": steps[errors[0]]["turn"] if errors else None,
        "unmarked_error": bool(errors) and not marker,
        "repeated_failed_action": len(failed_actions) != len(set(failed_actions)),
        "no_progress_steps": sum(bool(s.get("no_progress")) for s in steps),
        "marker_absence_proves_natural_error": False,
    }


def episode_evidence(receipt, path):
    steps = []
    for step in receipt["steps"]:
        result = step.get("result") or {}
        fields = {
            "action": step["action"],
            "stdout": str(result.get("stdout", "")),
            "stderr": str(result.get("stderr", "")),
        }
        steps.append(
            {
                "turn": step["turn"],
                "tool": step.get("tool"),
                "status": step.get("status"),
                "no_progress": bool(step.get("no_progress")),
                **{k: v[:6000] for k, v in fields.items()},
                "truncated_fields": [k for k, v in fields.items() if len(v) > 6000],
            }
        )
    return {
        "task_id": receipt["task_id"],
        "receipt_sha256": sha(path),
        "won": bool(receipt["won"]),
        "steps": steps,
        "features": episode_features(receipt),
        "feedback_scope": "captured_tool_result_text_not_hidden_verifier_diagnostics",
    }


def coverage(receipts):
    groups: dict[str, list[dict[str, Any]]] = {}
    counts: Counter[str] = Counter()
    families: dict[str, dict[str, int]] = {}
    for receipt in receipts:
        groups.setdefault(receipt["task_id"], []).append(receipt)
        family = families.setdefault(receipt["family"], {"episodes": 0, "success": 0})
        family["episodes"] += 1
        family["success"] += bool(receipt["won"])
        counts["episodes"] += 1
        counts["success"] += bool(receipt["won"])
        for key, value in episode_features(receipt).items():
            if isinstance(value, bool) and value:
                counts[key] += 1
    for group in groups.values():
        wins = sum(bool(r["won"]) for r in group)
        counts[
            "all_success_groups"
            if wins == len(group)
            else "all_failure_groups"
            if wins == 0
            else "mixed_groups"
        ] += 1
        rewards = [r["episode_reward"] for r in group]
        counts["zero_episode_reward_variance_groups"] += len(set(rewards)) == 1
    return {
        **counts,
        "task_groups": len(groups),
        "families": families,
        "episode_reward_variance_is_grpo_advantage": False,
    }


def validate_behavior_roster(receipts, task_ids, rollouts_per_task):
    counts = Counter(r["task_id"] for r in receipts)
    if set(counts) != set(task_ids) or any(n != rollouts_per_task for n in counts.values()):
        raise ValueError("behavior source roster must contain every training task exactly n times")


def enrich_context(context, receipt_root, failures):
    paths = sorted(receipt_root.glob("*.json"))
    by_task: dict[str, list[tuple[dict[str, Any], Path]]] = {}
    receipts = []
    for path in paths:
        receipt = json.loads(path.read_text())
        receipts.append(receipt)
        by_task.setdefault(receipt["task_id"], []).append((receipt, path))
    # Re-select distinct failed tasks: two rollouts of one task hide coverage.
    for cluster in context["clusters"]:
        for design in cluster["allowed_designs"]:
            task = generate_tasks(
                cluster["family"],
                seed=20261005,
                count=design["generator_index"] + 1,
                split="rl_train",
                public_contract_version=context["public_contract_version"],
            )[design["generator_index"]]
            design["input_conditions"] = condition_catalog(task)
        examples = []
        seen = set()
        public = {e["task_id"]: e for e in cluster["real_failure_examples"]}
        for failure in failures:
            tid = failure["task_id"]
            if tid not in cluster["task_ids"] or tid in seen or tid not in public:
                continue
            seen.add(tid)
            path = Path(failure["receipt_path"])
            if sha(path) != failure["receipt_sha256"]:
                raise ValueError("behavior source receipt hash changed")
            failed = episode_evidence(json.loads(path.read_text()), path)
            siblings = [(r, p) for r, p in by_task[tid] if r["won"]]
            siblings.sort(
                key=lambda pair: (
                    episode_features(pair[0])["explicit_self_induced_or_raise_marker"],
                    -episode_features(pair[0])["successful_python_calls"],
                    pair[1].name,
                )
            )
            failed.update({k: public[tid][k] for k in ("public_instruction", "answer_schema")})
            failed["successful_siblings"] = [episode_evidence(r, p) for r, p in siblings[:1]]
            examples.append(failed)
            if len(examples) == 2:
                break
        cluster["real_failure_examples"] = examples
    context["behavior_targeted"] = True
    context["input_condition_semantics"] = {
        "ordinary_input": "No extra input constraint; explain why this is sufficient.",
        "missing_even_observed": "Even count of non-null observed values.",
        "missing_fractional_fill": "Declared imputation statistic is non-integer.",
        "join_empty_customer": "At least one eligible customer has no orders; zero convention applies.",
        "join_fractional_average": "At least one non-empty customer average is non-integer.",
        "topk_relevant_tie": "Tie within returned top-k or crossing cutoff; ascending group tie rule applies.",
        "mad_nonzero": "Actual values yield nonzero median absolute deviation.",
        "population_covariance": "Multiple paired rows with declared population denominator n.",
        "explicit_quartile_halves": "Odd length input subject to explicit upper-middle halves convention.",
    }
    context["behavior_coverage"] = coverage(receipts)
    context["behavior_evidence_limits"] = [
        "Outcome labels are offline train labels, not feedback shown to the student.",
        "Marker absence does not establish a natural error. Explicit raises are separate.",
        "Successful siblings are observed alternatives, not proven causal repairs.",
        "No forced first error; avoidance and unmarked-error recovery must be measured separately.",
        "Episode reward variance is not the turn-weighted GRPO advantage.",
        "CPU verifies input opportunities and I/O only; teacher diagnosis is an unverified hypothesis.",
    ]
    return context
