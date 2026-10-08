from __future__ import annotations

from typing import Any

from veripy_agent.contracts import Evidence, Submission, TaskSpec, VerifyResult
from veripy_agent.data.generator import generate_tasks

from .basic import verify_exact, verify_numeric, verify_table, verify_unordered

REQUIRED_COLUMNS: dict[str, set[str]] = {
    "filter_aggregate": {"category", "value"},
    "group_rank_topk": {"group", "score"},
    "multi_table_join": {"customer_id", "amount"},
    "datetime_timeseries": {"date", "value"},
    "missing_values": {"value"},
    "outliers_robust_stats": {"value"},
    "correlation_inference": {"x", "y"},
    "multi_constraint_decision": {"name", "cost", "quality", "risk"},
}


def _regenerated_task(task: TaskSpec):
    if task.seed is None or task.split is None:
        raise ValueError("task seed and split are required for independent verification")
    index = int(task.verifier_spec.get("generator_index", -1))
    if index < 0:
        raise ValueError("verifier spec is missing generator_index")
    generated = generate_tasks(
        task.family,
        seed=task.seed,
        count=index + 1,
        split=task.split,
        public_contract_version=task.verifier_spec.get("public_contract_version"),
    )
    candidate = generated[index]
    if candidate.task_id != task.task_id:
        raise ValueError("regenerated task identity mismatch")
    if task.verifier_spec.get("public_contract_version") and task.instruction != candidate.prompt:
        raise ValueError("public contract instruction mismatch")
    return candidate


def recompute_oracle(task: TaskSpec) -> Any:
    return _regenerated_task(task).oracle


def verify_answer(task: TaskSpec, actual: Any, expected: Any | None = None) -> VerifyResult:
    expected = recompute_oracle(task) if expected is None else expected
    spec = task.verifier_spec
    kind = spec.get("answer_kind", spec.get("kind", "exact"))
    if kind == "numeric":
        return verify_numeric(
            actual,
            expected,
            abs_tol=float(spec.get("abs_tol", 1e-6)),
            rel_tol=float(spec.get("rel_tol", 1e-6)),
        )
    if kind == "unordered":
        return verify_unordered(actual, expected)
    if kind == "table":
        return verify_table(actual, expected, ordered=bool(spec.get("ordered", False)))
    return verify_exact(actual, expected)


def verify_evidence(
    task: TaskSpec, evidence: Evidence, answer: Any, expected: Any | None = None
) -> bool:
    expected = recompute_oracle(task) if expected is None else expected
    if task.seed is None or task.split is None:
        return False
    required = set(
        task.verifier_spec.get("structure", {}).get(
            "required_columns", REQUIRED_COLUMNS.get(task.family, set())
        )
    )
    if not required.issubset(set(evidence.columns)):
        return False
    claimed_answer = evidence.intermediate_values.get("answer")
    if claimed_answer is None:
        return False
    # Arrow struct unions pad absent keys with nulls across heterogeneous tasks.
    # The deterministic TaskSpec retains the original public filter contract;
    # do not turn transport padding into requirements on the policy's evidence.
    required_filters = list(_regenerated_task(task).task_spec.verifier_spec.get("evidence_filters", []))
    if any(item not in evidence.filters for item in required_filters):
        return False
    return verify_answer(task, claimed_answer, expected).passed and verify_answer(task, answer, expected).passed


def verify_task_submission(
    task: TaskSpec, submission: Submission | dict[str, Any]
) -> tuple[VerifyResult, bool]:
    parsed = submission if isinstance(submission, Submission) else Submission.model_validate(submission)
    expected = recompute_oracle(task)
    result = verify_answer(task, parsed.answer, expected)
    evidence_consistent = result.passed and verify_evidence(
        task, parsed.evidence, parsed.answer, expected
    )
    return result, evidence_consistent


def verify_public_submission(
    task: TaskSpec, submission: Submission | dict[str, Any], expected: Any
) -> tuple[VerifyResult, bool]:
    if task.verifier_spec.get("kind") != "private_oracle":
        raise ValueError("public verification requires a private_oracle task")
    parsed = submission if isinstance(submission, Submission) else Submission.model_validate(submission)
    result = verify_answer(task, parsed.answer, expected)
    claimed_answer = parsed.evidence.intermediate_values.get("answer")
    evidence_consistent = (
        result.passed
        and bool(parsed.evidence.columns)
        and claimed_answer is not None
        and verify_answer(task, claimed_answer, expected).passed
    )
    return result, evidence_consistent
