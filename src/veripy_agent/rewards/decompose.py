from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from veripy_agent.contracts import FailureCode


class RewardResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: float
    components: dict[str, float]
    failure_code: FailureCode | None = None
    terminated: bool = False
    tool_call_valid: bool = True
    submission_format_valid: bool = False
    task_success: bool = False
    evidence_consistent: bool = False
    partial_credit_score: float = Field(default=0, ge=0, le=1)
    error_recovered: bool = False
    repeated_no_progress: bool = False
    no_submission_at_limit: bool = False
    turn_count: int = Field(default=0, ge=0)
    tool_call_count: int = Field(default=0, ge=0)


def compute_reward(
    *,
    correct: bool,
    correctness_score: float | None = None,
    evidence_consistent: bool,
    format_valid: bool,
    invalid_call_count: int = 0,
    timeout_count: int = 0,
    security_violation: bool = False,
    tool_call_valid: bool = True,
    error_recovered: bool = False,
    turn_count: int = 0,
    tool_call_count: int = 0,
    repeated_no_progress_count: int = 0,
    no_submission_at_limit: bool = False,
) -> RewardResult:
    invalid_call_count = int(invalid_call_count)
    timeout_count = int(timeout_count)
    repeated_no_progress_count = int(repeated_no_progress_count)
    if (
        invalid_call_count < 0
        or timeout_count < 0
        or turn_count < 0
        or tool_call_count < 0
        or repeated_no_progress_count < 0
    ):
        raise ValueError("reward counters must be non-negative")
    score = float(correct if correctness_score is None else correctness_score)
    if not 0 <= score <= 1:
        raise ValueError("correctness_score must be between zero and one")
    if correct and score != 1:
        raise ValueError("correct answers must have correctness_score=1")
    if score > 0 and not format_valid:
        raise ValueError("partial credit requires a valid submission format")
    metrics: dict[str, Any] = {
        "tool_call_valid": bool(tool_call_valid),
        "submission_format_valid": bool(format_valid),
        "task_success": bool(correct),
        "evidence_consistent": bool(correct and evidence_consistent),
        "partial_credit_score": score,
        "error_recovered": bool(error_recovered),
        "repeated_no_progress": bool(repeated_no_progress_count),
        "no_submission_at_limit": bool(no_submission_at_limit),
        "turn_count": int(turn_count),
        "tool_call_count": int(tool_call_count),
    }
    if security_violation:
        return RewardResult(
            total=-1.0,
            components={"security_violation": -1.0},
            failure_code=FailureCode.SECURITY_VIOLATION,
            terminated=True,
            **metrics,
        )
    components = {
        "correctness": 1.0 if correct else 0.5 * score,
        "evidence": 0.2 if correct and evidence_consistent else 0.0,
        "valid_json": 0.05 if format_valid else 0.0,
        "invalid_call": -0.05 * invalid_call_count,
        "timeout": -0.1 * timeout_count,
        "repeated_no_progress": -0.05 * repeated_no_progress_count,
        "no_submission": -0.1 if no_submission_at_limit else 0.0,
    }
    code = None
    if no_submission_at_limit:
        code = FailureCode.NO_SUBMISSION
    elif timeout_count:
        code = FailureCode.TIMEOUT
    elif invalid_call_count:
        code = FailureCode.INVALID_REQUEST
    elif repeated_no_progress_count:
        code = FailureCode.REPEATED_NO_PROGRESS
    elif not correct:
        code = FailureCode.INCORRECT if format_valid else FailureCode.INVALID_FORMAT
    return RewardResult(
        total=sum(components.values()),
        components=components,
        failure_code=code,
        **metrics,
    )
