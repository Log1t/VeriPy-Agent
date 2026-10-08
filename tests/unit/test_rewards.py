import pytest

from veripy_agent.contracts import FailureCode
from veripy_agent.rewards import compute_reward


def test_reward_exact_components() -> None:
    reward = compute_reward(
        correct=True,
        evidence_consistent=True,
        format_valid=True,
        error_recovered=True,
        turn_count=4,
        tool_call_count=3,
    )
    assert reward.total == pytest.approx(1.25)
    assert reward.components["correctness"] == 1.0
    assert reward.components["evidence"] == 0.2
    assert reward.submission_format_valid
    assert reward.task_success
    assert reward.evidence_consistent
    assert reward.error_recovered
    assert reward.turn_count == 4
    assert reward.tool_call_count == 3


def test_evidence_requires_correct_answer() -> None:
    reward = compute_reward(correct=False, evidence_consistent=True, format_valid=True)
    assert reward.total == pytest.approx(0.05)
    assert reward.components["evidence"] == 0.0


def test_penalties_accumulate() -> None:
    reward = compute_reward(
        correct=False,
        evidence_consistent=False,
        format_valid=True,
        invalid_call_count=2,
        timeout_count=1,
    )
    assert reward.total == pytest.approx(-0.15)
    assert reward.failure_code is FailureCode.TIMEOUT


def test_security_violation_overrides_other_components() -> None:
    reward = compute_reward(
        correct=False,
        correctness_score=0.5,
        evidence_consistent=True,
        format_valid=True,
        invalid_call_count=2,
        timeout_count=1,
        repeated_no_progress_count=1,
        no_submission_at_limit=True,
        security_violation=True,
    )
    assert reward.total == -1.0
    assert reward.terminated
    assert reward.failure_code is FailureCode.SECURITY_VIOLATION


def test_format_and_tool_call_validity_are_separate() -> None:
    reward = compute_reward(
        correct=False,
        evidence_consistent=False,
        format_valid=False,
        invalid_call_count=1,
        tool_call_valid=True,
    )
    assert reward.total == pytest.approx(-0.05)
    assert reward.tool_call_valid
    assert not reward.submission_format_valid
    assert not reward.task_success


def test_partial_credit_and_behavior_penalties_are_task_progress_based() -> None:
    reward = compute_reward(
        correct=False,
        correctness_score=0.5,
        evidence_consistent=False,
        format_valid=True,
        repeated_no_progress_count=1,
    )
    assert reward.components["correctness"] == pytest.approx(0.25)
    assert reward.components["repeated_no_progress"] == pytest.approx(-0.05)
    assert reward.components["no_submission"] == 0.0
    assert reward.total == pytest.approx(0.25)
    assert reward.partial_credit_score == 0.5
    assert reward.repeated_no_progress
    assert not reward.no_submission_at_limit
    assert reward.failure_code is FailureCode.REPEATED_NO_PROGRESS


def test_correctness_score_contract_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        compute_reward(
            correct=False,
            correctness_score=1.1,
            evidence_consistent=False,
            format_valid=True,
        )
    with pytest.raises(ValueError, match="correct answers"):
        compute_reward(
            correct=True,
            correctness_score=0.5,
            evidence_consistent=False,
            format_valid=True,
        )
    with pytest.raises(ValueError, match="valid submission format"):
        compute_reward(
            correct=False,
            correctness_score=0.5,
            evidence_consistent=False,
            format_valid=False,
        )
