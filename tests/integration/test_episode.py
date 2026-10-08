from types import SimpleNamespace

import pytest

from veripy_agent.contracts import FailureCode, TaskSpec, ToolResult, ToolStatus
from veripy_agent.environment import EpisodeEnvironment
from veripy_agent.rollout import AgentAction, ToolName, run_episode
from veripy_agent.tools import submit_answer


def task() -> TaskSpec:
    return TaskSpec(
        task_id="task-1",
        family="filter_aggregate",
        instruction="Compute the requested value.",
        files=["table.csv"],
        seed=1,
        split="validation",
        difficulty="easy",
        answer_schema={"type": "object"},
        verifier_spec={"kind": "exact"},
    )


def test_episode_submits_structured_answer() -> None:
    payload = {
        "answer": {"value": 3},
        "evidence": {"columns": ["value"], "filters": [], "intermediate_values": {}},
    }
    result = run_episode(
        task(),
        lambda *_args, **_kwargs: AgentAction(tool=ToolName.SUBMIT_ANSWER, arguments=payload),
        lambda name, args: (
            submit_answer(args)
            if name is ToolName.SUBMIT_ANSWER
            else ToolResult(status=ToolStatus.OK)
        ),
        max_turns=1,
    )
    assert result.submitted
    assert result.submission is not None


def test_episode_stops_at_turn_limit() -> None:
    result = run_episode(
        task(),
        lambda *_args, **_kwargs: AgentAction(tool=ToolName.INSPECT_FILES),
        lambda *_args: ToolResult(status=ToolStatus.OK, stdout="[]"),
        max_turns=2,
    )
    assert result.termination_reason == "max_turns"
    assert len(result.steps) == 2


def test_episode_environment_penalizes_repetition_and_no_submission() -> None:
    environment = EpisodeEnvironment(
        task(),
        SimpleNamespace(),
        worker=SimpleNamespace(close=lambda: None),
        max_turns=2,
    )
    environment.route = lambda *_args: ToolResult(status=ToolStatus.OK, stdout="[]")
    result, reward = environment.run(
        lambda *_args, **_kwargs: AgentAction(tool=ToolName.INSPECT_FILES)
    )
    assert result.termination_reason == "max_turns"
    assert reward.total == pytest.approx(-0.15)
    assert reward.repeated_no_progress
    assert reward.no_submission_at_limit
    assert reward.failure_code is FailureCode.NO_SUBMISSION
