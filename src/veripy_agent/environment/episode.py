from __future__ import annotations

from collections.abc import Callable
from typing import Any

from veripy_agent.contracts import Submission, TaskSpec, ToolResult, VerifyResult
from veripy_agent.rewards import RewardResult, compute_reward
from veripy_agent.rollout import EpisodeResult, ToolName, run_episode
from veripy_agent.rollout.runner import Agent
from veripy_agent.tools import execute_python, inspect_files, inspect_table, submit_answer
from veripy_agent.verifiers import verify_task_submission

from .paths import SandboxPaths
from .worker import PythonWorker

SubmissionVerifier = Callable[[TaskSpec, Submission], tuple[VerifyResult, bool]]


class EpisodeEnvironment:
    """Deep module for one isolated task episode and its terminal verification."""

    def __init__(
        self,
        task: TaskSpec,
        paths: SandboxPaths,
        *,
        worker: PythonWorker,
        max_turns: int = 6,
        submission_verifier: SubmissionVerifier = verify_task_submission,
    ):
        self.task = task
        self.paths = paths
        self.worker = worker
        self.max_turns = max_turns
        self.submission_verifier = submission_verifier

    def route(self, tool: ToolName, arguments: dict[str, Any]) -> ToolResult:
        if tool is ToolName.INSPECT_FILES:
            return inspect_files(self.paths)
        if tool is ToolName.INSPECT_TABLE:
            return inspect_table(self.paths, **arguments)
        if tool is ToolName.EXECUTE_PYTHON:
            return execute_python(self.paths, worker=self.worker, **arguments)
        if tool is ToolName.SUBMIT_ANSWER:
            return submit_answer(arguments)
        raise ValueError(f"unsupported tool: {tool}")

    def run(self, agent: Agent) -> tuple[EpisodeResult, RewardResult]:
        try:
            result = run_episode(self.task, agent, self.route, max_turns=self.max_turns)
            return result, self._reward(result)
        finally:
            close = getattr(self.worker, "close", None)
            if callable(close):
                close()

    def _reward(self, result: EpisodeResult) -> RewardResult:
        invalid_calls = sum(
            step.reward_components.get("invalid_call", 0) < 0 for step in result.steps
        )
        timeouts = sum(
            step.tool_result is not None and step.tool_result.status.value == "timeout"
            for step in result.steps
        )
        security = result.termination_reason == "security_violation"
        tool_calls = sum(step.tool_call is not None for step in result.steps)
        recovered = bool(
            result.submission is not None
            and any(
                step.tool_result is not None
                and step.tool_result.status.value in {"error", "timeout"}
                for step in result.steps
            )
        )
        repeated = 0
        for previous, current in zip(result.steps, result.steps[1:], strict=False):
            if (
                previous.tool_call is not None
                and previous.tool_call == current.tool_call
                and previous.tool_result is not None
                and current.tool_result is not None
                and all(
                    getattr(previous.tool_result, field) == getattr(current.tool_result, field)
                    for field in ("status", "stdout", "stderr", "artifacts", "failure_code")
                )
            ):
                repeated += 1
        if result.submission is None:
            return compute_reward(
                correct=False,
                evidence_consistent=False,
                format_valid=False,
                invalid_call_count=invalid_calls,
                timeout_count=timeouts,
                security_violation=security,
                tool_call_valid=invalid_calls == 0,
                error_recovered=False,
                turn_count=len(result.steps),
                tool_call_count=tool_calls,
                repeated_no_progress_count=repeated,
                no_submission_at_limit=result.termination_reason == "max_turns",
            )
        verified, evidence_consistent = self.submission_verifier(self.task, result.submission)
        return compute_reward(
            correct=verified.passed,
            correctness_score=verified.score,
            evidence_consistent=evidence_consistent,
            format_valid=True,
            invalid_call_count=invalid_calls,
            timeout_count=timeouts,
            security_violation=security,
            tool_call_valid=invalid_calls == 0,
            error_recovered=recovered and verified.passed,
            turn_count=len(result.steps),
            tool_call_count=tool_calls,
            repeated_no_progress_count=repeated,
        )


WorkerFactory = Callable[[SandboxPaths], PythonWorker]
