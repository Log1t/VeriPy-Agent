from __future__ import annotations

import json
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from veripy_agent.contracts import (
    FailureCode,
    StepRecord,
    Submission,
    TaskSpec,
    ToolResult,
    ToolStatus,
)


class ToolName(StrEnum):
    INSPECT_FILES = "inspect_files"
    INSPECT_TABLE = "inspect_table"
    EXECUTE_PYTHON = "execute_python"
    SUBMIT_ANSWER = "submit_answer"


class AgentAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: ToolName
    arguments: dict[str, Any] = Field(default_factory=dict)


class EpisodeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    submission: Submission | None = None
    steps: list[StepRecord] = Field(default_factory=list)
    termination_reason: str
    submitted: bool = False


ToolRouter = Callable[[ToolName, dict[str, Any]], ToolResult]
Agent = Callable[..., AgentAction | dict[str, Any]]


def run_episode(
    task: TaskSpec, agent: Agent, tool_router: ToolRouter, *, max_turns: int = 6
) -> EpisodeResult:
    if max_turns <= 0:
        raise ValueError("max_turns must be positive")
    observation: Any = {"instruction": task.instruction, "files": task.files}
    steps: list[StepRecord] = []
    for _ in range(max_turns):
        raw_action = agent(task, observation=observation, steps=steps)
        try:
            if isinstance(raw_action, dict) and "raw_output" in raw_action:
                raise ValueError(raw_action.get("parse_error") or "invalid tool call")
            action = (
                raw_action
                if isinstance(raw_action, AgentAction)
                else AgentAction.model_validate(raw_action)
            )
        except Exception as exc:
            result = ToolResult(
                status=ToolStatus.ERROR, stderr=str(exc), failure_code=FailureCode.INVALID_REQUEST
            )
            next_observation = result.model_dump(mode="json")
            steps.append(
                StepRecord(
                    observation=observation,
                    assistant_action=raw_action,
                    tool_result=result,
                    reward_components={"invalid_call": -0.05},
                    next_observation=next_observation,
                )
            )
            observation = next_observation
            continue

        try:
            result = tool_router(action.tool, action.arguments)
        except Exception as exc:
            result = ToolResult(
                status=ToolStatus.ERROR,
                stderr=str(exc),
                failure_code=FailureCode.INVALID_REQUEST,
            )
        next_observation = result.model_dump(mode="json")
        security_violation = result.failure_code is FailureCode.SECURITY_VIOLATION
        steps.append(
            StepRecord(
                observation=observation,
                assistant_action=action.model_dump(mode="json"),
                tool_call={"name": action.tool.value, "arguments": action.arguments},
                tool_result=result,
                reward_components=(
                    {"invalid_call": -0.05}
                    if result.failure_code is FailureCode.INVALID_REQUEST
                    else {}
                ),
                terminated=security_violation,
                next_observation=next_observation,
            )
        )
        if security_violation:
            return EpisodeResult(
                task_id=task.task_id, steps=steps, termination_reason="security_violation"
            )
        if action.tool is ToolName.SUBMIT_ANSWER and result.status is ToolStatus.OK:
            try:
                submission = Submission.model_validate(json.loads(result.stdout))
            except Exception:
                observation = next_observation
                continue
            return EpisodeResult(
                task_id=task.task_id,
                submission=submission,
                steps=steps,
                termination_reason="submitted",
                submitted=True,
            )
        observation = next_observation
    if steps:
        steps[-1] = steps[-1].model_copy(update={"terminated": True})
    return EpisodeResult(task_id=task.task_id, steps=steps, termination_reason="max_turns")
