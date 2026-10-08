from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StepStatus(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class FailureCode(StrEnum):
    INVALID_PATH = "invalid_path"
    INVALID_REQUEST = "invalid_request"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"
    SECURITY_VIOLATION = "security_violation"
    REPLAY_FAILED = "replay_failed"
    VERIFIER_ERROR = "verifier_error"
    NO_TOOL = "no_tool"
    TOOL_ERROR = "tool_error"
    INCORRECT = "incorrect"
    INVALID_FORMAT = "invalid_format"
    REPEATED_NO_PROGRESS = "repeated_no_progress"
    NO_SUBMISSION = "no_submission"


class ToolStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    SECURITY_VIOLATION = "security_violation"


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    family: str
    instruction: str
    files: list[str] = Field(default_factory=list)
    seed: int | None = None
    split: str | None = None
    difficulty: str | None = None
    answer_schema: dict[str, Any]
    verifier_spec: dict[str, Any]


class StepRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observation: Any = None
    assistant_action: Any = None
    tool_call: Any = None
    tool_result: ToolResult | None = None
    reward_components: dict[str, float] = Field(default_factory=dict)
    terminated: bool = False
    next_observation: Any = None


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: ToolStatus
    stdout: str = ""
    stderr: str = ""
    artifacts: list[str] = Field(default_factory=list)
    duration: float = 0.0
    resource_usage: dict[str, Any] = Field(default_factory=dict)
    failure_code: FailureCode | None = None


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    columns: list[str] = Field(default_factory=list)
    filters: list[dict[str, Any] | str] = Field(default_factory=list)
    intermediate_values: dict[str, Any] = Field(default_factory=dict)


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: Any
    evidence: Evidence


class VerifyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    score: float = Field(ge=0, le=1)
    failure_code: FailureCode | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
