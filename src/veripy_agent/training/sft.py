from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from veripy_agent.contracts import StepRecord, TaskSpec
from veripy_agent.environment import EpisodeEnvironment
from veripy_agent.rollout import (
    AgentAction,
    ToolName,
    assert_no_hidden_leakage,
    task_message_content,
)
from veripy_agent.rollout.messages import SYSTEM_PROMPT
from veripy_agent.training.supervision import require_input_sensitive_program, require_stdout_answer


class TrajectoryKind(StrEnum):
    DIRECT_SUCCESS = "direct_success"
    RECOVERABLE_EXECUTION_ERROR = "recoverable_execution_error"
    PLAN_CORRECTION = "plan_correction"


class TeacherConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2048

    def safe_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def config_hash(self) -> str:
        raw = json.dumps(self.safe_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    def resolve_api_key(self, environ: dict[str, str] | None = None) -> str:
        if not self.api_key_env:
            raise RuntimeError("teacher api_key_env is required")
        environment = os.environ if environ is None else environ
        key = environment.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"missing credential environment variable: {self.api_key_env}")
        return key


class TeacherResponseError(ValueError):
    def __init__(self, message: str, response_raw: bytes) -> None:
        super().__init__(message)
        self.response_raw = response_raw


def parse_teacher_json(content: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as original_error:
        stack: list[str] = []
        in_string = False
        escaped = False
        pairs = {"}": "{", "]": "["}
        for character in content:
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character in "{[":
                stack.append(character)
            elif character in "}]":
                if not stack or stack.pop() != pairs[character]:
                    raise original_error
        if in_string or not stack:
            raise original_error
        closers = {"{": "}", "[": "]"}
        value = json.loads(content + "".join(closers[item] for item in reversed(stack)))
    if not isinstance(value, dict):
        raise ValueError("teacher response must be a JSON object")
    return value


def load_teacher_secret_file(path: str | Path) -> tuple[TeacherConfig, dict[str, str]]:
    source = Path(path)
    text = source.read_text(encoding="utf-8-sig")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raw = {}
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                raise ValueError(f"invalid teacher credential line in {source}") from None
            key, value = stripped.split("=", 1)
            raw[key.strip()] = value.strip().strip('"').strip("'")
    if not isinstance(raw, dict):
        raise ValueError("teacher credential file must contain an object or assignments")
    normalized = {str(key).lower(): value for key, value in raw.items()}

    def required_string(*keys: str) -> str:
        value = next((normalized[key] for key in keys if key in normalized), None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("teacher credential file requires BASE_URL, API_KEY, and model")
        return value.strip()

    api_key = required_string("api_key", "key")
    base_url = required_string("base_url", "url")
    model = required_string("model")
    key_env = "VERIPY_TEACHER_API_KEY_RUNTIME"
    config = TeacherConfig(
        provider="openai_compatible",
        model=model,
        api_key_env=key_env,
        base_url=base_url,
        temperature=0.0,
        max_tokens=4096,
    )
    return config, {key_env: api_key}


class TeacherProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    config_sha256: str
    request_sha256: str
    response_sha256: str
    raw_response_path: str | None = None


class AdmittedTrajectory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: TaskSpec
    kind: TrajectoryKind
    actions: list[AgentAction]
    steps: list[StepRecord]
    termination_reason: str
    reward_components: dict[str, float]
    provenance: TeacherProvenance
    trajectory_sha256: str
    computation_check: dict[str, Any] = Field(default_factory=lambda: {"status": "legacy_not_checked"})


TOOLS = [
    {
        "name": "inspect_files",
        "description": "List available input files.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "inspect_table",
        "description": "Preview a CSV or JSON input file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "rows": {"type": "integer"}},
            "required": ["path"],
        },
    },
    {
        "name": "execute_python",
        "description": "Execute Python in the isolated analysis sandbox.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
    {
        "name": "submit_answer",
        "description": "Submit the structured answer and reproducible evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {},
                "evidence": {
                    "type": "object",
                    "properties": {
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "filters": {"type": "array"},
                        "intermediate_values": {"type": "object"},
                    },
                    "required": ["columns", "filters", "intermediate_values"],
                },
            },
            "required": ["answer", "evidence"],
        },
    },
]


def assign_trajectory_kinds(task_ids: Iterable[str]) -> dict[str, TrajectoryKind]:
    ordered = sorted(task_ids, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    count = len(ordered)
    recoverable = round(count * 0.2)
    correction = round(count * 0.1)
    if count >= 3:
        recoverable = max(1, recoverable)
        correction = max(1, correction)
    direct = count - recoverable - correction
    if direct < 0:
        raise ValueError("trajectory allocation is invalid")
    kinds = (
        [TrajectoryKind.DIRECT_SUCCESS] * direct
        + [TrajectoryKind.RECOVERABLE_EXECUTION_ERROR] * recoverable
        + [TrajectoryKind.PLAN_CORRECTION] * correction
    )
    return dict(zip(ordered, kinds, strict=True))


def trajectory_digest(task: TaskSpec, kind: TrajectoryKind, actions: list[AgentAction]) -> str:
    payload = {
        "task": task.model_dump(mode="json"),
        "kind": kind.value,
        "actions": [action.model_dump(mode="json") for action in actions],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def admit_trajectory(
    *,
    task: TaskSpec,
    kind: TrajectoryKind,
    actions: list[AgentAction],
    environment: EpisodeEnvironment,
    provenance: TeacherProvenance,
) -> AdmittedTrajectory:
    if not actions or len(actions) > 6:
        raise ValueError("trajectory must contain between 1 and at most 6 actions")
    if actions[-1].tool is not ToolName.SUBMIT_ANSWER:
        raise ValueError("trajectory final action must be submit_answer")
    action_iter = iter(actions)
    result, reward = environment.run(lambda *_args, **_kwargs: next(action_iter))
    if not result.submitted or reward.components.get("correctness") != 1.0:
        raise ValueError("trajectory failed terminal answer verification")
    if reward.components.get("evidence") != 0.2:
        raise ValueError("trajectory failed evidence verification")
    if result.termination_reason != "submitted":
        raise ValueError("trajectory did not terminate by submission")
    failures = [
        step.tool_result.failure_code.value
        for step in result.steps
        if step.tool_result is not None and step.tool_result.failure_code is not None
    ]
    successful_python_steps = [
        index
        for index, step in enumerate(result.steps)
        if step.tool_call is not None
        and step.tool_call["name"] == ToolName.EXECUTE_PYTHON.value
        and step.tool_result is not None
        and step.tool_result.status.value == "ok"
    ]
    if not successful_python_steps:
        raise ValueError("trajectory never completed a successful Python analysis")
    python_steps = [step for step in result.steps if step.tool_call is not None
                    and step.tool_call["name"] == ToolName.EXECUTE_PYTHON.value]
    latest_python = python_steps[-1].tool_result
    if latest_python is None or latest_python.status.value != "ok":
        raise ValueError("latest Python stdout is not a successful analysis")
    assert latest_python is not None and result.submission is not None
    require_stdout_answer(latest_python.stdout, result.submission.answer)
    computation_check: dict[str, Any]
    if "generator_index" in task.verifier_spec:
        replay = require_input_sensitive_program(task, python_steps[-1].tool_call["arguments"]["code"],
                                                result.submission.answer)
        computation_check = {"status": "two_different_answer_input_replays_passed", "replays": replay}
    else:
        # Public corpus answer programs have no validated counterfactual generator.
        # Preserve their terminal verification, but never claim input-sensitive computation.
        computation_check = {"status": "stdout_consistency_only_counterfactual_unsupported",
                             "autonomous_computation_verified": False}
    if kind is TrajectoryKind.DIRECT_SUCCESS and failures:
        raise ValueError("direct-success trajectory contains a tool failure")
    if kind is TrajectoryKind.RECOVERABLE_EXECUTION_ERROR and "execution_error" not in failures:
        raise ValueError("recovery trajectory is missing an execution error")
    if kind is TrajectoryKind.PLAN_CORRECTION and "invalid_path" not in failures:
        raise ValueError("correction trajectory is missing an invalid-path observation")
    return AdmittedTrajectory(
        task=task,
        kind=kind,
        actions=actions,
        steps=result.steps,
        termination_reason=result.termination_reason,
        reward_components=reward.components,
        provenance=provenance,
        trajectory_sha256=trajectory_digest(task, kind, actions),
        computation_check=computation_check,
    )


def to_llamafactory_record(trajectory: AdmittedTrajectory) -> dict[str, Any]:
    conversations = [
        {
            "from": "system",
            "value": SYSTEM_PROMPT,
        },
        {
            "from": "human",
            "value": task_message_content(trajectory.task),
        },
    ]
    for step in trajectory.steps:
        if step.tool_call is None or step.tool_result is None:
            continue
        conversations.append(
            {
                "from": "function_call",
                "value": json.dumps(
                    {
                        "name": step.tool_call["name"],
                        "arguments": step.tool_call["arguments"],
                    },
                    ensure_ascii=False,
                ),
            }
        )
        if step.tool_call["name"] == ToolName.SUBMIT_ANSWER.value:
            continue
        conversations.append(
            {
                "from": "observation",
                "value": step.tool_result.model_dump_json(exclude_none=True),
            }
        )
    record = {"conversations": conversations, "tools": json.dumps(TOOLS, ensure_ascii=False)}
    assert_no_hidden_leakage(record)
    return record


def request_teacher_actions(
    config: TeacherConfig,
    prompt: dict[str, Any],
    *,
    environ: dict[str, str] | None = None,
    max_attempts: int = 3,
    retry_backoff_seconds: float = 1.0,
) -> tuple[list[AgentAction], bytes, bytes]:
    if config.provider != "openai_compatible":
        raise ValueError(f"unsupported teacher provider: {config.provider}")
    if not config.base_url:
        raise ValueError("openai-compatible teacher requires base_url")
    request_payload = {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return only JSON with an actions array. Every action has tool and arguments. "
                    "Use the supplied task bundle to derive the answer and create a natural, executable "
                    "tool trajectory. The final action must be submit_answer."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    }
    request_raw = json.dumps(request_payload, sort_keys=True).encode()
    request = urllib.request.Request(
        config.base_url.rstrip("/") + "/chat/completions",
        data=request_raw,
        method="POST",
        headers={
            "Authorization": f"Bearer {config.resolve_api_key(environ)}",
            "Content-Type": "application/json",
        },
    )
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    response_raw = b""
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response_raw = response.read()
            response_value = json.loads(response_raw)
            content = response_value["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            parsed = parse_teacher_json(content)
            actions = [AgentAction.model_validate(action) for action in parsed["actions"]]
            return actions, request_raw, response_raw
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == max_attempts:
                raise
        except urllib.error.URLError:
            if attempt == max_attempts:
                raise
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
            if attempt == max_attempts:
                raise TeacherResponseError(str(exc), response_raw) from exc
        time.sleep(retry_backoff_seconds * attempt)
    raise RuntimeError("teacher request exhausted without a response")


def append_jsonl_record(path: str | Path, payload: BaseModel | dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, BaseModel):
        line = payload.model_dump_json()
    else:
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_admitted_progress(path: str | Path) -> list[AdmittedTrajectory]:
    source = Path(path)
    if not source.is_file():
        return []
    trajectories: list[AdmittedTrajectory] = []
    seen: set[str] = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = AdmittedTrajectory.model_validate_json(line)
        task_id = item.task.task_id
        if task_id in seen:
            raise ValueError(f"duplicate accepted task {task_id} at line {line_number}")
        seen.add(task_id)
        trajectories.append(item)
    return trajectories


def provenance_for(
    config: TeacherConfig,
    request_raw: bytes,
    response_raw: bytes,
    *,
    raw_response_path: str | None = None,
) -> TeacherProvenance:
    return TeacherProvenance(
        provider=config.provider,
        model=config.model,
        config_sha256=config.config_hash(),
        request_sha256=hashlib.sha256(request_raw).hexdigest(),
        response_sha256=hashlib.sha256(response_raw).hexdigest(),
        raw_response_path=raw_response_path,
    )


def write_admitted_trajectories(
    trajectories: list[AdmittedTrajectory],
    *,
    audit_path: str | Path,
    dataset_path: str | Path,
) -> None:
    audit = Path(audit_path)
    dataset = Path(dataset_path)
    audit.parent.mkdir(parents=True, exist_ok=True)
    dataset.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(
        "".join(item.model_dump_json() + "\n" for item in trajectories), encoding="utf-8"
    )
    dataset.write_text(
        "".join(
            json.dumps(to_llamafactory_record(item), ensure_ascii=False) + "\n"
            for item in trajectories
        ),
        encoding="utf-8",
    )


def write_dataset_info(*, dataset_dir: str | Path, dataset_name: str, dataset_file: str) -> Path:
    root = Path(dataset_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "dataset_info.json"
    existing = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {}
    existing[dataset_name] = {
        "file_name": dataset_file,
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "tools": "tools"},
    }
    target.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def validate_trajectory_corpus(
    trajectories: list[AdmittedTrajectory],
    *,
    expected_count: int,
    expected_provider: str,
) -> dict[str, int]:
    if len(trajectories) != expected_count:
        raise ValueError(f"expected {expected_count} trajectories, found {len(trajectories)}")
    task_ids = [item.task.task_id for item in trajectories]
    digests = [item.trajectory_sha256 for item in trajectories]
    if len(set(task_ids)) != expected_count or len(set(digests)) != expected_count:
        raise ValueError("trajectory corpus contains duplicate tasks or trajectory digests")
    providers = {item.provenance.provider for item in trajectories}
    if providers != {expected_provider}:
        raise ValueError(f"unexpected trajectory providers: {sorted(providers)}")
    expected_allocation = assign_trajectory_kinds(task_ids)
    expected_kinds = Counter(kind.value for kind in expected_allocation.values())
    actual_kinds = Counter(item.kind.value for item in trajectories)
    if actual_kinds != expected_kinds:
        raise ValueError(
            f"trajectory mix mismatch: expected {dict(expected_kinds)}, found {dict(actual_kinds)}"
        )
    return dict(sorted(actual_kinds.items()))


def write_trajectory_manifest(
    trajectories: list[AdmittedTrajectory],
    *,
    path: str | Path,
    dataset_path: str | Path,
    audit_path: str | Path,
    expected_provider: str,
    protocol_receipt: dict[str, Any] | None = None,
) -> Path:
    dataset = Path(dataset_path)
    audit = Path(audit_path)
    mix = validate_trajectory_corpus(
        trajectories,
        expected_count=len(trajectories),
        expected_provider=expected_provider,
    )
    payload = {
        "schema": "veripy.sft.trajectory_manifest",
        "schema_version": 1,
        "count": len(trajectories),
        "provider": expected_provider,
        "trajectory_kinds": mix,
        "dataset": str(dataset),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "audit": str(audit),
        "audit_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
    }
    if protocol_receipt is not None:
        assert_no_hidden_leakage(protocol_receipt)
        payload["protocol"] = protocol_receipt
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
