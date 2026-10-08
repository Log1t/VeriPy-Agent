from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from veripy_agent.contracts import Submission, TaskSpec, ToolStatus
from veripy_agent.environment import EpisodeEnvironment, PythonWorker, SandboxPaths
from veripy_agent.rewards import compute_reward
from veripy_agent.rollout import (
    AgentAction,
    ToolName,
    append_rejected_action,
    append_tool_exchange,
    encode_message_observation,
    initial_agent_messages,
    parse_tool_call,
)
from veripy_agent.verifiers import verify_task_submission


class VeriPyVectorEnv:
    """Synchronous batched environment consumed by verl-agent's manager seam."""

    def __init__(
        self,
        *,
        worker_factory: Callable[[SandboxPaths], PythonWorker],
        dataset_root: str | Path,
        scratch_root: str | Path,
        max_turns: int = 6,
        submission_verifier_factory: Callable[[TaskSpec], Any] | None = None,
        receipt_root: str | Path | None = None,
    ):
        self.worker_factory = worker_factory
        self.dataset_root = Path(dataset_root).resolve()
        self.scratch_root = Path(scratch_root)
        self.max_turns = max_turns
        self.submission_verifier_factory = submission_verifier_factory
        self.receipt_root = Path(receipt_root) if receipt_root is not None else None
        self.episodes: list[EpisodeEnvironment] = []
        self.submission_verifiers: list[Any] = []
        self.turns: list[int] = []
        self.tool_calls: list[int] = []
        self.had_errors: list[bool] = []
        self.recovered_errors: list[bool] = []
        self.histories: list[list[dict[str, Any]]] = []
        self.message_histories: list[list[dict[str, Any]]] = []
        self.last_progress_signatures: list[str | None] = []
        self.repeated_no_progress_counts: list[int] = []
        self.done: list[bool] = []

    def _write_train_receipt(
        self,
        index: int,
        *,
        won: bool,
        submitted: bool,
        termination: dict[str, Any] | None = None,
    ) -> None:
        receipt_root = getattr(self, "receipt_root", None)
        if receipt_root is None:
            return
        task = self.episodes[index].task
        if task.split != "rl_train":
            raise ValueError("curriculum receipts may only contain rl_train episodes")
        history = self.histories[index]
        errors = [step["failure_code"] for step in history if step["failure_code"]]
        failure = (
            "success"
            if won
            else "incorrect"
            if submitted
            else (errors[-1] if errors else "no_submission")
        )
        record = {
            "schema": "veripy.train_rollout_receipt",
            "source": "train_rollout",
            "split": task.split,
            "task_id": task.task_id,
            "family": task.family,
            "structure": task.verifier_spec.get("structure", {}),
            "failure_code": failure,
            "step_failure_codes": errors,
            "won": won,
            "submitted": submitted,
            "turns": self.turns[index],
            "steps": history,
            "episode_reward": sum(step.get("reward", 0.0) for step in history),
            "error_recovered": self.recovered_errors[index],
        }
        if termination is not None:
            record["termination"] = termination
            record["message_history_at_termination"] = self.message_histories[index]
        receipt_root.mkdir(parents=True, exist_ok=True)
        path = receipt_root / f"{task.task_id}-{uuid.uuid4().hex}.json"
        temporary = path.with_suffix(".tmp")
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def reset(self, kwargs: list[dict[str, Any]] | None = None):
        if kwargs is None or len(kwargs) == 0:
            raise ValueError("VeriPy reset requires a non-empty env_kwargs batch")
        self.close()
        self.turns = []
        self.tool_calls = []
        self.had_errors = []
        self.recovered_errors = []
        self.histories = []
        self.message_histories = []
        self.last_progress_signatures = []
        self.repeated_no_progress_counts = []
        self.done = []
        self.submission_verifiers = []
        observations = []
        infos = []
        try:
            for item in kwargs:
                task = TaskSpec.model_validate(item["task"])
                input_root_reference = Path(item["input_root"])
                input_root = (
                    input_root_reference.resolve()
                    if input_root_reference.is_absolute()
                    else (self.dataset_root / input_root_reference).resolve()
                )
                if input_root != self.dataset_root and self.dataset_root not in input_root.parents:
                    raise ValueError("episode input root escapes configured dataset root")
                paths = SandboxPaths(
                    input_root, self.scratch_root / f"{task.task_id}-{uuid.uuid4().hex}"
                )
                worker = self.worker_factory(paths)
                verifier = (
                    self.submission_verifier_factory(task)
                    if self.submission_verifier_factory is not None
                    else verify_task_submission
                )
                self.submission_verifiers.append(verifier)
                if self.submission_verifier_factory is None:
                    self.episodes.append(
                        EpisodeEnvironment(
                            task,
                            paths,
                            worker=worker,
                            max_turns=self.max_turns,
                        )
                    )
                else:
                    self.episodes.append(
                        EpisodeEnvironment(
                            task,
                            paths,
                            worker=worker,
                            max_turns=self.max_turns,
                            submission_verifier=verifier,
                        )
                    )
                self.turns.append(0)
                self.tool_calls.append(0)
                self.had_errors.append(False)
                self.recovered_errors.append(False)
                self.histories.append([])
                self.message_histories.append(initial_agent_messages(task))
                self.last_progress_signatures.append(None)
                self.repeated_no_progress_counts.append(0)
                self.done.append(False)
                observations.append(self._observation(messages=self.message_histories[-1]))
                infos.append({"task_id": task.task_id, "won": False})
        except Exception:
            self.close()
            self.turns = []
            self.tool_calls = []
            self.had_errors = []
            self.recovered_errors = []
            self.histories = []
            self.message_histories = []
            self.submission_verifiers = []
            self.last_progress_signatures = []
            self.repeated_no_progress_counts = []
            self.done = []
            raise
        return observations, infos

    @staticmethod
    def _observation(*, messages: list[dict[str, Any]]) -> str:
        return encode_message_observation(messages)

    def _messages(self, index: int, episode: EpisodeEnvironment) -> list[dict[str, Any]]:
        histories = getattr(self, "message_histories", None)
        if histories is None:
            histories = []
            self.message_histories = histories
        while len(histories) <= index:
            histories.append(initial_agent_messages(episode.task))
        return histories[index]

    def terminate_context_limit(self, index: int, *, prompt_tokens: int, limit: int) -> None:
        """Finish an unfinished real trajectory, without a fake action or new reward."""
        if self.done[index]:
            return
        if not self.histories[index]:
            raise ValueError("context limit at bootstrap: no policy action to train on")
        self.done[index] = True
        self._write_train_receipt(
            index,
            won=False,
            submitted=False,
            termination={
                "reason": "context_limit",
                "prompt_tokens": prompt_tokens,
                "limit": limit,
                "reward_policy": "unchanged_sum_of_actual_steps",
                "synthetic_action": False,
            },
        )

    def step(self, actions: list[str]):
        if len(actions) != len(self.episodes):
            raise ValueError("action batch size differs from environment batch size")
        observations, rewards, dones, infos = [], [], [], []
        for index, (episode, raw_action) in enumerate(zip(self.episodes, actions, strict=True)):
            if self.done[index]:
                observations.append("episode already terminated")
                rewards.append(0.0)
                dones.append(True)
                infos.append(
                    {
                        "won": False,
                        "verified": False,
                        "is_action_valid": False,
                        "tool_call_valid": False,
                        "submission_format_valid": False,
                        "task_success": False,
                        "evidence_consistent": False,
                        "error_recovered": False,
                        "partial_credit_score": 0.0,
                        "repeated_no_progress": False,
                        "no_submission_at_limit": False,
                        "turn_count": self.turns[index],
                        "tool_call_count": self.tool_calls[index],
                        "tool_calling": 0,
                        "reward_components": {},
                    }
                )
                continue
            self.turns[index] += 1
            result_payload: dict[str, Any]
            try:
                action = AgentAction.model_validate_json(raw_action)
                result = episode.route(action.tool, action.arguments)
                is_valid = True
                self.tool_calls[index] += 1
                result_payload = result.model_dump(mode="json", exclude_none=True)
            except Exception as exc:
                action = None
                result = None
                is_valid = False
                diagnostic = str(exc)
                try:
                    parse_tool_call(raw_action)
                except (TypeError, ValueError) as parse_exc:
                    diagnostic = str(parse_exc)
                result_payload = {
                    "status": "error",
                    "failure_code": "invalid_request",
                    "stderr": diagnostic,
                }
            security = result is not None and result.status is ToolStatus.SECURITY_VIOLATION
            submitted = (
                result is not None
                and action is not None
                and action.tool is ToolName.SUBMIT_ANSWER
                and result.status is ToolStatus.OK
            )
            won = False
            evidence = False
            format_valid = False
            correctness_score = 0.0
            if submitted and result is not None:
                try:
                    submission = Submission.model_validate_json(result.stdout)
                    verifiers = getattr(self, "submission_verifiers", [])
                    verifier = (
                        verifiers[index] if index < len(verifiers) else verify_task_submission
                    )
                    verified, evidence = verifier(episode.task, submission)
                    won = verified.passed
                    correctness_score = verified.score
                    format_valid = True
                except Exception as exc:
                    submitted = False
                    result_payload = {
                        "status": "error",
                        "failure_code": "invalid_format",
                        "stderr": str(exc),
                    }
            invalid_submission = (
                action is not None and action.tool is ToolName.SUBMIT_ANSWER and not format_valid
            )
            timeout = result is not None and result.status is ToolStatus.TIMEOUT
            tool_error = result is not None and result.status is ToolStatus.ERROR
            repeated_no_progress = False
            if action is not None and result is not None:
                signature_payload = json.dumps(
                    {
                        "action": action.model_dump(mode="json"),
                        "result": {
                            "status": result.status.value,
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                            "artifacts": result.artifacts,
                            "failure_code": (
                                result.failure_code.value if result.failure_code else None
                            ),
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                signature = hashlib.sha256(signature_payload.encode()).hexdigest()
                repeated_no_progress = signature == self.last_progress_signatures[index]
                self.last_progress_signatures[index] = signature
                if repeated_no_progress:
                    self.repeated_no_progress_counts[index] += 1
            self.histories[index].append(
                {
                    "turn": self.turns[index],
                    "tool": action.tool.value if action is not None else "invalid_action",
                    "status": result_payload.get("status", "error"),
                    "failure_code": result_payload.get("failure_code"),
                    "artifacts": result_payload.get("artifacts", []),
                    "no_progress": repeated_no_progress,
                }
            )
            messages = self._messages(index, episode)
            if action is not None:
                append_tool_exchange(
                    messages,
                    name=action.tool.value,
                    arguments=action.arguments,
                    result=result_payload,
                )
            else:
                append_rejected_action(
                    messages,
                    raw_output=raw_action,
                    result=result_payload,
                )
            observation = self._observation(messages=messages)
            step_error = not is_valid or invalid_submission or timeout or tool_error
            if self.had_errors[index] and won:
                self.recovered_errors[index] = True
            recovered = self.recovered_errors[index]
            self.had_errors[index] = bool(self.had_errors[index] or step_error)
            no_submission_at_limit = bool(
                self.turns[index] >= self.max_turns and not (submitted and format_valid)
            )
            done = bool(security or (submitted and format_valid) or no_submission_at_limit)
            reward = compute_reward(
                correct=won,
                correctness_score=correctness_score,
                evidence_consistent=evidence,
                format_valid=format_valid,
                invalid_call_count=int(not is_valid or invalid_submission),
                timeout_count=int(timeout),
                security_violation=security,
                tool_call_valid=is_valid,
                error_recovered=recovered,
                turn_count=self.turns[index],
                tool_call_count=self.tool_calls[index],
                repeated_no_progress_count=int(repeated_no_progress),
                no_submission_at_limit=no_submission_at_limit,
            )
            self.done[index] = done
            self.histories[index][-1].update(
                {
                    "action": raw_action,
                    "result": result_payload,
                    "reward": reward.total,
                    "reward_components": reward.components,
                }
            )
            if done:
                self._write_train_receipt(index, won=won, submitted=submitted)
            observations.append(observation)
            rewards.append(reward.total)
            dones.append(done)
            infos.append(
                {
                    "won": won,
                    "verified": won,
                    "is_action_valid": is_valid,
                    "tool_call_valid": reward.tool_call_valid,
                    "submission_format_valid": reward.submission_format_valid,
                    "task_success": reward.task_success,
                    "evidence_consistent": reward.evidence_consistent,
                    "error_recovered": reward.error_recovered,
                    "partial_credit_score": reward.partial_credit_score,
                    "repeated_no_progress": reward.repeated_no_progress,
                    "repeated_no_progress_count": self.repeated_no_progress_counts[index],
                    "no_submission_at_limit": reward.no_submission_at_limit,
                    "turn_count": reward.turn_count,
                    "tool_call_count": reward.tool_call_count,
                    "tool_calling": int(action is not None and result is not None),
                    "reward_components": reward.components,
                    "failure_code": (
                        result.failure_code.value
                        if result is not None and result.failure_code is not None
                        else reward.failure_code.value
                        if reward.failure_code
                        and (
                            not is_valid
                            or invalid_submission
                            or timeout
                            or security
                            or submitted
                            or repeated_no_progress
                            or no_submission_at_limit
                        )
                        else None
                    ),
                }
            )
        return observations, rewards, dones, infos

    def close(self):
        for episode in self.episodes:
            close = getattr(episode.worker, "close", None)
            if callable(close):
                close()
        self.episodes = []
