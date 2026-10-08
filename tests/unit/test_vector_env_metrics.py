import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from veripy_agent.contracts import FailureCode, ToolResult, ToolStatus, VerifyResult
from veripy_agent.data import generate_tasks
from veripy_agent.rollout import ToolName, initial_agent_messages
from veripy_agent.verl_adapter.vector_env import VeriPyVectorEnv


def task_spec(files=None):
    task = generate_tasks("filter_aggregate", seed=41000, count=1, split="rl_train")[0].task_spec
    return task.model_copy(update={"files": ["input.csv"] if files is None else files})


class FakeEpisode:
    task = task_spec()

    def route(self, tool, arguments):
        assert tool is ToolName.SUBMIT_ANSWER
        return ToolResult(status=ToolStatus.OK, stdout=json.dumps(arguments))


def test_malformed_submit_feedback_identifies_json_error():
    env = VeriPyVectorEnv.__new__(VeriPyVectorEnv)
    env.max_turns = 6
    env.episodes = [FakeEpisode()]
    env.turns = [0]
    env.tool_calls = [0]
    env.had_errors = [False]
    env.recovered_errors = [False]
    env.histories = [[]]
    env.last_progress_signatures = [None]
    env.repeated_no_progress_counts = [0]
    env.done = [False]
    raw = '{"name":"submit_answer","arguments":{"answer":{}}}}<|im_end|>'

    observations, rewards, dones, infos = env.step([raw])

    assert not dones[0]
    assert rewards[0] == pytest.approx(-0.05)
    assert not infos[0]["tool_call_valid"]
    feedback = json.loads(observations[0])["messages"][-1]["content"]
    assert "Extra data" in feedback
    assert "Field required" not in feedback


def test_reset_accepts_numpy_object_batch_without_ambiguous_truth_value():
    env = VeriPyVectorEnv.__new__(VeriPyVectorEnv)
    env.episodes = []
    env.scratch_root = SimpleNamespace()
    with pytest.raises(KeyError, match="task"):
        env.reset(kwargs=np.asarray([{}], dtype=object))


def test_relative_episode_root_is_resolved_under_dataset_root(tmp_path, monkeypatch):
    dataset_root = tmp_path / "generated"
    episode_root = dataset_root / "rl_train" / "episode"
    episode_root.mkdir(parents=True)
    task = task_spec()

    monkeypatch.setattr(
        "veripy_agent.verl_adapter.vector_env.TaskSpec.model_validate",
        lambda _value: task,
    )
    created_paths = []

    def make_worker(paths):
        created_paths.append(paths)
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(
        "veripy_agent.verl_adapter.vector_env.EpisodeEnvironment",
        lambda task, paths, worker, max_turns: SimpleNamespace(
            task=task, paths=paths, worker=worker
        ),
    )

    env = VeriPyVectorEnv(
        worker_factory=make_worker,
        dataset_root=dataset_root,
        scratch_root=tmp_path / "scratch",
    )
    observations, _ = env.reset(
        kwargs=[{"task": {}, "input_root": str(Path("rl_train") / "episode")}]
    )
    assert observations
    initial = json.loads(observations[0])
    assert initial == {"protocol": "veripy.agent.messages", "messages": initial_agent_messages(task)}
    assert env.episodes[0].paths.inputs == episode_root
    assert created_paths[0].inputs == episode_root


def test_relative_episode_root_cannot_escape_dataset_root(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "veripy_agent.verl_adapter.vector_env.TaskSpec.model_validate",
        lambda _value: SimpleNamespace(task_id="task-1"),
    )
    env = VeriPyVectorEnv(
        worker_factory=lambda _paths: SimpleNamespace(close=lambda: None),
        dataset_root=tmp_path / "generated",
        scratch_root=tmp_path / "scratch",
    )
    with pytest.raises(ValueError, match="escapes configured dataset root"):
        env.reset(kwargs=[{"task": {}, "input_root": "../outside"}])


def test_reset_failure_closes_partially_created_workers(tmp_path, monkeypatch):
    dataset_root = tmp_path / "generated"
    episode_root = dataset_root / "rl_train" / "episode"
    episode_root.mkdir(parents=True)
    task = task_spec(files=[])
    closed = []
    validations = iter([task, ValueError("bad task")])

    def validate(_value):
        value = next(validations)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(
        "veripy_agent.verl_adapter.vector_env.TaskSpec.model_validate", validate
    )
    monkeypatch.setattr(
        "veripy_agent.verl_adapter.vector_env.EpisodeEnvironment",
        lambda task, paths, worker, max_turns: SimpleNamespace(
            task=task, paths=paths, worker=worker
        ),
    )
    env = VeriPyVectorEnv(
        worker_factory=lambda _paths: SimpleNamespace(close=lambda: closed.append(True)),
        dataset_root=dataset_root,
        scratch_root=tmp_path / "scratch",
    )
    kwargs = [
        {"task": {}, "input_root": str(Path("rl_train") / "episode")},
        {"task": {}, "input_root": str(Path("rl_train") / "episode")},
    ]

    with pytest.raises(ValueError, match="bad task"):
        env.reset(kwargs=kwargs)

    assert closed == [True]
    assert env.episodes == []
    assert env.turns == []
    assert env.histories == []


def test_invalid_submission_schema_is_observed_and_does_not_escape(monkeypatch):
    env = VeriPyVectorEnv.__new__(VeriPyVectorEnv)
    env.max_turns = 6
    env.episodes = [FakeEpisode()]
    env.turns = [0]
    env.tool_calls = [0]
    env.had_errors = [False]
    env.recovered_errors = [False]
    env.histories = [[]]
    env.last_progress_signatures = [None]
    env.repeated_no_progress_counts = [0]
    env.done = [False]

    observations, rewards, dones, infos = env.step(
        [json.dumps({"tool": "submit_answer", "arguments": {"answer": 1}})]
    )

    assert not dones[0]
    assert rewards[0] == -0.05
    messages = json.loads(observations[0])["messages"]
    assert messages[-1]["role"] == "tool"
    assert json.loads(messages[-1]["content"])["failure_code"] == "invalid_format"
    assert infos[0]["tool_call_valid"]
    assert not infos[0]["submission_format_valid"]
    assert infos[0]["turn_count"] == 1
    assert infos[0]["tool_call_count"] == 1
    assert infos[0]["tool_calling"] == 1
    assert infos[0]["reward_components"]["invalid_call"] == -0.05


def test_valid_tool_after_error_does_not_mark_recovery_before_success(monkeypatch):
    class RecoveringEpisode:
        task = task_spec()

        def route(self, tool, arguments):
            return ToolResult(status=ToolStatus.OK, stdout="[]")

    env = VeriPyVectorEnv.__new__(VeriPyVectorEnv)
    env.max_turns = 6
    env.episodes = [RecoveringEpisode()]
    env.turns = [1]
    env.tool_calls = [0]
    env.had_errors = [True]
    env.recovered_errors = [False]
    env.histories = [[]]
    env.last_progress_signatures = [None]
    env.repeated_no_progress_counts = [0]
    env.done = [False]
    _, _, dones, infos = env.step(
        [json.dumps({"tool": "inspect_files", "arguments": {}})]
    )
    assert not dones[0]
    assert not infos[0]["error_recovered"]
    assert infos[0]["failure_code"] is None
    assert infos[0]["tool_calling"] == 1


def test_python_error_then_verified_submission_marks_recovery(monkeypatch):
    class RecoveringEpisode:
        task = task_spec()

        def route(self, tool, arguments):
            if tool is ToolName.EXECUTE_PYTHON:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    stderr="boom",
                    failure_code=FailureCode.EXECUTION_ERROR,
                )
            return ToolResult(status=ToolStatus.OK, stdout=json.dumps(arguments))

    monkeypatch.setattr(
        "veripy_agent.verl_adapter.vector_env.verify_task_submission",
        lambda *_args: (VerifyResult(passed=True, score=1.0), True),
    )
    env = VeriPyVectorEnv.__new__(VeriPyVectorEnv)
    env.max_turns = 6
    env.episodes = [RecoveringEpisode()]
    env.turns = [0]
    env.tool_calls = [0]
    env.had_errors = [False]
    env.recovered_errors = [False]
    env.histories = [[]]
    env.last_progress_signatures = [None]
    env.repeated_no_progress_counts = [0]
    env.done = [False]

    _, _, dones, infos = env.step(
        [json.dumps({"tool": "execute_python", "arguments": {"code": "1 / 0"}})]
    )
    assert not dones[0]
    assert infos[0]["failure_code"] == "execution_error"
    assert env.had_errors == [True]

    _, _, dones, infos = env.step(
        [
            json.dumps(
                {
                    "tool": "submit_answer",
                    "arguments": {
                        "answer": 1,
                        "evidence": {
                            "columns": ["x"],
                            "filters": [],
                            "intermediate_values": {"answer": 1},
                        },
                    },
                }
            )
        ]
    )
    assert dones[0]
    assert infos[0]["task_success"]
    assert infos[0]["error_recovered"]


def test_timeout_is_observed_and_counted():
    class TimeoutEpisode:
        task = task_spec()

        def route(self, tool, arguments):
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                stderr="execution timed out",
                failure_code=FailureCode.TIMEOUT,
            )

    env = VeriPyVectorEnv.__new__(VeriPyVectorEnv)
    env.max_turns = 6
    env.episodes = [TimeoutEpisode()]
    env.turns = [0]
    env.tool_calls = [0]
    env.had_errors = [False]
    env.recovered_errors = [False]
    env.histories = [[]]
    env.last_progress_signatures = [None]
    env.repeated_no_progress_counts = [0]
    env.done = [False]

    observations, rewards, dones, infos = env.step(
        [json.dumps({"tool": "execute_python", "arguments": {"code": "while True: pass"}})]
    )
    assert not dones[0]
    assert rewards[0] == pytest.approx(-0.1)
    assert infos[0]["failure_code"] == "timeout"
    assert infos[0]["reward_components"]["timeout"] == -0.1
    assert json.loads(observations[0])["messages"][-1]["role"] == "tool"


def test_invalid_action_recovery_uses_sft_compatible_message_history():
    class RecoveringEpisode:
        task = task_spec()

        def route(self, tool, arguments):
            return ToolResult(status=ToolStatus.OK, stdout='["input.csv"]')

    env = VeriPyVectorEnv.__new__(VeriPyVectorEnv)
    env.max_turns = 6
    env.episodes = [RecoveringEpisode()]
    env.turns = [0]
    env.tool_calls = [0]
    env.had_errors = [False]
    env.recovered_errors = [False]
    env.histories = [[]]
    env.last_progress_signatures = [None]
    env.repeated_no_progress_counts = [0]
    env.done = [False]

    observations, _, dones, _ = env.step(["not-json"])

    assert not dones[0]
    messages = json.loads(observations[0])["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[-2]["content"] == "not-json"
    assert "Return one complete tool call with valid JSON." in messages[-1]["content"]
    assert "recent_actions" not in observations[0]

    observations, _, _, _ = env.step(
        [json.dumps({"tool": "inspect_files", "arguments": {}})]
    )
    messages = json.loads(observations[0])["messages"]
    assert [message["role"] for message in messages[-2:]] == ["assistant", "tool"]


def test_repeated_identical_action_and_result_is_penalized_and_visible():
    class RepeatingEpisode:
        task = task_spec()

        def route(self, tool, arguments):
            return ToolResult(status=ToolStatus.OK, stdout='["input.csv"]')

    env = VeriPyVectorEnv.__new__(VeriPyVectorEnv)
    env.max_turns = 3
    env.episodes = [RepeatingEpisode()]
    env.turns = [0]
    env.tool_calls = [0]
    env.had_errors = [False]
    env.recovered_errors = [False]
    env.histories = [[]]
    env.last_progress_signatures = [None]
    env.repeated_no_progress_counts = [0]
    env.done = [False]
    action = json.dumps({"tool": "inspect_files", "arguments": {}})

    env.step([action])
    observations, rewards, dones, infos = env.step([action])

    assert not dones[0]
    assert rewards[0] == pytest.approx(-0.05)
    assert infos[0]["repeated_no_progress"]
    assert infos[0]["repeated_no_progress_count"] == 1
    payload = json.loads(observations[0])
    assert [message["role"] for message in payload["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert "recent_actions" not in payload
    assert "no_progress" not in observations[0]


def test_model_history_uses_sft_messages_and_reset_clears_episode_state(
    tmp_path, monkeypatch
):
    dataset_root = tmp_path / "generated"
    episode_root = dataset_root / "rl_train" / "episode"
    episode_root.mkdir(parents=True)
    task = task_spec()
    closed = []

    monkeypatch.setattr(
        "veripy_agent.verl_adapter.vector_env.TaskSpec.model_validate",
        lambda _value: task,
    )

    class StatefulEpisode:
        def __init__(self, task, paths, worker, max_turns):
            self.task = task
            self.paths = paths
            self.worker = worker

        def route(self, tool, arguments):
            return ToolResult(status=ToolStatus.OK, stdout=json.dumps(arguments))

    monkeypatch.setattr(
        "veripy_agent.verl_adapter.vector_env.EpisodeEnvironment", StatefulEpisode
    )
    env = VeriPyVectorEnv(
        worker_factory=lambda _paths: SimpleNamespace(close=lambda: closed.append(True)),
        dataset_root=dataset_root,
        scratch_root=tmp_path / "scratch",
        max_turns=6,
    )
    kwargs = [{"task": {}, "input_root": str(Path("rl_train") / "episode")}]
    env.reset(kwargs=kwargs)
    observation = ""
    for index in range(5):
        observation = env.step(
            [json.dumps({"tool": "inspect_table", "arguments": {"rows": index + 1}})]
        )[0][0]
    payload = json.loads(observation)
    assert [message["role"] for message in payload["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert len(env.histories[0]) == 5
    assert len(env.message_histories[0]) == 12

    observations, _ = env.reset(kwargs=kwargs)
    assert closed == [True]
    assert len(json.loads(observations[0])["messages"]) == 2
    assert len(env.message_histories[0]) == 2
    assert env.turns == [0]
    assert env.tool_calls == [0]
    assert env.last_progress_signatures == [None]
    assert env.repeated_no_progress_counts == [0]


def test_security_violation_overrides_progress_shaping(monkeypatch):
    class SecurityEpisode:
        task = task_spec()

        def route(self, tool, arguments):
            return ToolResult(
                status=ToolStatus.SECURITY_VIOLATION,
                failure_code="security_violation",
            )

    env = VeriPyVectorEnv.__new__(VeriPyVectorEnv)
    env.max_turns = 1
    env.episodes = [SecurityEpisode()]
    env.turns = [0]
    env.tool_calls = [0]
    env.had_errors = [False]
    env.recovered_errors = [False]
    env.histories = [[]]
    env.last_progress_signatures = [None]
    env.repeated_no_progress_counts = [0]
    env.done = [False]

    _, rewards, dones, infos = env.step(
        [json.dumps({"tool": "inspect_files", "arguments": {}})]
    )

    assert dones == [True]
    assert rewards == [-1.0]
    assert infos[0]["failure_code"] == "security_violation"
    assert infos[0]["reward_components"] == {"security_violation": -1.0}


def test_turn_limit_without_submission_has_distinct_penalty():
    class InspectEpisode:
        task = task_spec()

        def route(self, tool, arguments):
            return ToolResult(status=ToolStatus.OK, stdout='["input.csv"]')

    env = VeriPyVectorEnv.__new__(VeriPyVectorEnv)
    env.max_turns = 1
    env.episodes = [InspectEpisode()]
    env.turns = [0]
    env.tool_calls = [0]
    env.had_errors = [False]
    env.recovered_errors = [False]
    env.histories = [[]]
    env.last_progress_signatures = [None]
    env.repeated_no_progress_counts = [0]
    env.done = [False]

    _, rewards, dones, infos = env.step(
        [json.dumps({"tool": "inspect_files", "arguments": {}})]
    )

    assert dones[0]
    assert rewards[0] == pytest.approx(-0.1)
    assert infos[0]["no_submission_at_limit"]
    assert infos[0]["failure_code"] == "no_submission"


def test_partial_verifier_score_reaches_reward_without_leaking_diagnostics(monkeypatch):
    env = VeriPyVectorEnv.__new__(VeriPyVectorEnv)
    env.max_turns = 6
    env.episodes = [FakeEpisode()]
    env.turns = [0]
    env.tool_calls = [0]
    env.had_errors = [False]
    env.recovered_errors = [False]
    env.histories = [[]]
    env.last_progress_signatures = [None]
    env.repeated_no_progress_counts = [0]
    env.done = [False]
    monkeypatch.setattr(
        "veripy_agent.verl_adapter.vector_env.verify_task_submission",
        lambda _task, _submission: (
            VerifyResult(passed=False, score=0.5, diagnostics={"expected": "hidden"}),
            False,
        ),
    )
    action = {
        "tool": "submit_answer",
        "arguments": {
            "answer": {"a": 1},
            "evidence": {"columns": [], "filters": [], "intermediate_values": {}},
        },
    }

    observations, rewards, dones, infos = env.step([json.dumps(action)])

    assert dones[0]
    assert rewards[0] == pytest.approx(0.3)
    assert infos[0]["partial_credit_score"] == 0.5
    assert "hidden" not in observations[0]
