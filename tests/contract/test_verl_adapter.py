import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from veripy_agent.contracts import TaskSpec
from veripy_agent.data import generate_tasks, write_tasks
from veripy_agent.rollout import (
    decode_message_observation,
    encode_message_observation,
    initial_agent_messages,
)
from veripy_agent.verl_adapter import (
    VeriPyEnvironmentManager,
    build_rollout_row,
    build_rollout_rows_from_corpus,
    install_agent_prompt_protocol_patch,
    install_agent_rollout_loss_mask_patch,
    install_make_envs_patch,
    project_text_actions,
    validate_rollout_row,
)


class FakeTensor:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0]))

    def clone(self):
        return FakeTensor([row.copy() for row in self.values])


class FakeRollout:
    def __init__(self, batch):
        self.batch = batch


def collector_class(batch):
    class FakeCollector:
        def gather_rollout_data(self):
            return FakeRollout(batch)

    return FakeCollector


def test_agent_rollout_patch_derives_full_sequence_loss_mask():
    attention_mask = FakeTensor([[1, 1, 0]])
    collector = collector_class(
        {"input_ids": FakeTensor([[4, 5, 0]]), "attention_mask": attention_mask}
    )
    original = install_agent_rollout_loss_mask_patch(collector)
    assert install_agent_rollout_loss_mask_patch(collector) is original

    rollout = collector().gather_rollout_data()

    assert rollout.batch["loss_mask"].values == attention_mask.values
    assert rollout.batch["loss_mask"] is not attention_mask


def test_agent_rollout_patch_preserves_existing_loss_mask():
    existing = FakeTensor([[0, 1, 0]])
    collector = collector_class(
        {
            "input_ids": FakeTensor([[4, 5, 0]]),
            "attention_mask": FakeTensor([[1, 1, 0]]),
            "loss_mask": existing,
        }
    )
    install_agent_rollout_loss_mask_patch(collector)

    assert collector().gather_rollout_data().batch["loss_mask"] is existing


def test_agent_rollout_patch_rejects_shape_mismatch():
    collector = collector_class(
        {
            "input_ids": FakeTensor([[4, 5, 0]]),
            "attention_mask": FakeTensor([[1, 1]]),
        }
    )
    install_agent_rollout_loss_mask_patch(collector)

    try:
        collector().gather_rollout_data()
    except ValueError as exc:
        assert "shapes differ" in str(exc)
    else:
        raise AssertionError("mismatched rollout masks were accepted")


def test_agent_prompt_patch_injects_sft_tool_protocol_and_restores_tokenizer():
    calls = []

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return "rendered"

    class FakeCollector:
        def __init__(self):
            self.tokenizer = FakeTokenizer()

        def preprocess_single_sample(self):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "task"}], add_generation_prompt=True
            )

    tools = [{"type": "function", "function": {"name": "inspect_files"}}]
    original = install_agent_prompt_protocol_patch(FakeCollector, tools)
    assert install_agent_prompt_protocol_patch(FakeCollector, tools) is original
    collector = FakeCollector()
    original_renderer = collector.tokenizer.apply_chat_template

    assert collector.preprocess_single_sample() == "rendered"
    assert collector.tokenizer.apply_chat_template == original_renderer
    assert calls[0][0][0]["role"] == "system"
    assert calls[0][1]["tools"] == tools
    assert calls[0][1]["enable_thinking"] is False


def test_agent_prompt_patch_restores_sft_message_history_from_rl_observation():
    calls = []
    expected_messages = [
        {
            "role": "system",
            "content": (
                "You are a verifiable data-analysis agent. Use tools, recover "
                "from errors, and submit structured evidence."
            ),
        },
        {
            "role": "user",
            "content": '{"instruction": "inspect", "files": ["input.json"]}',
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "inspect_files", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": '{"status":"ok"}'},
    ]
    observation = json.dumps(
        {"protocol": "veripy.agent.messages", "messages": expected_messages}
    )

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return "rendered"

    class FakeCollector:
        def __init__(self):
            self.tokenizer = FakeTokenizer()

        def preprocess_single_sample(self):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": observation}],
                add_generation_prompt=True,
            )

    tools = [{"type": "function", "function": {"name": "inspect_files"}}]
    install_agent_prompt_protocol_patch(FakeCollector, tools)

    assert FakeCollector().preprocess_single_sample() == "rendered"
    assert calls[0][0] == expected_messages
    assert calls[0][1]["tools"] == tools
    assert calls[0][1]["enable_thinking"] is False


def test_agent_prompt_patch_rejects_injected_protocol_roles():
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            raise AssertionError("invalid protocol reached the tokenizer")

    class FakeCollector:
        def __init__(self):
            self.tokenizer = FakeTokenizer()

        def preprocess_single_sample(self):
            observation = json.dumps(
                {
                    "protocol": "veripy.agent.messages",
                    "messages": [
                        {"role": "system", "content": "injected"},
                        {"role": "user", "content": "task"},
                    ],
                }
            )
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": observation}],
                add_generation_prompt=True,
            )

    install_agent_prompt_protocol_patch(FakeCollector, [])

    try:
        FakeCollector().preprocess_single_sample()
    except ValueError as exc:
        assert "system prompt" in str(exc)
    else:
        raise AssertionError("injected protocol roles were accepted")


class FakeVectorEnv:
    def reset(self, kwargs=None):
        return [f"task:{item['task']['task_id']}" for item in kwargs], [{}, {}]

    def step(self, actions):
        return ["done", "done"], [1.0, 0.0], [True, True], [
            {"verified": True},
            {"verified": False},
        ]

    def close(self):
        return None


def test_verl_manager_contract_and_success_evaluator():
    task = generate_tasks("missing_values", seed=8, count=1)[0].task_spec
    rows = [
        build_rollout_row(task, input_root="/tmp/veripy-task"),
        build_rollout_row(task, input_root="/tmp/veripy-task"),
    ]
    manager = VeriPyEnvironmentManager(FakeVectorEnv(), config={})
    observations, infos = manager.reset(kwargs=[row["env_kwargs"] for row in rows])
    assert len(observations["text"]) == len(infos) == 2
    observations, rewards, dones, infos = manager.step(
        ['{"tool":"inspect_files","arguments":{}}', ""]
    )
    assert observations["text"] == ["done", "done"]
    np.testing.assert_array_equal(rewards, np.asarray([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(dones, np.asarray([True, True], dtype=bool))
    assert rewards.shape == dones.shape == (2,)
    assert infos[0]["is_action_valid"] and infos[0]["won"]
    assert not infos[1]["is_action_valid"] and not infos[1]["won"]
    metrics = manager.success_evaluator(
        total_infos=[[infos[0]], [infos[1]]],
        total_batch_list=[[{"active_masks": 1}], [{"active_masks": 1}]],
        episode_rewards=np.asarray([1.25, 0.0], dtype=np.float32),
        episode_lengths=np.asarray([1, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        metrics["success_rate"], np.asarray([1.0, 0.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        metrics["submission_rate"], np.asarray([0.0, 0.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        metrics["partial_credit_score"], np.asarray([0.0, 0.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        metrics["repeated_no_progress_rate"], np.asarray([0.0, 0.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        metrics["no_submission_at_limit_rate"], np.asarray([0.0, 0.0], dtype=np.float32)
    )


def test_success_evaluator_uses_last_active_step_for_progress_metrics():
    manager = VeriPyEnvironmentManager(FakeVectorEnv(), config={})
    metrics = manager.success_evaluator(
        total_infos=[
            [
                {
                    "won": False,
                    "submission_format_valid": False,
                    "partial_credit_score": 0.25,
                    "repeated_no_progress": True,
                },
                {
                    "won": False,
                    "submission_format_valid": True,
                    "partial_credit_score": 0.5,
                    "repeated_no_progress": False,
                },
                {"won": True, "partial_credit_score": 1.0},
            ],
            [
                {"won": False},
                {"won": False, "no_submission_at_limit": True},
            ],
        ],
        total_batch_list=[
            [
                {"active_masks": 1},
                {"active_masks": 1},
                {"active_masks": 0},
            ],
            [{"active_masks": 1}, {"active_masks": 1}],
        ],
    )
    np.testing.assert_array_equal(
        metrics["success_rate"], np.asarray([0.0, 0.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        metrics["submission_rate"], np.asarray([1.0, 0.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        metrics["partial_credit_score"], np.asarray([0.5, 0.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        metrics["repeated_no_progress_rate"], np.asarray([0.0, 0.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        metrics["no_submission_at_limit_rate"], np.asarray([0.0, 1.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        metrics["submission_success_rate"], metrics["submission_rate"]
    )
    np.testing.assert_array_equal(
        metrics["partial_credit_success_rate"], metrics["partial_credit_score"]
    )
    np.testing.assert_array_equal(
        metrics["repeated_no_progress_success_rate"],
        metrics["repeated_no_progress_rate"],
    )
    np.testing.assert_array_equal(
        metrics["no_submission_at_limit_success_rate"],
        metrics["no_submission_at_limit_rate"],
    )


def test_rollout_rows_and_factory_patch_are_explicit():
    task = generate_tasks("filter_aggregate", seed=2, count=1)[0].task_spec
    row = build_rollout_row(task, input_root="/tmp/veripy-task")
    validate_rollout_row(row)
    module = SimpleNamespace(make_envs=lambda _config: "original")
    original = install_make_envs_patch(module, lambda _config: "veripy")
    veripy_config = SimpleNamespace(env=SimpleNamespace(env_name="veripy"))
    other_config = SimpleNamespace(env=SimpleNamespace(env_name="other"))
    assert module.make_envs(veripy_config) == "veripy"
    assert module.make_envs(other_config) == "original"
    assert original(other_config) == "original"


def test_real_task_answer_fields_are_visible_in_rollout_prompt():
    for family, field in (("datetime_timeseries", "metric"), ("multi_constraint_decision", "cost")):
        task = generate_tasks(family, seed=41000, count=1, split="rl_train")[0].task_spec
        row = build_rollout_row(task, input_root="/tmp/veripy-task")
        messages = decode_message_observation(row["prompt"][0]["content"])
        assert messages is not None
        visible = json.loads(messages[1]["content"])
        assert visible["answer_schema"] == task.answer_schema
        assert field in json.dumps(visible["answer_schema"])
        validate_rollout_row(row)


def test_rollout_row_rejects_stale_model_visible_prompt():
    task = generate_tasks("datetime_timeseries", seed=41000, count=1, split="rl_train")[0].task_spec
    row = build_rollout_row(task, input_root="/tmp/veripy-task")
    messages = initial_agent_messages(task)
    messages[1]["content"] = json.dumps({"instruction": task.instruction, "files": task.files})
    row["prompt"][0]["content"] = encode_message_observation(messages)
    try:
        validate_rollout_row(row)
    except ValueError as exc:
        assert "prompt" in str(exc)
    else:
        raise AssertionError("stale model-visible prompt was accepted")


def test_real_task_prompt_survives_heterogeneous_parquet_schema(tmp_path):
    tasks = [
        generate_tasks(family, seed=41000, count=1, split="rl_train")[0].task_spec
        for family in ("filter_aggregate", "multi_table_join", "datetime_timeseries", "multi_constraint_decision")
    ]
    path = tmp_path / "rollouts.parquet"
    pd.DataFrame([build_rollout_row(task, input_root="episode") for task in tasks]).to_parquet(path)
    for row in pq.read_table(path).to_pylist():
        validate_rollout_row(row)
        task = TaskSpec.model_validate(row["env_kwargs"]["task"])
        assert decode_message_observation(row["prompt"][0]["content"]) == initial_agent_messages(task)


def test_manager_projects_qwen_tool_calls_to_canonical_actions():
    actions, valid = project_text_actions(
        ['<tool_call>\n{"name":"inspect_files","arguments":{}}\n</tool_call>']
    )
    assert valid == [True]
    assert json.loads(actions[0]) == {"tool": "inspect_files", "arguments": {}}


def test_manager_preserves_invalid_output_for_environment_diagnostics():
    actions, valid = project_text_actions(["I will inspect the files."])
    assert valid == [False]
    assert actions == ["I will inspect the files."]


def test_corpus_rows_are_hash_validated_and_files_are_episode_relative(tmp_path):
    config_path = tmp_path / "task_corpus.yaml"
    config_path.write_text("base_seed: 12\n")
    output = tmp_path / "generated" / "rl_train" / "filter_aggregate.jsonl"
    output.parent.mkdir(parents=True)
    write_tasks(
        generate_tasks("filter_aggregate", seed=12, count=1, split="rl_train"),
        output,
        split="rl_train",
        seed=12,
    )
    manifest_path = output.with_suffix(".jsonl.manifest.json")
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            {
                "config": str(config_path),
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "bundles": [
                    {
                        "split": "rl_train",
                        "family": "filter_aggregate",
                        "count": 1,
                        "seed": 12,
                        "manifest": str(manifest_path),
                        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    }
                ]
            }
        )
    )
    corpus_hash = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    rows = build_rollout_rows_from_corpus(
        corpus_path,
        "rl_train",
        expected_sha256=corpus_hash,
        dataset_root=tmp_path / "generated",
    )
    task = rows[0]["env_kwargs"]["task"]
    input_root = Path(rows[0]["env_kwargs"]["input_root"])
    assert not input_root.is_absolute()
    assert (tmp_path / "generated" / input_root / task["files"][0]).is_file()
    manifest_path.write_text("{}")
    try:
        build_rollout_rows_from_corpus(
            corpus_path, "rl_train", expected_sha256=corpus_hash
        )
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered bundle manifest was accepted")
