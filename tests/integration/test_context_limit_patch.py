import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from veripy_agent.training.sft import TOOLS
from veripy_agent.verl_adapter.context_limit_patch import (
    ContextBudgetEnvironment,
    install_context_limit_patch,
    observation_tokens,
)
from veripy_agent.verl_adapter.prompt_patch import install_agent_prompt_protocol_patch
from veripy_agent.verl_adapter.vector_env import VeriPyVectorEnv


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    from veripy_agent.verl_adapter import install_auto_tokenizer_compat_patch

    install_auto_tokenizer_compat_patch()
    return AutoTokenizer.from_pretrained(
        "/root/autodl-tmp/veripy-agent/models/Qwen3-1.7B-ai4ai_behavior-start",
        local_files_only=True,
    )


def tools():
    return [{"type": "function", "function": x} for x in TOOLS]


class Manager:
    def __init__(self, long_text, initial="short"):
        self.initial = initial
        self.long_text = long_text
        self.step_count = 0
        self.ended = []
        self.closed = False
        self.envs = SimpleNamespace(terminate_context_limit=self.end, close=self.close)

    def close(self):
        self.closed = True

    def end(self, index, **kw):
        self.ended.append((index, kw))

    def reset(self, **kw):
        return {"text": [self.initial, self.initial], "image": None, "anchor": None}, [{}, {}]

    def step(self, actions):
        self.step_count += 1
        return (
            {"text": [self.long_text, "short"], "image": None, "anchor": None},
            np.array([0.3, 0.4]) if self.step_count == 1 else np.array([0.0, 0.5]),
            np.array([False, False]) if self.step_count == 1 else np.array([True, True]),
            [
                {"won": False, "is_action_valid": True},
                {"won": self.step_count == 2, "is_action_valid": True},
            ],
        )

    def success_evaluator(self, **kw):
        return {"success_rate": np.array([0.0, 1.0])}


def test_exact_tokenizer_budget_and_no_observation_truncation(tokenizer):
    long = "overflow " * 3000
    manager = Manager(long)
    guard = ContextBudgetEnvironment(manager, tokenizer, tools(), 2560)
    guard.reset()
    obs, rewards, dones, infos = guard.step(["real1", "real2"])
    assert dones.tolist() == [True, False]
    assert rewards.tolist() == [0.3, 0.4]
    assert manager.long_text == long and obs["text"][0] == "episode already terminated"
    assert infos[0]["context_limit"] and infos[0]["context_prompt_tokens"] > 2560
    assert manager.ended[0][0] == 0


@pytest.mark.parametrize("guard_enabled", [False, True])
def test_real_pinned_collector_mixed_batch_keeps_prior_action_and_other_episode(
    tokenizer, guard_enabled
):
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    from verl import DataProto

    class Collector(TrajectoryCollector):
        pass

    install_agent_prompt_protocol_patch(Collector, tools())
    if guard_enabled:
        install_context_limit_patch(Collector, tools())
    cfg = OmegaConf.create(
        {
            "data": {
                "max_prompt_length": 2560,
                "max_response_length": 512,
                "truncation": "error",
                "return_raw_chat": True,
            },
            "env": {"rollout": {"n": 2}, "max_steps": 6},
            "actor_rollout_ref": {"rollout": {"max_model_len": 4096}},
        }
    )
    collector = Collector(cfg, tokenizer)
    raw = np.empty(2, dtype=object)
    raw[:] = [[{"role": "user", "content": "short"}]] * 2
    batch = DataProto.from_single_dict(
        {
            "input_ids": torch.zeros((2, 2), dtype=torch.long),
            "raw_prompt": raw,
            "data_source": np.array(["veripy", "veripy"], dtype=object),
        }
    )

    class Actor:
        world_size = 1
        calls = 0

        def generate_sequences(self, batch):
            self.calls += 1
            responses = torch.ones((2, 1), dtype=torch.long)
            ids = torch.cat([batch.batch["input_ids"], responses], dim=1)
            mask = torch.cat([batch.batch["attention_mask"], torch.ones_like(responses)], dim=1)
            return DataProto.from_single_dict(
                {"responses": responses, "input_ids": ids, "attention_mask": mask}
            )

    actor = Actor()
    manager = Manager("overflow " * 3000)
    if not guard_enabled:
        with pytest.raises(NotImplementedError, match="larger than max_length"):
            collector.vanilla_multi_turn_loop(batch, actor, manager)
        assert actor.calls == 1 and manager.step_count == 1
        return
    result = collector.vanilla_multi_turn_loop(batch, actor, manager)
    batches, rewards, lengths, _, _, _ = result
    assert actor.calls == 2 and manager.step_count == 2
    assert rewards.tolist() == pytest.approx([0.3, 0.9])
    assert lengths.tolist() == [1.0, 2.0]
    assert [bool(x["active_masks"]) for x in batches[0]] == [True, False]
    assert [bool(x["active_masks"]) for x in batches[1]] == [True, True]
    assert len(manager.ended) == 1
    from veripy_agent.verl_adapter.loss_mask_patch import install_agent_rollout_loss_mask_patch

    install_agent_rollout_loss_mask_patch(Collector)
    gathered = collector.gather_rollout_data(
        total_batch_list=batches,
        episode_rewards=rewards,
        episode_lengths=lengths,
        success=result[3],
        traj_uid=result[4],
        tool_callings=result[5],
    )
    assert len(gathered) == 3  # one true action + two sibling actions, no placeholder row
    assert torch.equal(gathered.batch["loss_mask"], gathered.batch["attention_mask"])


def test_bootstrap_overflow_not_fabricated(tokenizer):
    manager = Manager("short", initial="overflow " * 3000)
    guard = ContextBudgetEnvironment(manager, tokenizer, tools(), 2560)
    with pytest.raises(ValueError, match="bootstrap"):
        guard.reset()
    assert manager.step_count == 0 and not manager.ended
    assert manager.closed


def test_real_environment_context_receipt_preserves_rewards_and_no_fake_action(tmp_path):
    from veripy_agent.data import generate_tasks

    task = generate_tasks("missing_values", seed=42, count=1, split="rl_train")[0].task_spec
    env = VeriPyVectorEnv(
        worker_factory=lambda _: None,
        dataset_root=tmp_path,
        scratch_root=tmp_path,
        receipt_root=tmp_path / "receipts",
    )
    env.episodes = [SimpleNamespace(task=task)]
    env.done = [False]
    env.turns = [1]
    env.recovered_errors = [False]
    env.histories = [
        [
            {
                "failure_code": None,
                "reward": 0.2,
                "action": "actual action",
                "result": {"stdout": "actual output"},
            }
        ]
    ]
    env.message_histories = [[{"role": "user", "content": "original history"}]]
    before = copy.deepcopy(env.histories)
    env.terminate_context_limit(0, prompt_tokens=2619, limit=2560)
    env.terminate_context_limit(0, prompt_tokens=2619, limit=2560)
    records = list((tmp_path / "receipts").glob("*.json"))
    assert len(records) == 1
    d = json.loads(records[0].read_text())
    assert env.histories == before and d["steps"] == before[0]
    assert d["episode_reward"] == 0.2 and not d["submitted"] and not d["won"]
    assert d["failure_code"] == "no_submission" and d["termination"]["reason"] == "context_limit"
    assert not d["termination"]["synthetic_action"] and env.done[0]


def test_no_limits_and_boundary_do_not_terminate(tokenizer):
    manager = Manager("short")
    size = observation_tokens(tokenizer, "short", tools())
    guard = ContextBudgetEnvironment(manager, tokenizer, tools(), size)
    guard.reset()
    _, rewards, dones, infos = guard.step(["actual", "actual"])
    assert not dones.any() and not manager.ended
    assert "context_limit" not in infos[0]
