from __future__ import annotations

import subprocess
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from veripy_agent.environment import PythonWorker, SandboxPaths
from veripy_agent.rollout import parse_tool_call

try:
    from agent_system.environments.base import EnvironmentManagerBase
except ImportError:  # Allows CPU contract tests without importing the RL stack.

    class EnvironmentManagerBase:  # type: ignore[no-redef]
        def __init__(self, envs: Any, projection_f: Any, config: Any):
            self.envs = envs
            self.projection_f = projection_f
            self.config = config

        def close(self) -> None:
            self.envs.close()


def project_text_actions(actions: list[str]) -> tuple[list[str], list[bool]]:
    projected = []
    valid = []
    for action in actions:
        try:
            parsed = parse_tool_call(action)
        except (TypeError, ValueError):
            projected.append(action if isinstance(action, str) else "")
            valid.append(False)
        else:
            projected.append(parsed.model_dump_json())
            valid.append(True)
    return projected, valid


class VeriPyEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs: Any, config: Any):
        super().__init__(envs, project_text_actions, config)

    @staticmethod
    def _observations(values: Any) -> dict[str, Any]:
        if isinstance(values, dict) and {"text", "image", "anchor"}.issubset(values):
            return values
        return {"text": list(values), "image": None, "anchor": None}

    def reset(self, kwargs: Any = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        try:
            observations, infos = self.envs.reset(kwargs=kwargs)
        except TypeError:
            observations, infos = self.envs.reset(kwargs)
        return self._observations(observations), list(infos)

    def step(
        self, text_actions: list[str]
    ) -> tuple[dict[str, Any], Any, Any, list[dict[str, Any]]]:
        actions, valid = self.projection_f(text_actions)
        result = self.envs.step(actions)
        if len(result) == 5:
            observations, rewards, terminated, truncated, infos = result
            try:
                dones = terminated | truncated
            except TypeError:
                dones = [
                    bool(left or right) for left, right in zip(terminated, truncated, strict=True)
                ]
        elif len(result) == 4:
            observations, rewards, dones, infos = result
        else:
            raise ValueError("VeriPy vector environment must return a 4- or 5-tuple")
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        dones = np.asarray(dones, dtype=bool).reshape(-1)
        normalized_infos = []
        for is_valid, done, info in zip(valid, dones, infos, strict=True):
            item = dict(info)
            item["is_action_valid"] = bool(is_valid and item.get("is_action_valid", True))
            item.setdefault("won", bool(done and item.get("verified", False)))
            normalized_infos.append(item)
        return self._observations(observations), rewards, dones, normalized_infos

    def build_text_obs(self, raw_obs: Any = None, init: bool = False) -> list[str]:
        del init
        if raw_obs is None:
            return []
        if isinstance(raw_obs, dict):
            return list(raw_obs.get("text") or [])
        return list(raw_obs)

    def success_evaluator(
        self,
        *,
        total_infos: Any,
        total_batch_list: Any,
        episode_rewards: Any = None,
        episode_lengths: Any = None,
    ) -> dict[str, np.ndarray]:
        del episode_lengths
        metric_names = {
            "success_rate": ("won", False),
            "submission_rate": ("submission_format_valid", False),
            "partial_credit_score": ("partial_credit_score", 0.0),
            "repeated_no_progress_rate": ("repeated_no_progress", False),
            "no_submission_at_limit_rate": ("no_submission_at_limit", False),
            "error_recovery_rate": ("error_recovered", False),
        }
        metrics: dict[str, list[float]] = {name: [] for name in metric_names}
        valid_rates = []
        for infos, batch_steps in zip(total_infos, total_batch_list, strict=True):
            active_infos = [
                info
                for info, item in zip(infos, batch_steps, strict=True)
                if bool(np.asarray(item["active_masks"]).any())
            ]
            valid_rates.append(
                sum(bool(info.get("tool_call_valid", False)) for info in active_infos)
                / max(1, len(active_infos))
            )
            values = {name: float(default) for name, (_, default) in metric_names.items()}
            for info, batch_item in reversed(list(zip(infos, batch_steps, strict=True))):
                if bool(np.asarray(batch_item["active_masks"]).any()):
                    for name, (key, default) in metric_names.items():
                        values[name] = float(info.get(key, default))
                    break
            for name, value in values.items():
                metrics[name].append(value)
        result = {name: np.asarray(values, dtype=np.float32) for name, values in metrics.items()}
        result["valid_tool_call_rate"] = np.asarray(valid_rates, dtype=np.float32)
        # Pinned collector repeats tasks interleaved and groups each consecutive n trajectories.
        group_size = int(getattr(getattr(self.config, "env", None), "rollout", {}).get("n", 4))
        if episode_rewards is not None and group_size > 1:
            scores = np.asarray(episode_rewards, dtype=np.float64).reshape(-1)
            if scores.size and scores.size % group_size == 0:
                variances = scores.reshape(-1, group_size).var(axis=1)
                result["group_reward_variance_success_rate"] = np.repeat(
                    variances, group_size
                ).astype(np.float32)
                result["zero_variance_group_success_rate"] = np.repeat(
                    variances == 0, group_size
                ).astype(np.float32)
        # verl-agent only forwards environment metrics whose key contains
        # "success_rate" into trainer/W&B metrics. Keep canonical names for
        # contracts and add transport aliases at this integration boundary.
        result.update(
            {
                "submission_success_rate": result["submission_rate"],
                "partial_credit_success_rate": result["partial_credit_score"],
                "repeated_no_progress_success_rate": result["repeated_no_progress_rate"],
                "no_submission_at_limit_success_rate": result["no_submission_at_limit_rate"],
                "valid_tool_call_success_rate": result["valid_tool_call_rate"],
                "error_recovery_success_rate": result["error_recovery_rate"],
            }
        )
        return result


def make_veripy_envs(config: Any):
    from veripy_agent.environment import DockerSandboxWorker, HardenedProcessWorker

    from .vector_env import VeriPyVectorEnv

    backend = str(getattr(config.env, "sandbox_backend", "docker"))
    image = str(config.env.sandbox_image)
    worker_factory: Callable[[SandboxPaths], PythonWorker]
    if backend == "docker":
        expected_image_id = str(config.env.sandbox_image_id)
        actual_image_id = subprocess.run(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual_image_id != expected_image_id:
            raise ValueError(
                f"sandbox image id mismatch: expected {expected_image_id}, got {actual_image_id}"
            )
        worker_factory = partial(DockerSandboxWorker, image=image)
    elif backend == "hardened_process":
        user = str(getattr(config.env, "sandbox_user", "veripy-sandbox"))
        report = HardenedProcessWorker.capability_report(user=user)
        if not report.available:
            raise RuntimeError(f"hardened process sandbox unavailable: {report.reason}")
        worker_factory = partial(
            HardenedProcessWorker,
            user=user,
            timeout=float(getattr(config.env, "sandbox_timeout_seconds", 10)),
            cpu_seconds=int(getattr(config.env, "sandbox_cpu_seconds", 5)),
            memory_bytes=int(getattr(config.env, "sandbox_memory_bytes", 2 * 1024**3)),
        )
    else:
        raise ValueError(f"unknown sandbox backend: {backend}")
    dataset_root = Path(str(config.env.dataset_root))
    scratch_root = Path(str(config.env.scratch_root))
    max_turns = int(config.actor_rollout_ref.rollout.multi_turn.max_turns)
    train = VeriPyVectorEnv(
        worker_factory=worker_factory,
        dataset_root=dataset_root,
        scratch_root=scratch_root / "train",
        max_turns=max_turns,
        receipt_root=scratch_root / "train_receipts",
    )
    validation = VeriPyVectorEnv(
        worker_factory=worker_factory,
        dataset_root=dataset_root,
        scratch_root=scratch_root / "validation",
        max_turns=max_turns,
    )
    return (
        VeriPyEnvironmentManager(train, config),
        VeriPyEnvironmentManager(validation, config),
    )
