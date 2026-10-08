#!/usr/bin/env python3
"""Install the VeriPy environment seam before entering veRL's PPO runner."""

import importlib
import os
import sys
from pathlib import Path

import ray
from hydra import compose, initialize_config_dir


def _seed_scoped_training() -> None:
    """Opt-in shared driver/worker RNG initialization; historical defaults unchanged."""
    value = os.environ.get("VERIPY_COMPUTE_STUDY_SEED")
    if value is None:
        return
    import random

    import numpy as np
    import torch

    seed = int(value)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _install_global_tokenizer_compatibility() -> None:
    from veripy_agent.verl_adapter import install_auto_tokenizer_compat_patch

    install_auto_tokenizer_compat_patch()


def _install_veripy_environment() -> None:
    _seed_scoped_training()
    from agent_system import environments
    from agent_system.environments import env_manager
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    from verl import utils as verl_utils
    from verl.utils import tokenizer as tokenizer_module

    from veripy_agent.verl_adapter import (
        install_agent_prompt_protocol_patch,
        install_agent_rollout_loss_mask_patch,
        install_auto_tokenizer_compat_patch,
        install_make_envs_patch,
        install_tokenizer_compat_patch,
        make_veripy_envs,
    )

    install_auto_tokenizer_compat_patch()
    from veripy_agent.training.sft import TOOLS

    install_agent_prompt_protocol_patch(
        TrajectoryCollector,
        [{"type": "function", "function": item} for item in TOOLS],
    )
    install_agent_rollout_loss_mask_patch(TrajectoryCollector)
    if os.environ.get("VERIPY_CONTEXT_LIMIT_POLICY") == "terminate_episode_v1":
        from veripy_agent.verl_adapter.context_limit_patch import install_context_limit_patch

        install_context_limit_patch(
            TrajectoryCollector, [{"type": "function", "function": item} for item in TOOLS]
        )
    install_make_envs_patch(env_manager, make_veripy_envs)
    install_make_envs_patch(environments, make_veripy_envs)
    install_tokenizer_compat_patch(tokenizer_module)
    install_tokenizer_compat_patch(verl_utils)


def main(argv: list[str] | None = None) -> None:
    """Compose from the pinned source tree so Hydra does not require packaged YAML files."""
    _install_global_tokenizer_compatibility()
    main_ppo = importlib.import_module("verl.trainer.main_ppo")
    if main_ppo.__file__ is None:
        raise RuntimeError("verl.trainer.main_ppo has no filesystem location")
    config_dir = Path(main_ppo.__file__).resolve().parent / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="ppo_trainer", overrides=argv or sys.argv[1:])
    _install_veripy_environment()
    ray.init(
        runtime_env={
            "env_vars": {
                "NCCL_DEBUG": "WARN",
                "TOKENIZERS_PARALLELISM": "true",
                "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                "VLLM_LOGGING_LEVEL": "WARN",
                **(
                    {"VERIPY_COMPUTE_STUDY_SEED": os.environ["VERIPY_COMPUTE_STUDY_SEED"]}
                    if "VERIPY_COMPUTE_STUDY_SEED" in os.environ
                    else {}
                ),
                **(
                    {"VERIPY_CONTEXT_LIMIT_POLICY": os.environ["VERIPY_CONTEXT_LIMIT_POLICY"]}
                    if "VERIPY_CONTEXT_LIMIT_POLICY" in os.environ
                    else {}
                ),
            },
            "worker_process_setup_hook": _install_veripy_environment,
        },
        num_cpus=config.ray_init.num_cpus,
        object_store_memory=config.ray_init.get("object_store_memory"),
        include_dashboard=config.ray_init.get("include_dashboard", False),
    )
    main_ppo.run_ppo(config)


if __name__ == "__main__":
    main()
