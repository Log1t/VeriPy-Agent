#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from veripy_agent.rollout import protocol_manifest_from_resources
from veripy_agent.training.gates import ResourceLock, ResumeGate
from veripy_agent.training.sft import TOOLS
from veripy_agent.verl_adapter.data import validate_rollout_row

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/rl/qwen3_1_7b_grpo.yaml"
VERL_CONFIG_DIR = ROOT / "external/verl-agent/verl/trainer/config"


@dataclass(frozen=True)
class LaunchPlan:
    mode: str
    steps: int
    output_dir: Path
    command: list[str]
    environment: dict[str, str]
    protocol: dict[str, Any]


def _absolute(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "veripy.grpo.launch":
        raise ValueError("invalid VeriPy GRPO launch config")
    return value


def _hydra_list(values: list[str]) -> str:
    return "[" + ",".join(values) + "]"


def _validate_model_rope(model_path: Path) -> None:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    theta = config.get("rope_theta")
    if not isinstance(theta, int | float) or isinstance(theta, bool) or theta <= 0:
        raise ValueError("model config requires a positive rope_theta for the pinned runtime")
    parameters = config.get("rope_parameters")
    if parameters is not None and (
        not isinstance(parameters, dict) or parameters.get("rope_theta") != theta
    ):
        raise ValueError("model rope_theta conflicts with rope_parameters")


def _validate_episode_paths(plan: LaunchPlan) -> None:
    import pyarrow.parquet as pq

    overrides = dict(item.split("=", 1) for item in plan.command[2:] if "=" in item)
    dataset_root = Path(overrides["+env.dataset_root"]).resolve()
    for split in ("train", "val"):
        parquet = Path(overrides[f"data.{split}_files"])
        for row in pq.read_table(parquet).to_pylist():
            validate_rollout_row(row)
            episode = row["env_kwargs"]
            reference = Path(episode["input_root"])
            input_root = (
                reference.resolve()
                if reference.is_absolute()
                else (dataset_root / reference).resolve()
            )
            if not input_root.is_relative_to(dataset_root) or not input_root.is_dir():
                raise ValueError(f"stale or escaping episode input root: {input_root}")
            for name in episode["task"]["files"]:
                candidate = (input_root / name).resolve()
                if not candidate.is_relative_to(input_root) or not candidate.is_file():
                    raise ValueError(f"missing or escaping episode input file: {candidate}")


def _validate_probe_manifest(plan: LaunchPlan) -> None:
    if plan.mode != "reward_probe":
        return
    import pyarrow.parquet as pq

    path = ROOT / "data/manifests/reward_probe.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "veripy.reward_probe" or manifest.get("families") != [
        "filter_aggregate",
        "multi_table_join",
        "datetime_timeseries",
        "multi_constraint_decision",
    ]:
        raise ValueError("invalid reward probe manifest")
    resources = ROOT / "resources.lock.json"
    corpus = ROOT / "data/manifests/task_corpus.json"
    for source, expected in (
        (resources, manifest["resources_sha256"]),
        (corpus, manifest["corpus_sha256"]),
    ):
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"reward probe source changed: {source}")
    overrides = dict(item.split("=", 1) for item in plan.command[2:] if "=" in item)
    for label, key in (("train", "data.train_files"), ("validation", "data.val_files")):
        info = manifest["splits"][label]
        target = (ROOT / info["file"]).resolve()
        source = (ROOT / info["source"]).resolve()
        if target != Path(overrides[key]).resolve():
            raise ValueError(f"reward probe {label} path mismatch")
        for current, digest in ((target, info["sha256"]), (source, info["source_sha256"])):
            if hashlib.sha256(current.read_bytes()).hexdigest() != digest:
                raise ValueError(f"reward probe {label} file changed: {current}")
        table = pq.read_table(target, columns=["ability", "extra_info"])
        rows = table.to_pylist()
        if len(rows) != 4 or [row["ability"] for row in rows] != manifest["families"]:
            raise ValueError(f"reward probe {label} family coverage changed")
        if [row["extra_info"]["task_id"] for row in rows] != info["task_ids"]:
            raise ValueError(f"reward probe {label} task identity changed")


def _validate_feedback_manifest(plan: LaunchPlan) -> None:
    if plan.mode != "feedback_probe":
        return
    import pyarrow.parquet as pq

    path = ROOT / "data/manifests/feedback_probe.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "veripy.feedback_probe":
        raise ValueError("invalid feedback probe manifest")
    corpus = ROOT / "data/manifests/task_corpus.json"
    if hashlib.sha256(corpus.read_bytes()).hexdigest() != manifest["corpus_sha256"]:
        raise ValueError("feedback probe corpus hash mismatch")
    selection = ROOT / "data/manifests/paired_validation_64.json"
    if (
        hashlib.sha256(selection.read_bytes()).hexdigest()
        != manifest["validation_selection_sha256"]
    ):
        raise ValueError("feedback probe validation selection changed")
    overrides = dict(item.split("=", 1) for item in plan.command[2:] if "=" in item)
    required = {
        "data.train_batch_size": "8",
        "data.shuffle": "false",
        "env.rollout.n": "4",
        "trainer.total_training_steps": "4",
    }
    if any(overrides.get(key) != value for key, value in required.items()):
        raise ValueError("feedback probe must consume 32 tasks in four group-size-four steps")
    ids_by_split = {}
    for label, key in (("train", "data.train_files"), ("validation", "data.val_files")):
        info = manifest["splits"][label]
        target = (ROOT / info["file"]).resolve()
        source = (ROOT / info["source"]).resolve()
        if target != Path(overrides[key]).resolve():
            raise ValueError(f"feedback probe {label} path mismatch")
        for current, digest in ((target, info["sha256"]), (source, info["source_sha256"])):
            if hashlib.sha256(current.read_bytes()).hexdigest() != digest:
                raise ValueError(f"feedback probe {label} file hash mismatch: {current}")
        table = pq.read_table(target, columns=["ability", "extra_info"])
        rows = table.to_pylist()
        ids = [row["extra_info"]["task_id"] for row in rows]
        expected_count = 32 if label == "train" else 8
        if len(ids) != expected_count or ids != info["task_ids"]:
            raise ValueError(f"feedback probe {label} task identity changed")
        if len({row["ability"] for row in rows}) != 8:
            raise ValueError(f"feedback probe {label} must cover eight families")
        ids_by_split[label] = set(ids)
    if ids_by_split["train"] & ids_by_split["validation"]:
        raise ValueError("feedback probe training and validation tasks overlap")
    if len(ids_by_split["train"]) != 32 or len(ids_by_split["validation"]) != 8:
        raise ValueError("feedback probe contains duplicate tasks")


def _validate_staged_curriculum_manifest(plan: LaunchPlan) -> None:
    if plan.mode != "staged_curriculum":
        return
    import pyarrow.parquet as pq

    overrides = dict(item.split("=", 1) for item in plan.command[2:] if "=" in item)
    required = {
        "data.train_batch_size": "8",
        "data.shuffle": "false",
        "env.rollout.n": "4",
        "trainer.total_training_steps": "24",
        "trainer.total_epochs": "1",
    }
    if any(overrides.get(key) != value for key, value in required.items()):
        raise ValueError("staged curriculum schedule differs from its frozen 24-step manifest")
    train = Path(overrides["data.train_files"])
    validation = Path(overrides["data.val_files"])
    path = train.parent / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    frozen_manifest = _absolute(
        manifest.get("manifest_output", "data/manifests/staged_curriculum.json")
    )
    if (
        hashlib.sha256(path.read_bytes()).hexdigest()
        != hashlib.sha256(frozen_manifest.read_bytes()).hexdigest()
    ):
        raise ValueError("staged curriculum differs from its tracked manifest")
    if (
        manifest.get("schema") != "veripy.staged_curriculum"
        or manifest.get("task_mutation_used") is not False
    ):
        raise ValueError("invalid staged curriculum manifest")
    if manifest.get("benchmark_feedback_used") is not False:
        raise ValueError("benchmark feedback entered the staged curriculum")
    sources = (
        (ROOT / "data/manifests/task_corpus.json", manifest["source_corpus_sha256"]),
        (ROOT / "data/generated/verl/rl_train.parquet", manifest["source_train_sha256"]),
        (ROOT / "data/generated/verl/validation.parquet", manifest["source_validation_sha256"]),
        (train, manifest["train_parquet_sha256"]),
        (validation, manifest["evaluation_parquet_sha256"]),
    )
    if any(
        hashlib.sha256(source.read_bytes()).hexdigest() != expected for source, expected in sources
    ):
        raise ValueError("staged curriculum source or output hash mismatch")
    train_rows = pq.read_table(train, columns=["ability", "extra_info"]).to_pylist()
    validation_rows = pq.read_table(validation, columns=["extra_info"]).to_pylist()
    ids = [row["extra_info"]["task_id"] for row in train_rows]
    val_ids = [row["extra_info"]["task_id"] for row in validation_rows]
    if (
        len(ids) != 192
        or len(set(ids)) != len(ids)
        or len(val_ids) != 32
        or len(set(val_ids)) != 32
    ):
        raise ValueError("staged curriculum row count or uniqueness mismatch")
    if set(ids) & set(val_ids) or val_ids != manifest["evaluation_task_ids"]:
        raise ValueError("staged curriculum train/evaluation identities mismatch")
    selection = _absolute(
        manifest.get("evaluation_selection_path", "data/manifests/staged_validation_32.json")
    )
    if (
        hashlib.sha256(selection.read_bytes()).hexdigest()
        != manifest["evaluation_selection_sha256"]
    ):
        raise ValueError("staged curriculum evaluation selection changed")
    if json.loads(selection.read_text(encoding="utf-8"))["task_ids"] != val_ids:
        raise ValueError("staged curriculum validation parquet differs from selection")
    stages = manifest.get("stages", [])
    if [stage.get("name") for stage in stages] != ["accessible", "balanced", "challenge"]:
        raise ValueError("staged curriculum phase order mismatch")
    for index, stage in enumerate(stages):
        start, end = index * 64, (index + 1) * 64
        if (
            stage.get("start_row") != start
            or stage.get("end_row") != end
            or stage.get("task_ids") != ids[start:end]
        ):
            raise ValueError("staged curriculum phase boundaries mismatch")
        for offset in range(start, end, 8):
            if len({row["ability"] for row in train_rows[offset : offset + 8]}) != 8:
                raise ValueError("staged curriculum batch lost eight-family coverage")
    if (
        manifest.get("task_ids_sha256")
        != hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    ):
        raise ValueError("staged curriculum task order hash mismatch")


def build_plan(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    mode: str = "smoke",
    model: str | Path | None = None,
    output_dir: str | Path | None = None,
    resume: bool = False,
    checkpoint: str | Path | None = None,
    train_file: str | Path | None = None,
    experiment_name: str | None = None,
    cpu_preflight: bool = False,
) -> LaunchPlan:
    config_file = _absolute(config_path)
    config = _load(config_file)
    modes = config["modes"]
    if mode not in modes:
        raise ValueError(f"unknown GRPO mode: {mode}")
    profile = modes[mode]
    if experiment_name is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_name):
        raise ValueError("experiment name must contain only letters, digits, underscore or hyphen")
    model_path = _absolute(model or os.environ.get("VERIPY_MODEL_PATH", config["model_path"]))
    train_override = train_file is not None
    if mode == "curriculum" and not train_override:
        raise ValueError("curriculum mode requires a manifest-locked --train-file")
    if train_override and mode != "curriculum":
        raise ValueError("--train-file is reserved for manifest-locked curriculum arms")
    if mode == "curriculum" and output_dir is None:
        raise ValueError("curriculum arms require separate explicit --output-dir paths")
    train_file = _absolute(train_file or profile.get("train_file", config["data"]["train_file"]))
    val_file = _absolute(profile.get("val_file", config["data"]["val_file"]))
    if train_override:
        if experiment_name is None or not experiment_name.endswith(
            ("_adaptive", "_random", "_fixed")
        ):
            raise ValueError(
                "curriculum arm requires an explicit adaptive/random/fixed experiment name"
            )
        curriculum_path = train_file.parent / "manifest.json"
        curriculum = json.loads(curriculum_path.read_text(encoding="utf-8"))
        if (
            curriculum.get("schema") != "veripy.curriculum.candidates"
            or curriculum.get("strategy") not in {"adaptive", "random", "fixed"}
            or not experiment_name.endswith(f"_{curriculum.get('strategy')}")
            or curriculum.get("benchmark_feedback_used") is not False
            or curriculum.get("train_parquet_sha256")
            != hashlib.sha256(train_file.read_bytes()).hexdigest()
            or curriculum.get("frozen_validation_sha256")
            != hashlib.sha256(val_file.read_bytes()).hexdigest()
        ):
            raise ValueError("curriculum train override failed manifest/hash validation")
        if curriculum.get("strategy") == "fixed":
            source_manifest = Path(curriculum["source_manifest"])
            source_audit = Path(curriculum["source_audit"])
            original = json.loads(source_manifest.read_text())
            audited = json.loads(source_audit.read_text())
            if (
                curriculum.get("generation_controller") != "predefined_nonteacher"
                or hashlib.sha256(source_manifest.read_bytes()).hexdigest()
                != curriculum["source_manifest_sha256"]
                or hashlib.sha256(source_audit.read_bytes()).hexdigest()
                != curriculum["source_audit_sha256"]
                or audited.get("passed") is not True
                or audited["manifest_sha256"] != curriculum["source_manifest_sha256"]
                or original["train_parquet_sha256"] != curriculum["train_parquet_sha256"]
            ):
                raise ValueError("fixed control requires its pre-teacher CPU admission")
    default_output = profile["output_dir"]
    if output_dir is None and os.environ.get("VERIPY_CHECKPOINT_ROOT"):
        default_output = str(Path(os.environ["VERIPY_CHECKPOINT_ROOT"]) / mode)
    target_dir = _absolute(output_dir or default_output)
    checkpoint_path = _absolute(checkpoint) if checkpoint else None

    for required in (model_path, train_file, val_file):
        if not required.exists():
            raise FileNotFoundError(required)
    _validate_model_rope(model_path)
    ResumeGate(resume=resume, checkpoint=checkpoint_path).validate()
    if resume:
        if checkpoint_path is None or "global_step_" not in checkpoint_path.name:
            raise ValueError("veRL resume checkpoint must be a global_step_* directory")
        if mode == "staged_curriculum":
            if checkpoint_path.parent != target_dir:
                raise ValueError("staged curriculum checkpoint must belong to the output directory")
            prior = json.loads(
                (checkpoint_path.parent / "launch_manifest.json").read_text(encoding="utf-8")
            )
            prior_train = prior.get("data_artifacts", {}).get("train", {}).get("sha256")
            if (
                prior.get("mode") != mode
                or prior_train != hashlib.sha256(train_file.read_bytes()).hexdigest()
            ):
                raise ValueError("staged curriculum resume requires the same frozen schedule")
    else:
        ResumeGate.validate_fresh(target_dir)

    resource_lock = _absolute(config["resource_lock"])
    environment_manifest = _absolute(config["environment_manifest"])
    python = _absolute(os.environ.get("VERIPY_PYTHON", config["python"]))
    entrypoint = _absolute(config["entrypoint"])
    if cpu_preflight:
        from veripy_agent.resources import load_validate, verify_local_resources

        locked = load_validate(resource_lock)
        verify_local_resources(locked, resource_lock.parent, names={"verl-agent", "veRL"})
        if not python.is_file() or not entrypoint.is_file():
            raise FileNotFoundError("CPU preflight requires the real pinned entrypoints")
    else:
        ResourceLock(resource_lock, environment_manifest).validate_grpo(
            f"{python} {entrypoint}", allow_exceptions=mode == "smoke"
        )
    protocol = protocol_manifest_from_resources(
        TOOLS,
        resources_lock=resource_lock,
        tokenizer_path=model_path,
        max_turns=int(config["rollout"]["max_turns"]),
    )

    rollout = config["rollout"]
    actor = config["actor"]
    sandbox = config["sandbox"]
    wandb = config["wandb"]
    max_prompt_length = profile.get("max_prompt_length", rollout["max_prompt_length"])
    max_response_length = profile.get("max_response_length", rollout["max_response_length"])
    max_model_len = profile.get("max_model_len", rollout["max_model_len"])
    max_num_seqs = profile.get("max_num_seqs", rollout["group_size"])
    overrides = [
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=false",
        f"algorithm.kl_ctrl.kl_coef={actor['kl_coefficient']}",
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        f"data.train_batch_size={profile['train_batch_size']}",
        f"data.val_batch_size={profile['val_batch_size']}",
        f"data.shuffle={str(profile.get('shuffle', True)).lower()}",
        f"data.max_prompt_length={max_prompt_length}",
        f"data.max_response_length={max_response_length}",
        "data.return_raw_chat=true",
        "data.filter_overlong_prompts=true",
        "data.truncation=error",
        f"actor_rollout_ref.model.path={model_path}",
        f"actor_rollout_ref.model.lora_rank={actor['lora_rank']}",
        f"actor_rollout_ref.model.lora_alpha={actor['lora_alpha']}",
        "actor_rollout_ref.model.target_modules=all-linear",
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        "actor_rollout_ref.model.enable_activation_offload=true",
        "+actor_rollout_ref.actor.fsdp_config.model_dtype=bf16",
        f"actor_rollout_ref.actor.optim.lr={actor['learning_rate']}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={profile['ppo_mini_batch_size']}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        f"actor_rollout_ref.actor.use_kl_loss={str(profile['use_kl_loss']).lower()}",
        f"actor_rollout_ref.actor.kl_loss_coef={actor['kl_coefficient']}",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.use_torch_compile=false",
        "actor_rollout_ref.actor.fsdp_config.param_offload="
        f"{str(profile.get('actor_param_offload', True)).lower()}",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true",
        "actor_rollout_ref.ref.fsdp_config.param_offload=true",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={profile['gpu_memory_utilization']}",
        "actor_rollout_ref.rollout.enforce_eager=true",
        "actor_rollout_ref.rollout.free_cache_engine=true",
        "actor_rollout_ref.rollout.enable_chunked_prefill=false",
        f"actor_rollout_ref.rollout.load_format={profile.get('load_format', 'dummy_dtensor')}",
        "actor_rollout_ref.rollout.layered_summon="
        f"{str(profile.get('layered_summon', False)).lower()}",
        f"actor_rollout_ref.rollout.max_model_len={max_model_len}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={max_model_len}",
        f"actor_rollout_ref.rollout.max_num_seqs={max_num_seqs}",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.multi_turn.enable=true",
        f"actor_rollout_ref.rollout.multi_turn.max_turns={rollout['max_turns']}",
        f"actor_rollout_ref.rollout.multi_turn.format={rollout['format']}",
        "actor_rollout_ref.rollout.multi_turn.tool_config_path="
        f"{_absolute(rollout['tool_config_path'])}",
        "env.env_name=veripy",
        f"env.max_steps={rollout['max_turns']}",
        f"env.rollout.n={rollout['group_size']}",
        f"env.resources_per_worker.num_cpus={sandbox['cpus_per_worker']}",
        "+env.sandbox_backend="
        f"{os.environ.get('VERIPY_SANDBOX_BACKEND', sandbox.get('backend', 'docker'))}",
        "+env.sandbox_user="
        f"{os.environ.get('VERIPY_SANDBOX_USER', sandbox.get('user', 'veripy-sandbox'))}",
        f"+env.sandbox_timeout_seconds={sandbox.get('timeout_seconds', 10)}",
        f"+env.sandbox_cpu_seconds={sandbox.get('cpu_seconds', 5)}",
        f"+env.sandbox_memory_bytes={sandbox.get('memory_bytes', 2 * 1024**3)}",
        f"+env.sandbox_image={sandbox['image']}",
        f"+env.sandbox_image_id={sandbox['image_id']}",
        f"+env.dataset_root={_absolute(sandbox['dataset_root'])}",
        f"+env.scratch_root={target_dir / 'rollouts'}",
        "trainer.critic_warmup=0",
        f"trainer.logger={_hydra_list(['console', 'wandb'])}",
        f"trainer.project_name={wandb['project']}",
        f"trainer.experiment_name={experiment_name or f'qwen3_1_7b_grpo_{mode}'}",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        f"trainer.save_freq={profile['save_freq']}",
        f"trainer.test_freq={profile['test_freq']}",
        f"trainer.total_training_steps={profile['steps']}",
        f"trainer.total_epochs={profile['total_epochs']}",
        f"trainer.val_before_train={str(profile['val_before_train']).lower()}",
        f"trainer.default_local_dir={target_dir}",
        f"trainer.rollout_data_dir={target_dir / 'rollout_samples'}",
        f"trainer.resume_mode={'resume_path' if resume else 'disable'}",
        f"ray_init.num_cpus={profile['ray_cpus']}",
    ]
    if profile.get("ray_object_store_memory") is not None:
        overrides.append(f"+ray_init.object_store_memory={profile['ray_object_store_memory']}")
    if profile.get("ray_include_dashboard") is not None:
        overrides.append(
            f"+ray_init.include_dashboard={str(profile['ray_include_dashboard']).lower()}"
        )
    if checkpoint_path is not None:
        overrides.append(f"trainer.resume_from_path={checkpoint_path}")

    environment = {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS") or "2",
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS") or "2",
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS") or "2",
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS") or "2",
        "BLIS_NUM_THREADS": os.environ.get("BLIS_NUM_THREADS") or "2",
        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS") or "2",
        "WANDB_MODE": os.environ.get("WANDB_MODE") or str(profile.get("wandb_mode", wandb["mode"])),
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT") or str(wandb["project"]),
        "HF_HOME": str(_absolute(os.environ.get("VERIPY_HF_CACHE", config["storage"]["hf_cache"]))),
        "VLLM_CACHE_ROOT": str(
            _absolute(os.environ.get("VERIPY_VLLM_CACHE", config["storage"]["vllm_cache"]))
        ),
        "VLLM_ATTENTION_BACKEND": str(
            os.environ.get("VERIPY_ATTENTION_BACKEND")
            or profile.get("attention_backend", config["runtime"]["attention_backend"])
        ),
        "PYTHONPATH": os.pathsep.join(
            [str(ROOT / "src"), str(ROOT / "external/verl-agent"), str(ROOT / "external/verl")]
        ),
    }
    if profile.get("vllm_use_v1") is not None:
        environment["VLLM_USE_V1"] = "1" if profile["vllm_use_v1"] else "0"
    if profile.get("ray_memory_usage_threshold") is not None:
        environment["RAY_memory_usage_threshold"] = str(profile["ray_memory_usage_threshold"])
    if profile.get("ray_memory_monitor_refresh_ms") is not None:
        environment["RAY_memory_monitor_refresh_ms"] = str(profile["ray_memory_monitor_refresh_ms"])
    return LaunchPlan(
        mode=mode,
        steps=int(profile["steps"]),
        output_dir=target_dir,
        command=[str(python), str(entrypoint), *overrides],
        environment=environment,
        protocol=protocol,
    )


def validate_hydra(plan: LaunchPlan) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    overrides = plan.command[2:]
    config_dir = VERL_CONFIG_DIR
    if not config_dir.exists():
        spec = importlib.util.find_spec("verl")
        if spec is None or spec.origin is None:
            raise RuntimeError("cannot locate veRL Hydra configuration")
        config_dir = Path(spec.origin).resolve().parent / "trainer" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        resolved = compose(config_name="ppo_trainer", overrides=overrides)
    OmegaConf.resolve(resolved)
    return cast(dict[str, Any], OmegaConf.to_container(resolved, resolve=True))


def _write_manifest(plan: LaunchPlan) -> Path:
    path = plan.output_dir / "launch_manifest.json"
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    probe = None
    if plan.mode in {"reward_probe", "feedback_probe"}:
        source = ROOT / f"data/manifests/{plan.mode}.json"
        probe = json.loads(source.read_text(encoding="utf-8"))
    overrides = dict(item.split("=", 1) for item in plan.command[2:] if "=" in item)
    data_artifacts = {
        split: {
            "path": overrides[f"data.{split}_files"],
            "sha256": hashlib.sha256(
                Path(overrides[f"data.{split}_files"]).read_bytes()
            ).hexdigest(),
        }
        for split in ("train", "val")
    }
    curriculum_path = Path(overrides["data.train_files"]).parent / "manifest.json"
    curriculum = (
        {
            "path": str(curriculum_path),
            "sha256": hashlib.sha256(curriculum_path.read_bytes()).hexdigest(),
            "strategy": json.loads(curriculum_path.read_text(encoding="utf-8")).get(
                "strategy", "staged"
            ),
        }
        if plan.mode in {"curriculum", "staged_curriculum"}
        else None
    )
    path.write_text(
        json.dumps(
            {
                "mode": plan.mode,
                "steps": plan.steps,
                "command": plan.command,
                "environment": plan.environment,
                "protocol": plan.protocol,
                "model_merge_receipt": (
                    json.loads(
                        (
                            Path(overrides["actor_rollout_ref.model.path"]) / "merge_receipt.json"
                        ).read_text(encoding="utf-8")
                    )
                    if (
                        Path(overrides["actor_rollout_ref.model.path"]) / "merge_receipt.json"
                    ).is_file()
                    else None
                ),
                "data_artifacts": data_artifacts,
                "curriculum": curriculum,
                "reward_probe": probe if plan.mode == "reward_probe" else None,
                "feedback_probe": probe if plan.mode == "feedback_probe" else None,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _record_exit(path: Path, exit_code: int) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["exit_code"] = exit_code
    manifest["status"] = "completed" if exit_code == 0 else "failed"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode",
        choices=(
            "smoke",
            "reward_probe",
            "feedback_probe",
            "bounded",
            "curriculum_collect",
            "staged_curriculum",
            "curriculum",
            "pilot",
            "formal",
        ),
        default="smoke",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--cpu-preflight",
        action="store_true",
        help="dry-run composition only; does not certify CUDA or merged weights",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cpu_preflight and not args.dry_run:
        raise ValueError("--cpu-preflight is only allowed with --dry-run")
    plan = build_plan(
        config_path=args.config,
        mode=args.mode,
        model=args.model,
        output_dir=args.output_dir,
        resume=args.resume,
        checkpoint=args.checkpoint,
        train_file=args.train_file,
        experiment_name=args.experiment_name,
        cpu_preflight=args.cpu_preflight,
    )
    _validate_episode_paths(plan)
    _validate_probe_manifest(plan)
    _validate_feedback_manifest(plan)
    _validate_staged_curriculum_manifest(plan)
    resolved = validate_hydra(plan)
    print(shlex.join(plan.command))
    print(
        json.dumps(
            {
                "mode": plan.mode,
                "steps": plan.steps,
                "model": resolved["actor_rollout_ref"]["model"]["path"],
                "output_dir": str(plan.output_dir),
                "wandb_mode": plan.environment["WANDB_MODE"],
                "protocol_sha256": plan.protocol["protocol_sha256"],
                "gpu_runtime_verified": False if args.cpu_preflight else None,
                "model_learning_verified": False,
            },
            sort_keys=True,
        )
    )
    if args.dry_run:
        return 0
    manifest = _write_manifest(plan)
    exit_code = subprocess.call(plan.command, cwd=ROOT, env={**os.environ, **plan.environment})
    _record_exit(manifest, exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
