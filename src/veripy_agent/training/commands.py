from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from pathlib import Path


def _args(flags: Mapping[str, object]) -> list[str]:
    out = []
    for k, v in flags.items():
        if v is None or v is False:
            continue
        key = "--" + k.replace("_", "-")
        if v is True:
            out.append(key)
        else:
            out += [key, str(v)]
    return out


def write_sft_config(
    *,
    model: str,
    dataset: str,
    output_dir: str,
    path: str | Path,
    flags: Mapping[str, object] | None = None,
) -> Path:
    if not model or not dataset or not output_dir:
        raise ValueError("model, dataset, and output_dir are required")
    payload: dict[str, object] = {
        "model_name_or_path": model,
        "dataset": dataset,
        "output_dir": output_dir,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "lora_target": "all",
        "dataset_dir": "data/sft",
        "template": "qwen3_nothink",
        "cutoff_len": 8192,
        "val_size": 0.05,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "num_train_epochs": 3.0,
        "bf16": True,
        "gradient_checkpointing": True,
        "eval_strategy": "steps",
        "eval_steps": 50,
        "save_steps": 50,
        "report_to": "wandb",
        "resume_from_checkpoint": None,
    }
    payload.update(flags or {})
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    except ImportError:
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def build_sft_command(
    *,
    model: str,
    dataset: str,
    output_dir: str,
    config_path: str | Path | None = None,
    flags: Mapping[str, object] | None = None,
    executable: str = "llamafactory-cli",
    resource_lock: str | Path | None = None,
    environment_manifest: str | Path | None = None,
) -> list[str]:
    if config_path is None:
        raise ValueError("config_path is required; generate and validate the locked YAML first")
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not resource_lock or not environment_manifest:
        raise ValueError("pinned resource_lock and verified environment_manifest are required")
    try:
        import yaml

        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ImportError:
        config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "model_name_or_path": model,
        "dataset": dataset,
        "output_dir": output_dir,
    }
    mismatches = [key for key, value in expected.items() if config.get(key) != value]
    if config.get("stage") != "sft" or config.get("do_train") is not True or mismatches:
        raise ValueError(f"SFT config does not match requested run: {mismatches}")
    from .gates import ResourceLock

    ResourceLock(resource_lock, environment_manifest).validate_environment("sft")
    return [executable, "train", str(path)]


def build_grpo_command(
    *,
    model: str,
    task: str,
    output_dir: str,
    mode: str = "smoke",
    flags: Mapping[str, object] | None = None,
    executable: str | None = None,
    resource_lock: str | Path | None = None,
    environment_manifest: str | Path | None = None,
    entrypoint: str | None = None,
) -> list[str]:
    if mode not in {"smoke", "pilot", "formal"}:
        raise ValueError("mode must be smoke, pilot, or formal")
    if not resource_lock or not environment_manifest or not entrypoint:
        raise ValueError(
            "pinned resource_lock, verified environment_manifest, and explicit entrypoint are required"
        )
    from .gates import ResourceLock

    ResourceLock(resource_lock, environment_manifest).validate_grpo(
        entrypoint, allow_exceptions=mode == "smoke"
    )
    return [
        *shlex.split(entrypoint),
        "--model",
        model,
        "--task",
        task,
        "--output-dir",
        output_dir,
        "--run-mode",
        mode,
        *_args(flags or {}),
    ]
