#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from veripy_agent.training.commands import build_sft_command
from veripy_agent.training.gates import ResumeGate


def command(
    *,
    config_path: str | Path,
    resource_lock: str | Path,
    environment_manifest: str | Path,
    resume: bool = False,
    checkpoint: str | Path | None = None,
    executable: str | None = None,
) -> list[str]:
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    ResumeGate(resume=resume, checkpoint=checkpoint).validate()
    configured_checkpoint = config.get("resume_from_checkpoint")
    if resume:
        if configured_checkpoint not in (None, str(checkpoint)):
            raise ValueError("config contains a different resume_from_checkpoint")
        config["resume_from_checkpoint"] = str(checkpoint)
        resolved_dir = output_dir.parent / "manifests"
        resolved_dir.mkdir(parents=True, exist_ok=True)
        config_file = resolved_dir / f"{output_dir.name}_resume.yaml"
        config_file.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    else:
        if configured_checkpoint is not None:
            raise ValueError("fresh SFT config must set resume_from_checkpoint to null")
        ResumeGate.validate_fresh(output_dir)
    model_path = os.environ.get("VERIPY_MODEL_PATH", config["model_name_or_path"])
    if model_path != config["model_name_or_path"]:
        config["model_name_or_path"] = model_path
        resolved_dir = output_dir.parent / "manifests"
        resolved_dir.mkdir(parents=True, exist_ok=True)
        config_file = resolved_dir / f"{output_dir.name}_model.yaml"
        config_file.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return build_sft_command(
        model=model_path,
        dataset=config["dataset"],
        output_dir=config["output_dir"],
        config_path=config_file,
        executable=executable or str(Path(sys.executable).parent / "llamafactory-cli"),
        resource_lock=resource_lock,
        environment_manifest=environment_manifest,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resource-lock", type=Path, default=Path("resources.lock.json"))
    parser.add_argument(
        "--environment-manifest",
        type=Path,
        default=Path("configs/resources/sft_environment.json"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    argv = command(
        config_path=args.config,
        resource_lock=args.resource_lock,
        environment_manifest=args.environment_manifest,
        resume=args.resume,
        checkpoint=args.checkpoint,
    )
    print(json.dumps({"command": argv, "execute": args.execute}))
    if args.execute:
        subprocess.run(argv, check=True)


if __name__ == "__main__":
    main()
