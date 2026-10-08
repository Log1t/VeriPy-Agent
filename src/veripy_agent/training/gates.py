from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from veripy_agent.resources import (
    canonical_hash,
    load_validate,
    validate_environment_manifest,
    verify_local_resources,
)


class ResourceLock:
    """Fail-closed marker lock; acquisition is explicit and never overwrites."""

    def __init__(self, path: str | Path, environment_manifest: str | Path | None = None):
        self.path = Path(path)
        self.environment_manifest = Path(environment_manifest) if environment_manifest else None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError as e:
            raise RuntimeError(f"resource locked: {self.path}") from e

    def release(self) -> None:
        self.path.unlink(missing_ok=True)

    def validate_environment(self, expected_name: str, *, allow_exceptions: bool = False) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        lock = load_validate(self.path)
        if any(resource["status"] != "verified" for resource in lock["resources"]):
            raise ValueError("training requires every resource to be verified")
        required_resources = (
            {"verl-agent", "veRL"}
            if expected_name == "rl"
            else {"Qwen/Qwen3-1.7B", "LLaMA-Factory"}
        )
        verify_local_resources(lock, self.path.resolve().parent, names=required_resources)
        if self.environment_manifest is None:
            raise ValueError(f"verified {expected_name} environment manifest required")
        manifest = validate_environment_manifest(
            self.environment_manifest, canonical_hash(lock), expected_name=expected_name
        )
        if manifest.get("exceptions") and not allow_exceptions:
            raise ValueError(f"{expected_name} environment has unresolved exceptions")

    def validate_grpo(self, entrypoint: str, *, allow_exceptions: bool = False) -> None:
        self.validate_environment("rl", allow_exceptions=allow_exceptions)
        if not isinstance(entrypoint, str) or not entrypoint.strip():
            raise ValueError("explicit GRPO entrypoint required")
        argv = shlex.split(entrypoint)
        if not argv or any(not x.strip() for x in argv):
            raise ValueError("invalid GRPO entrypoint")
        executable = Path(argv[0])
        if executable.is_absolute() or "/" in argv[0]:
            if not executable.is_file():
                raise ValueError("GRPO entrypoint path does not exist")
        elif not argv[0].replace("-", "").replace("_", "").isalnum():
            raise ValueError("invalid GRPO executable")


@dataclass(frozen=True)
class DryRun:
    enabled: bool = True

    def check(self, *, execute: bool = False) -> None:
        if self.enabled and execute:
            raise RuntimeError("dry-run gate: execution is disabled")


class ResumeGate:
    def __init__(self, *, resume: bool = False, checkpoint: str | Path | None = None):
        self.resume, self.checkpoint = resume, Path(checkpoint) if checkpoint else None

    def validate(self) -> None:
        if self.resume and self.checkpoint is None:
            raise ValueError("--resume requires an explicit checkpoint")
        checkpoint = self.checkpoint
        if self.resume and checkpoint is not None and not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

    @staticmethod
    def validate_fresh(output_dir: str | Path, optimizer_path: str | Path | None = None) -> None:
        out = Path(output_dir)
        if out.exists() and any(out.iterdir()):
            raise RuntimeError("resume=false refuses non-empty output directory")
        if optimizer_path and Path(optimizer_path).exists():
            raise RuntimeError("resume=false refuses old optimizer state")


def accept_global_step(before: int, after: int) -> bool:
    return after == before + 1


def validate_global_step(before: int, after: int) -> None:
    if not accept_global_step(before, after):
        raise RuntimeError(f"global step must advance by one: {before} -> {after}")
