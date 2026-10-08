"""Validation, hashing, and local verification for pinned resources."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

SCHEMA = "veripy.resources.lock"
SCHEMA_VERSION = 1
EXPECTED_SOURCES = {
    "Qwen/Qwen3-1.7B": "https://huggingface.co/Qwen/Qwen3-1.7B",
    "LLaMA-Factory": "https://github.com/hiyouga/LLaMA-Factory",
    "verl-agent": "https://github.com/langfengQ/verl-agent",
    "veRL": "https://github.com/volcengine/verl",
    "python:3.12-slim": "python:3.12-slim",
}
RESOURCE_TYPES = {"model", "git-archive", "docker-base"}
ENVIRONMENT_PROFILES: dict[str, dict[str, Any]] = {
    "sft-a100-cu124": {
        "stack": "sft",
        "torch_cuda": "12.4",
        "packages": {
            "torch": "2.6.0",
            "transformers": "5.8.0",
            "llamafactory": "0.9.6.dev0",
            "datasets": "4.0.0",
            "accelerate": "1.11.0",
            "peft": "0.18.1",
            "trl": "0.24.0",
            "flash-attn": "2.7.4.post1",
        },
    },
}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_validate(path: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid lock JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError("invalid resource lock schema")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {SCHEMA_VERSION}")
    if data.get("verification") != {"algorithm": "sha256"}:
        raise ValueError("sha256 verification required")

    resources = data.get("resources")
    if not isinstance(resources, list):
        raise ValueError("resources must be a list")
    names = [resource.get("name") for resource in resources if isinstance(resource, dict)]
    if len(names) != len(resources) or len(names) != len(set(names)):
        raise ValueError("resource names must be unique strings")
    if set(names) != set(EXPECTED_SOURCES):
        raise ValueError("required resource set mismatch")

    for resource in resources:
        name = resource["name"]
        if resource.get("source") != EXPECTED_SOURCES[name]:
            raise ValueError(f"{name}: unexpected source")
        if resource.get("type") not in RESOURCE_TYPES:
            raise ValueError(f"{name}: invalid resource type")
        if not isinstance(resource.get("revision"), str) or not resource["revision"]:
            raise ValueError(f"{name}: revision required")
        if not _is_sha256(resource.get("sha256")):
            raise ValueError(f"{name}: invalid sha256")
        if resource.get("status") not in {"pinned", "verified"}:
            raise ValueError(f"{name}: invalid status")
        if not isinstance(resource.get("license"), str) or not resource["license"].strip():
            raise ValueError(f"{name}: license required")
    return data


def canonical_hash(data: dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _model_snapshot_hash(path: Path) -> str:
    if not path.is_dir():
        raise ValueError(f"model path does not exist: {path}")
    aggregate = hashlib.sha256()
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and ".cache" not in item.relative_to(path).parts
    )
    if not files:
        raise ValueError(f"model path is empty: {path}")
    for item in files:
        file_hash = hashlib.sha256()
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                file_hash.update(chunk)
        digest = file_hash.hexdigest()
        relative = item.relative_to(path).as_posix()
        aggregate.update(f"{digest}  ./{relative}\n".encode())
    return aggregate.hexdigest()


def _git_archive_hash(path: Path, revision: str) -> str:
    if not (path / ".git").is_dir():
        raise ValueError(f"git checkout does not exist: {path}")
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != revision:
        raise ValueError(f"git revision mismatch: {path}: {head} != {revision}")
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError(f"git checkout is dirty: {path}")
    archive = subprocess.run(
        ["git", "-C", str(path), "archive", "--format=tar", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(archive).hexdigest()


def _docker_digest(source: str) -> str:
    output = subprocess.run(
        ["docker", "image", "inspect", source, "--format", "{{json .RepoDigests}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    digests = json.loads(output)
    if not digests:
        raise ValueError(f"docker image has no repository digest: {source}")
    return str(digests[0]).rsplit("@sha256:", maxsplit=1)[-1]


def verify_local_resources(
    data: dict[str, Any], repo_root: str | Path, names: set[str] | None = None
) -> dict[str, str]:
    root = Path(repo_root)
    verified: dict[str, str] = {}
    git_paths = {
        "LLaMA-Factory": root / "external" / "LLaMA-Factory",
        "verl-agent": root / "external" / "verl-agent",
        "veRL": root / "external" / "verl",
    }
    for resource in data["resources"]:
        name = resource["name"]
        if names is not None and name not in names:
            continue
        if resource["type"] == "model":
            environment_name = resource.get("local_path_env")
            configured_path = resource["local_path"]
            if isinstance(environment_name, str):
                configured_path = os.environ.get(environment_name, configured_path)
            actual = _model_snapshot_hash(Path(configured_path))
        elif resource["type"] == "git-archive":
            actual = _git_archive_hash(git_paths[name], resource["revision"])
        else:
            actual = _docker_digest(resource["source"])
        if actual != resource["sha256"]:
            raise ValueError(f"{name}: local sha256 mismatch: {actual}")
        verified[name] = actual
    return verified


def validate_environment_manifest(
    path: str | Path, lock_hash: str, expected_name: str | None = None
) -> dict[str, Any]:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid RL environment manifest") from exc
    if not isinstance(manifest, dict) or manifest.get("status") != "verified":
        raise ValueError("RL environment manifest must be verified")
    if manifest.get("resource_lock_sha256") != lock_hash:
        raise ValueError("RL environment manifest does not match resource lock")
    if manifest.get("schema") != "veripy.environment.manifest":
        raise ValueError("invalid environment manifest schema")
    if manifest.get("schema_version") != 1:
        raise ValueError("invalid environment manifest schema_version")
    if expected_name is not None and manifest.get("name") != expected_name:
        raise ValueError(f"expected {expected_name} environment manifest")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("packages"), dict):
        raise ValueError("environment manifest packages are required")
    expected_by_stack = {
        "sft": {
            "torch": "2.8.0",
            "transformers": "5.8.0",
            "llamafactory": "0.9.6.dev0",
        },
        "rl": {
            "torch": "2.6.0",
            "transformers": "4.51.1",
            "vllm": "0.8.5",
            "verl": "0.3.1.dev0",
        },
    }
    stack = expected_name or manifest.get("name")
    if not isinstance(stack, str):
        raise ValueError("environment manifest name is required")
    expected = expected_by_stack.get(stack, {})
    profile_name = manifest.get("profile")
    if profile_name is not None:
        if not isinstance(profile_name, str) or profile_name not in ENVIRONMENT_PROFILES:
            raise ValueError("unknown environment profile")
        profile = ENVIRONMENT_PROFILES[profile_name]
        if profile["stack"] != stack:
            raise ValueError("environment profile does not match stack")
        if runtime.get("torch_cuda") != profile["torch_cuda"]:
            raise ValueError("environment profile CUDA mismatch")
        expected = profile["packages"]
    for package, version in expected.items():
        actual = runtime["packages"].get(package, "")
        if str(actual).split("+", maxsplit=1)[0] != version:
            raise ValueError(f"RL environment requires {package} {version}, got {actual}")
    if runtime.get("cuda_available") is not True:
        raise ValueError("RL environment must expose CUDA")
    if expected_name == "rl":
        model = manifest.get("model")
        if not isinstance(model, dict) or not _is_sha256(model.get("sha256")):
            raise ValueError("RL environment manifest must pin the initialization model")
        path_env = model.get("path_env")
        configured_path = model.get("local_path")
        if isinstance(path_env, str):
            configured_path = os.environ.get(path_env, configured_path)
        if not isinstance(configured_path, str) or not configured_path:
            raise ValueError("RL initialization model path is required")
        actual_model_hash = _model_snapshot_hash(Path(configured_path))
        if actual_model_hash != model["sha256"]:
            raise ValueError("RL initialization model sha256 mismatch")
    return manifest
