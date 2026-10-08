import os
from pathlib import Path


def _docker_host_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    mappings = os.environ.get("VERIPY_SANDBOX_PATH_MAP", "")
    for item in mappings.split(";"):
        if not item or "=" not in item:
            continue
        container_root_text, host_root_text = item.split("=", 1)
        container_root = Path(container_root_text).resolve()
        if resolved == container_root or container_root in resolved.parents:
            return Path(host_root_text).resolve() / resolved.relative_to(container_root)
    return resolved


def build_docker_command(
    image: str,
    inputs: str | Path,
    scratch: str | Path,
    timeout: int = 10,
    container_name: str | None = None,
) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--interactive",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--network=none",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--read-only",
        "--pids-limit=256",
        "--memory=2g",
        "--cpus=2",
        "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
        "-v",
        f"{_docker_host_path(inputs)}:/inputs:ro",
        "-v",
        f"{_docker_host_path(scratch)}:/scratch:rw",
        "-e",
        f"SANDBOX_TIMEOUT={int(timeout)}",
    ]
    if container_name:
        command.extend(["--name", container_name])
    command.append(image)
    return command
