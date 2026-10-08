from pathlib import Path

from veripy_agent.environment.launcher import build_docker_command


def test_nested_trainer_maps_container_paths_to_host(monkeypatch, tmp_path: Path) -> None:
    container_data = tmp_path / "container-data"
    host_data = tmp_path / "host-data"
    container_scratch = tmp_path / "container-checkpoints"
    host_scratch = tmp_path / "host-checkpoints"
    monkeypatch.setenv(
        "VERIPY_SANDBOX_PATH_MAP",
        f"{container_data}={host_data};{container_scratch}={host_scratch}",
    )
    command = build_docker_command(
        "sandbox",
        container_data / "episode",
        container_scratch / "rollouts" / "task",
    )
    assert f"{host_data / 'episode'}:/inputs:ro" in command
    assert f"{host_scratch / 'rollouts' / 'task'}:/scratch:rw" in command
