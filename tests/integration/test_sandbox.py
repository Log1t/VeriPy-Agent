import shutil
import subprocess

import pytest

from veripy_agent.environment.paths import SandboxPaths
from veripy_agent.environment.worker import (
    MAX_STDERR_CHARS,
    MAX_STDOUT_CHARS,
    DockerSandboxWorker,
    LocalSandboxWorker,
)
from veripy_agent.tools.basic import execute_python


def test_worker_protocol_reports_timeout(tmp_path):
    (tmp_path / "inputs").mkdir()
    paths = SandboxPaths(tmp_path / "inputs", tmp_path / "scratch")
    r = execute_python(paths, "while True: pass", worker=LocalSandboxWorker(paths, timeout=0.1))
    assert r.status.value == "timeout" and r.failure_code.value == "timeout"


def test_host_fallback_is_disabled(tmp_path):
    (tmp_path / "inputs").mkdir()
    paths = SandboxPaths(tmp_path / "inputs", tmp_path / "scratch")
    result = execute_python(paths, "print('unsafe')")
    assert result.failure_code.value == "security_violation"


def test_local_worker_declares_output_truncation(tmp_path):
    (tmp_path / "inputs").mkdir()
    paths = SandboxPaths(tmp_path / "inputs", tmp_path / "scratch")
    result = LocalSandboxWorker(paths).run(
        f"import sys; print('x' * {MAX_STDOUT_CHARS + 10}); "
        f"sys.stderr.write('y' * {MAX_STDERR_CHARS + 10})"
    )
    assert result.status.value == "ok"
    assert len(result.stdout) == MAX_STDOUT_CHARS
    assert len(result.stderr) == MAX_STDERR_CHARS
    assert result.resource_usage["output_truncated"] == {"stdout": True, "stderr": True}


def test_persistent_docker_worker_enforces_mounts_and_network(tmp_path):
    if shutil.which("docker") is None:
        pytest.skip("docker CLI is unavailable")
    inspected = subprocess.run(
        ["docker", "image", "inspect", "veripy-python-sandbox:latest"],
        capture_output=True,
        check=False,
    )
    if inspected.returncode:
        pytest.skip("sandbox image is not built")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "x.txt").write_text("immutable")
    paths = SandboxPaths(inputs, tmp_path / "scratch")
    worker = DockerSandboxWorker(paths, "veripy-python-sandbox:latest", timeout=2)
    try:
        first = worker.run("from pathlib import Path; Path('/scratch/state').write_text('kept')")
        second = worker.run("from pathlib import Path; print(Path('/scratch/state').read_text())")
        read_only = worker.run("open('/inputs/x.txt', 'w').write('changed')")
        network = worker.run("import urllib.request; urllib.request.urlopen('https://example.com')")
    finally:
        worker.close()
    assert first.status.value == "ok"
    assert second.status.value == "ok" and second.stdout.strip() == "kept"
    assert read_only.status.value == "error" and (inputs / "x.txt").read_text() == "immutable"
    assert network.status.value == "error"
