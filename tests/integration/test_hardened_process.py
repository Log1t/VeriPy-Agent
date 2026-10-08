import subprocess
import time
from pathlib import Path

import pytest

from veripy_agent.environment import HardenedProcessWorker, SandboxPaths
from veripy_agent.environment.worker import MAX_STDERR_CHARS, MAX_STDOUT_CHARS


def _worker(tmp_path: Path) -> HardenedProcessWorker:
    report = HardenedProcessWorker.capability_report()
    if not report.available:
        pytest.skip(report.reason)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "x.txt").write_text("immutable", encoding="utf-8")
    for parent in (tmp_path.parent.parent, tmp_path.parent, tmp_path, inputs):
        subprocess.run(
            ["setfacl", "-m", "u:veripy-sandbox:--x", str(parent)],
            check=True,
        )
    subprocess.run(
        ["setfacl", "-m", "u:veripy-sandbox:r-x", str(inputs)],
        check=True,
    )
    return HardenedProcessWorker(SandboxPaths(inputs, tmp_path / "scratch"), timeout=2)


def test_hardened_worker_maps_episode_paths_and_persists_scratch(tmp_path):
    worker = _worker(tmp_path)
    first = worker.run(
        "from pathlib import Path; "
        "print(Path('/inputs/x.txt').read_text()); "
        "Path('/scratch/state').write_text('kept')"
    )
    second = worker.run("from pathlib import Path; print(Path('/scratch/state').read_text())")
    assert first.status.value == "ok" and first.stdout.strip() == "immutable"
    assert second.status.value == "ok" and second.stdout.strip() == "kept"


def test_hardened_worker_blocks_input_writes_host_reads_and_network(tmp_path):
    worker = _worker(tmp_path)
    input_write = worker.run("open('/inputs/x.txt', 'w').write('changed')")
    host_read = worker.run("print(open('/etc/passwd').read())")
    network = worker.run("import socket; socket.socket()")
    assert input_write.status.value == "security_violation"
    assert host_read.status.value == "security_violation"
    assert network.status.value == "security_violation"
    assert (worker.paths.inputs / "x.txt").read_text(encoding="utf-8") == "immutable"


def test_hardened_worker_enforces_timeout(tmp_path):
    worker = _worker(tmp_path)
    result = worker.run("while True: pass")
    assert result.status.value == "timeout"
    assert result.failure_code.value == "timeout"


def test_hardened_worker_kills_descendants_after_success(tmp_path):
    worker = _worker(tmp_path)
    marker = worker.paths.scratch / "leaked-child"
    result = worker.run(
        "import os, time; "
        "pid = os.fork(); "
        "time.sleep(0.5) if pid == 0 else None; "
        "open('/scratch/leaked-child', 'w').write('alive') if pid == 0 else None"
    )
    time.sleep(0.8)
    assert result.status.value == "ok"
    assert not marker.exists()


def test_hardened_worker_declares_output_truncation(tmp_path):
    worker = _worker(tmp_path)
    result = worker.run(
        f"import sys; print('x' * {MAX_STDOUT_CHARS + 10}); "
        f"sys.stderr.write('y' * {MAX_STDERR_CHARS + 10})"
    )
    assert result.status.value == "ok"
    assert len(result.stdout) == MAX_STDOUT_CHARS
    assert len(result.stderr) == MAX_STDERR_CHARS
    assert result.resource_usage["output_truncated"] == {"stdout": True, "stderr": True}
