#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from veripy_agent.environment import HardenedProcessWorker, SandboxPaths


def main() -> int:
    report = HardenedProcessWorker.capability_report()
    if not report.available:
        raise SystemExit(report.reason)
    root = Path(tempfile.mkdtemp(prefix="veripy-hardened-", dir="/tmp"))
    root.chmod(0o755)
    try:
        inputs = root / "inputs"
        inputs.mkdir(mode=0o755)
        (inputs / "input.txt").write_text("immutable", encoding="utf-8")
        worker = HardenedProcessWorker(SandboxPaths(inputs, root / "scratch"), timeout=20)
        results = {
            "read_write": worker.run(
                "from pathlib import Path; "
                "print(Path('/inputs/input.txt').read_text()); "
                "Path('/scratch/result.txt').write_text('ok')"
            ),
            "input_write": worker.run("open('/inputs/input.txt', 'w').write('changed')"),
            "host_read": worker.run("print(open('/etc/passwd').read())"),
            "network": worker.run("import socket; socket.socket()"),
            "pandas": worker.run("import pandas as pd; print(int(pd.Series([1, 2, 3]).sum()))"),
            "timeout": HardenedProcessWorker(
                SandboxPaths(inputs, root / "timeout-scratch"), timeout=0.2
            ).run("while True: pass"),
        }
        expected = {
            "read_write": "ok",
            "input_write": "security_violation",
            "host_read": "security_violation",
            "network": "security_violation",
            "pandas": "ok",
            "timeout": "timeout",
        }
        actual = {name: result.status.value for name, result in results.items()}
        payload = {
            "capability": report.reason,
            "statuses": actual,
            "read_stdout": results["read_write"].stdout.strip(),
            "pandas_stdout": results["pandas"].stdout.strip(),
            "input_unchanged": (inputs / "input.txt").read_text(encoding="utf-8")
            == "immutable",
        }
        print(json.dumps(payload, sort_keys=True))
        if actual != expected or not payload["input_unchanged"]:
            return 1
        if payload["read_stdout"] != "immutable" or payload["pandas_stdout"] != "6":
            return 1
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
