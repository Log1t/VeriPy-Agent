from __future__ import annotations

import os
import pwd
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from veripy_agent.contracts import FailureCode, ToolResult, ToolStatus

from .hardened_runner import SECURITY_EXIT
from .paths import SandboxPaths
from .worker import MAX_STDERR_CHARS, MAX_STDOUT_CHARS, bound_tool_result


@dataclass(frozen=True)
class HardenedCapabilityReport:
    available: bool
    reason: str


class HardenedProcessWorker:
    """Fail-closed process sandbox for managed hosts without a Docker daemon."""

    def __init__(
        self,
        paths: SandboxPaths,
        *,
        user: str = "veripy-sandbox",
        timeout: float = 10,
        cpu_seconds: int = 5,
        memory_bytes: int = 2 * 1024 * 1024 * 1024,
    ):
        report = self.capability_report(user=user)
        if not report.available:
            raise RuntimeError(f"hardened process sandbox unavailable: {report.reason}")
        account = pwd.getpwnam(user)
        self.paths = paths
        self.user = user
        self.uid = account.pw_uid
        self.gid = account.pw_gid
        self.timeout = timeout
        self.cpu_seconds = cpu_seconds
        self.memory_bytes = memory_bytes
        os.chown(self.paths.scratch, self.uid, self.gid)
        os.chmod(self.paths.scratch, 0o700)
        self._validate_episode_paths()

    def _validate_episode_paths(self) -> None:
        command = self._build_command(uid=self.uid, gid=self.gid)[:-3]
        checks = (
            ("readable input directory", ["/usr/bin/test", "-r", str(self.paths.inputs)]),
            ("searchable input directory", ["/usr/bin/test", "-x", str(self.paths.inputs)]),
            ("searchable scratch directory", ["/usr/bin/test", "-x", str(self.paths.scratch)]),
        )
        for label, check in checks:
            result = subprocess.run([*command, *check], check=False)
            if result.returncode != 0:
                raise RuntimeError(f"sandbox user lacks {label}: {check[-1]}")

    @classmethod
    @cache
    def capability_report(cls, *, user: str = "veripy-sandbox") -> HardenedCapabilityReport:
        if sys.platform != "linux":
            return HardenedCapabilityReport(False, "Linux is required")
        if os.geteuid() != 0:
            return HardenedCapabilityReport(False, "root launcher privileges are required")
        missing = [name for name in ("setpriv",) if shutil.which(name) is None]
        if missing:
            return HardenedCapabilityReport(False, f"missing executables: {', '.join(missing)}")
        try:
            account = pwd.getpwnam(user)
        except KeyError:
            return HardenedCapabilityReport(False, f"missing sandbox user: {user}")
        runner = Path(__file__).with_name("hardened_runner.py")
        if not runner.is_file():
            return HardenedCapabilityReport(False, "hardened runner is missing")
        probe = subprocess.run(
            [
                shutil.which("setpriv") or "setpriv",
                f"--reuid={account.pw_uid}",
                f"--regid={account.pw_gid}",
                "--clear-groups",
                "--bounding-set=-all",
                "--inh-caps=-all",
                "--ambient-caps=-all",
                "--no-new-privs",
                "/usr/bin/test",
                "-x",
                sys.executable,
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        if probe.returncode != 0:
            return HardenedCapabilityReport(False, "sandbox user cannot execute the Python runtime")
        with tempfile.TemporaryDirectory(prefix="veripy-capability-", dir="/tmp") as root_value:
            root = Path(root_value)
            root.chmod(0o755)
            inputs = root / "inputs"
            scratch = root / "scratch"
            inputs.mkdir(mode=0o755)
            scratch.mkdir(mode=0o700)
            os.chown(scratch, account.pw_uid, account.pw_gid)
            environment = cls._environment(
                inputs=inputs,
                scratch=scratch,
                uid=account.pw_uid,
                gid=account.pw_gid,
                cpu_seconds=2,
                memory_bytes=512 * 1024 * 1024,
            )
            command = cls._build_command(
                uid=account.pw_uid,
                gid=account.pw_gid,
            )
            hardened_probe = subprocess.run(
                command,
                input="print('ready')",
                cwd=scratch,
                env=environment,
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
            )
            if hardened_probe.returncode != 0 or hardened_probe.stdout.strip() != "ready":
                diagnostics = hardened_probe.stderr.strip()[-500:]
                return HardenedCapabilityReport(
                    False, f"Landlock/seccomp execution probe failed: {diagnostics}"
                )
        return HardenedCapabilityReport(True, "hardened process prerequisites are available")

    @staticmethod
    def _build_command(*, uid: int, gid: int) -> list[str]:
        setpriv = shutil.which("setpriv")
        if setpriv is None:
            raise RuntimeError("hardened process prerequisites disappeared")
        runner = Path(__file__).with_name("hardened_runner.py")
        return [
            setpriv,
            f"--reuid={uid}",
            f"--regid={gid}",
            "--clear-groups",
            "--bounding-set=-all",
            "--inh-caps=-all",
            "--ambient-caps=-all",
            "--no-new-privs",
            sys.executable,
            "-I",
            str(runner),
        ]

    @staticmethod
    def _environment(
        *, inputs: Path, scratch: Path, uid: int, gid: int, cpu_seconds: int, memory_bytes: int
    ) -> dict[str, str]:
        return {
            "BLIS_NUM_THREADS": "2",
            "HOME": str(scratch),
            "LANG": "C.UTF-8",
            "MKL_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "PATH": "/usr/bin:/bin",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "TMPDIR": str(scratch),
            "VECLIB_MAXIMUM_THREADS": "2",
            "VERIPY_SANDBOX_CPU_SECONDS": str(cpu_seconds),
            "VERIPY_SANDBOX_GID": str(gid),
            "VERIPY_SANDBOX_MEMORY_BYTES": str(memory_bytes),
            "VERIPY_SANDBOX_INPUTS": str(inputs),
            "VERIPY_SANDBOX_SCRATCH": str(scratch),
            "VERIPY_SANDBOX_UID": str(uid),
        }

    def _command(self) -> list[str]:
        return self._build_command(
            uid=self.uid,
            gid=self.gid,
        )

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen[str]) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return

    def run(self, code: str) -> ToolResult:
        started = time.monotonic()
        environment = self._environment(
            inputs=self.paths.inputs,
            scratch=self.paths.scratch,
            uid=self.uid,
            gid=self.gid,
            cpu_seconds=self.cpu_seconds,
            memory_bytes=self.memory_bytes,
        )
        process: subprocess.Popen[str] | None = None
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout_file, tempfile.TemporaryFile(
            mode="w+t", encoding="utf-8"
        ) as stderr_file:
            try:
                process = subprocess.Popen(
                    self._command(),
                    cwd=self.paths.scratch,
                    env=environment,
                    stdin=subprocess.PIPE,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                    start_new_session=True,
                )
                process.communicate(code, timeout=self.timeout)
                self._terminate_process_group(process)
            except subprocess.TimeoutExpired:
                if process is not None:
                    self._terminate_process_group(process)
                    process.wait(timeout=2)
                return ToolResult(
                    status=ToolStatus.TIMEOUT,
                    stderr="execution timed out",
                    failure_code=FailureCode.TIMEOUT,
                    duration=time.monotonic() - started,
                )
            except Exception as exc:
                if process is not None and process.poll() is None:
                    self._terminate_process_group(process)
                    process.wait(timeout=2)
                return ToolResult(
                    status=ToolStatus.ERROR,
                    stderr=str(exc),
                    failure_code=FailureCode.EXECUTION_ERROR,
                    duration=time.monotonic() - started,
                )
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read(MAX_STDOUT_CHARS + 1)
            stderr = stderr_file.read()
        duration = time.monotonic() - started
        if process.returncode == 0:
            return bound_tool_result(
                ToolResult(status=ToolStatus.OK, stdout=stdout, stderr=stderr, duration=duration)
            )
        if process.returncode == -signal.SIGXCPU:
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                stderr=stderr[-MAX_STDERR_CHARS:] or "CPU time limit exceeded",
                failure_code=FailureCode.TIMEOUT,
                duration=duration,
            )
        security_return_codes = {
            SECURITY_EXIT,
            -signal.SIGSYS,
            128 + signal.SIGSYS,
        }
        policy_signal = f"signal {signal.SIGSYS}" in stderr
        if process.returncode in security_return_codes or process.returncode < 0 or policy_signal:
            return ToolResult(
                status=ToolStatus.SECURITY_VIOLATION,
                stderr=stderr[-MAX_STDERR_CHARS:] or "sandbox policy terminated execution",
                failure_code=FailureCode.SECURITY_VIOLATION,
                duration=duration,
            )
        return ToolResult(
            status=ToolStatus.ERROR,
            stderr=stderr[-MAX_STDERR_CHARS:],
            failure_code=FailureCode.EXECUTION_ERROR,
            duration=duration,
        )

    def close(self) -> None:
        return None
