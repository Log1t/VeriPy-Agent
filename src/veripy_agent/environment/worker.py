import json
import resource
import selectors
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Protocol, TextIO

from veripy_agent.contracts import FailureCode, ToolResult, ToolStatus

from .launcher import build_docker_command
from .paths import SandboxPaths

MAX_STDOUT_CHARS = 1_000_000
MAX_STDERR_CHARS = 4_000


def _read_prefix(stream: TextIO, limit: int) -> tuple[str, bool]:
    stream.seek(0)
    value = stream.read(limit + 1)
    return value[:limit], len(value) > limit


def _read_tail(stream: TextIO, limit: int) -> tuple[str, bool]:
    stream.seek(0)
    value = stream.read()
    return value[-limit:], len(value) > limit


def bound_tool_result(result: ToolResult) -> ToolResult:
    stdout_truncated = len(result.stdout) > MAX_STDOUT_CHARS
    stderr_truncated = len(result.stderr) > MAX_STDERR_CHARS
    if not stdout_truncated and not stderr_truncated:
        return result
    usage = dict(result.resource_usage)
    usage["output_truncated"] = {
        "stdout": stdout_truncated,
        "stderr": stderr_truncated,
    }
    return result.model_copy(
        update={
            "stdout": result.stdout[:MAX_STDOUT_CHARS],
            "stderr": result.stderr[-MAX_STDERR_CHARS:],
            "resource_usage": usage,
        }
    )


class PythonWorker(Protocol):
    def run(self, code: str) -> ToolResult: ...


class SandboxWorker:
    def __init__(
        self, paths: SandboxPaths, timeout=10, cpu_seconds=5, memory_bytes=2 * 1024 * 1024 * 1024
    ):
        self.paths, self.timeout, self.cpu_seconds, self.memory_bytes = (
            paths,
            timeout,
            cpu_seconds,
            memory_bytes,
        )

    def run(self, code: str) -> ToolResult:
        started = time.monotonic()

        def limits():
            resource.setrlimit(resource.RLIMIT_CPU, (self.cpu_seconds, self.cpu_seconds))
            resource.setrlimit(resource.RLIMIT_AS, (self.memory_bytes, self.memory_bytes))

        try:
            with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout_file, tempfile.TemporaryFile(
                mode="w+t", encoding="utf-8"
            ) as stderr_file:
                p = subprocess.run(
                    [sys.executable, "-I", "-c", code],
                    cwd=self.paths.scratch,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    check=False,
                    text=True,
                    timeout=self.timeout,
                    preexec_fn=limits,
                )
                stdout, stdout_truncated = _read_prefix(stdout_file, MAX_STDOUT_CHARS)
                stderr, stderr_truncated = _read_tail(stderr_file, MAX_STDERR_CHARS)
            usage = (
                {"output_truncated": {"stdout": stdout_truncated, "stderr": stderr_truncated}}
                if stdout_truncated or stderr_truncated
                else {}
            )
            if p.returncode != 0:
                return ToolResult(
                    status=ToolStatus.SECURITY_VIOLATION if p.returncode < 0 else ToolStatus.ERROR,
                    stdout=stdout,
                    stderr=stderr,
                    resource_usage=usage,
                    failure_code=FailureCode.SECURITY_VIOLATION
                    if p.returncode < 0
                    else FailureCode.EXECUTION_ERROR,
                    duration=time.monotonic() - started,
                )
            return ToolResult(
                status=ToolStatus.OK,
                stdout=stdout,
                stderr=stderr,
                resource_usage=usage,
                duration=time.monotonic() - started,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                stderr="execution timed out",
                failure_code=FailureCode.TIMEOUT,
                duration=time.monotonic() - started,
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                stderr=str(e),
                failure_code=FailureCode.EXECUTION_ERROR,
                duration=time.monotonic() - started,
            )

    def serve(self, stream):
        for line in stream:
            try:
                req = json.loads(line)
                result = self.run(req.get("code", ""))
            except Exception as e:
                result = ToolResult(
                    status=ToolStatus.ERROR, stderr=str(e), failure_code=FailureCode.INVALID_REQUEST
                )
            print(result.model_dump_json(), flush=True)


class LocalSandboxWorker(SandboxWorker):
    """Explicit development-only local worker; production uses the container client."""

    dev_only = True


class DockerSandboxWorker:
    """Lazy persistent Docker worker used by production episode tool routing."""

    def __init__(self, paths: SandboxPaths, image: str, timeout: int = 10):
        self.paths = paths
        self.image = image
        self.timeout = timeout
        self.container_name = f"veripy-{uuid.uuid4().hex}"
        self._process: subprocess.Popen[str] | None = None
        self._stderr_stream: TextIO | None = None

    def _start(self) -> subprocess.Popen[str]:
        if self._process is None or self._process.poll() is not None:
            self._stderr_stream = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
            self._process = subprocess.Popen(
                build_docker_command(
                    self.image,
                    self.paths.inputs,
                    self.paths.scratch,
                    timeout=self.timeout,
                    container_name=self.container_name,
                ),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr_stream,
                text=True,
                bufsize=1,
            )
        return self._process

    def run(self, code: str) -> ToolResult:
        started = time.monotonic()
        try:
            process = self._start()
            if process.stdin is None or process.stdout is None:
                raise RuntimeError("sandbox process pipes are unavailable")
            process.stdin.write(json.dumps({"code": code}) + "\n")
            process.stdin.flush()
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            ready = selector.select(self.timeout + 2)
            selector.close()
            if not ready:
                self.close()
                return ToolResult(
                    status=ToolStatus.TIMEOUT,
                    stderr="sandbox container timed out",
                    failure_code=FailureCode.TIMEOUT,
                    duration=time.monotonic() - started,
                )
            response = process.stdout.readline()
        except Exception as exc:
            self.close()
            return ToolResult(
                status=ToolStatus.ERROR,
                stderr=str(exc),
                failure_code=FailureCode.EXECUTION_ERROR,
                duration=time.monotonic() - started,
            )
        if not response.strip():
            diagnostics = ""
            if self._stderr_stream is not None:
                self._stderr_stream.flush()
                self._stderr_stream.seek(0)
                diagnostics = self._stderr_stream.read()[-4000:]
            self.close()
            return ToolResult(
                status=ToolStatus.ERROR,
                stderr=diagnostics or "sandbox returned no result",
                failure_code=FailureCode.EXECUTION_ERROR,
                duration=time.monotonic() - started,
            )
        try:
            return bound_tool_result(ToolResult.model_validate_json(response))
        except Exception as exc:
            return ToolResult(
                status=ToolStatus.ERROR,
                stderr=f"invalid sandbox response: {exc}",
                failure_code=FailureCode.EXECUTION_ERROR,
                duration=time.monotonic() - started,
            )

    def close(self) -> None:
        if self._process is not None:
            if self._process.stdin is not None:
                self._process.stdin.close()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=2)
        subprocess.run(
            ["docker", "rm", "--force", self.container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if self._stderr_stream is not None:
            self._stderr_stream.close()
        self._stderr_stream = None
        self._process = None
