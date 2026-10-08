from __future__ import annotations

import ast
import ctypes
import errno
import os
import resource
import sys
import traceback
from pathlib import Path

SECURITY_EXIT = 77

_LANDLOCK_CREATE_RULESET = 444
_LANDLOCK_ADD_RULE = 445
_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1

_FS_EXECUTE = 1 << 0
_FS_WRITE_FILE = 1 << 1
_FS_READ_FILE = 1 << 2
_FS_READ_DIR = 1 << 3
_FS_REMOVE_DIR = 1 << 4
_FS_REMOVE_FILE = 1 << 5
_FS_MAKE_CHAR = 1 << 6
_FS_MAKE_DIR = 1 << 7
_FS_MAKE_REG = 1 << 8
_FS_MAKE_SOCK = 1 << 9
_FS_MAKE_FIFO = 1 << 10
_FS_MAKE_BLOCK = 1 << 11
_FS_MAKE_SYM = 1 << 12
_FS_READ = _FS_EXECUTE | _FS_READ_FILE | _FS_READ_DIR
_FS_WRITE = (
    _FS_WRITE_FILE
    | _FS_REMOVE_DIR
    | _FS_REMOVE_FILE
    | _FS_MAKE_CHAR
    | _FS_MAKE_DIR
    | _FS_MAKE_REG
    | _FS_MAKE_SOCK
    | _FS_MAKE_FIFO
    | _FS_MAKE_BLOCK
    | _FS_MAKE_SYM
)
_FS_HANDLED = _FS_READ | _FS_WRITE

_SCMP_ACT_ALLOW = 0x7FFF0000
_SCMP_ACT_ERRNO = 0x00050000 | errno.EPERM


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def _status_value(name: str) -> str:
    prefix = f"{name}:"
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    raise RuntimeError(f"missing {name} in process status")


def _verify_process_identity() -> None:
    expected_uid = int(os.environ["VERIPY_SANDBOX_UID"])
    expected_gid = int(os.environ["VERIPY_SANDBOX_GID"])
    if os.getuid() != expected_uid or os.getgid() != expected_gid:
        raise RuntimeError("sandbox identity drop failed")
    if int(_status_value("CapEff"), 16) != 0 or int(_status_value("CapBnd"), 16) != 0:
        raise RuntimeError("sandbox capabilities were not fully dropped")
    if _status_value("NoNewPrivs") != "1":
        raise RuntimeError("no_new_privs is not active")


def _set_limits() -> None:
    cpu_seconds = int(os.environ["VERIPY_SANDBOX_CPU_SECONDS"])
    memory_bytes = int(os.environ["VERIPY_SANDBOX_MEMORY_BYTES"])
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _syscall(libc: ctypes.CDLL, number: int, *args: object) -> int:
    result = int(libc.syscall(number, *args))
    if result < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    return result


def _add_landlock_path(libc: ctypes.CDLL, ruleset_fd: int, path: Path, access: int) -> None:
    flags = getattr(os, "O_PATH", 0o10000000) | os.O_CLOEXEC
    parent_fd = os.open(path, flags)
    try:
        attr = _PathBeneathAttr(access, parent_fd)
        _syscall(
            libc,
            _LANDLOCK_ADD_RULE,
            ruleset_fd,
            _LANDLOCK_RULE_PATH_BENEATH,
            ctypes.byref(attr),
            0,
        )
    finally:
        os.close(parent_fd)


def _install_landlock() -> None:
    if os.uname().machine not in {"x86_64", "aarch64"}:
        raise RuntimeError("unsupported Landlock syscall architecture")
    libc = ctypes.CDLL(None, use_errno=True)
    abi = _syscall(
        libc,
        _LANDLOCK_CREATE_RULESET,
        0,
        0,
        _LANDLOCK_CREATE_RULESET_VERSION,
    )
    if abi < 1:
        raise RuntimeError("Landlock ABI 1 is required")
    ruleset_attr = _RulesetAttr(_FS_HANDLED)
    ruleset_fd = _syscall(
        libc,
        _LANDLOCK_CREATE_RULESET,
        ctypes.byref(ruleset_attr),
        ctypes.sizeof(ruleset_attr),
        0,
    )
    try:
        read_roots = {
            Path(sys.prefix),
            Path(sys.base_prefix),
            Path("/usr/lib"),
            Path("/usr/local/lib"),
            Path("/usr/share/zoneinfo"),
            Path("/lib"),
            Path("/lib64"),
            Path("/etc/ld.so.cache"),
            Path("/etc/localtime"),
            Path("/dev/null"),
            Path("/dev/urandom"),
            Path("/dev/random"),
            Path("/dev/zero"),
            Path("/proc/cpuinfo"),
            Path("/proc/meminfo"),
            Path(os.environ["VERIPY_SANDBOX_INPUTS"]),
        }
        for path in sorted(read_roots, key=str):
            if path.exists():
                access = _FS_READ if path.is_dir() else _FS_READ_FILE
                _add_landlock_path(libc, ruleset_fd, path, access)
        _add_landlock_path(
            libc, ruleset_fd, Path(os.environ["VERIPY_SANDBOX_SCRATCH"]), _FS_READ | _FS_WRITE
        )
        _syscall(libc, _LANDLOCK_RESTRICT_SELF, ruleset_fd, 0)
    finally:
        os.close(ruleset_fd)


def _install_seccomp() -> None:
    seccomp = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    seccomp.seccomp_rule_add.restype = ctypes.c_int
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_load.restype = ctypes.c_int
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]

    context = seccomp.seccomp_init(_SCMP_ACT_ALLOW)
    if not context:
        raise RuntimeError("seccomp_init failed")
    denied = {
        "accept",
        "accept4",
        "add_key",
        "bind",
        "bpf",
        "chroot",
        "connect",
        "delete_module",
        "execve",
        "execveat",
        "finit_module",
        "getpeername",
        "getsockname",
        "getsockopt",
        "init_module",
        "io_uring_enter",
        "io_uring_register",
        "io_uring_setup",
        "kexec_load",
        "keyctl",
        "listen",
        "mount",
        "name_to_handle_at",
        "open_by_handle_at",
        "perf_event_open",
        "pivot_root",
        "process_vm_readv",
        "process_vm_writev",
        "ptrace",
        "reboot",
        "recvfrom",
        "recvmmsg",
        "recvmsg",
        "request_key",
        "sendmmsg",
        "sendmsg",
        "sendto",
        "setns",
        "setsockopt",
        "shutdown",
        "socket",
        "socketpair",
        "unshare",
        "userfaultfd",
    }
    try:
        for name in sorted(denied):
            number = seccomp.seccomp_syscall_resolve_name(name.encode())
            if number < 0:
                raise RuntimeError(f"cannot resolve required seccomp syscall: {name}")
            if seccomp.seccomp_rule_add(context, _SCMP_ACT_ERRNO, number, 0) != 0:
                raise RuntimeError(f"cannot deny syscall: {name}")
        if seccomp.seccomp_load(context) != 0:
            raise RuntimeError("seccomp_load failed")
    finally:
        seccomp.seccomp_release(context)


class _LogicalPathMapper(ast.NodeTransformer):
    def __init__(self, inputs: Path, scratch: Path):
        self.roots = {"/inputs": str(inputs), "/scratch": str(scratch)}

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if not isinstance(node.value, str):
            return node
        for logical, actual in self.roots.items():
            if node.value == logical or node.value.startswith(logical + "/"):
                return ast.copy_location(
                    ast.Constant(actual + node.value[len(logical) :]), node
                )
        return node


def _compile_user_code(code: str):
    tree = ast.parse(code, filename="<veripy-sandbox>", mode="exec")
    mapped = _LogicalPathMapper(
        Path(os.environ["VERIPY_SANDBOX_INPUTS"]),
        Path(os.environ["VERIPY_SANDBOX_SCRATCH"]),
    ).visit(tree)
    ast.fix_missing_locations(mapped)
    return compile(mapped, "<veripy-sandbox>", "exec")


def main() -> int:
    try:
        code = sys.stdin.read()
        compiled = _compile_user_code(code)
        _verify_process_identity()
        _set_limits()
        _install_landlock()
        _install_seccomp()
        namespace = {"__name__": "__main__"}
        exec(compiled, namespace, namespace)
        return 0
    except PermissionError:
        traceback.print_exc()
        return SECURITY_EXIT
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
