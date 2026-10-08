"""Small immutable-artifact boundaries for staged Agentic RL and later adaptation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bindings(bindings: dict[str, str]) -> None:
    for filename, expected in bindings.items():
        if sha(Path(filename)) != expected:
            raise ValueError(f"frozen input changed: {filename}")


def artifact_sha256(root: Path) -> str:
    """Same directory identity used by actual free-agent evaluation receipts."""
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"empty model artifact: {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_merge_receipt(plan_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text())
    verify_bindings(plan["source_files"])
    verify_bindings(plan["bindings"])
    merged = Path(plan["merged_model"])
    receipt = json.loads((merged / "merge_receipt.json").read_text())
    if receipt["plan_sha256"] != sha(plan_path) or receipt["optimizer_transferred"] is not False:
        raise ValueError("merge belongs to another plan or carries optimizer state")
    verify_bindings(receipt["merged_files"])
    runtime = json.loads(Path(plan["gpu_runtime_receipt"]).read_text())
    if runtime["plan_sha256"] != sha(plan_path):
        raise ValueError("GPU runtime belongs to another plan")
    verify_bindings(
        {
            plan["gpu_environment_manifest"]: runtime["environment_sha256"],
            str(merged / "merge_receipt.json"): runtime["merge_receipt_sha256"],
        }
    )
    return plan
