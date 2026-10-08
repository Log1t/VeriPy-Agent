#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from veripy_agent.data.public_sources import load_source_lock, load_source_registry


def _run(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file() and ".git" not in item.parts):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _checkout(source: dict[str, Any], root: Path) -> Path:
    destination = (root / str(source["name"])).resolve()
    if destination.parent != root.resolve():
        raise ValueError(f"invalid source destination: {destination}")
    if not destination.exists():
        _run("git", "clone", "--filter=blob:none", "--no-checkout", str(source["source"]), str(destination))
    if not (destination / ".git").is_dir():
        raise ValueError(f"source destination is not a git checkout: {destination}")
    _run("git", "fetch", "--depth", "1", "origin", str(source["revision"]), cwd=destination)
    _run("git", "checkout", "--detach", str(source["revision"]), cwd=destination)
    head = _run("git", "rev-parse", "HEAD", cwd=destination)
    if head != source["revision"]:
        raise ValueError(f"revision mismatch for {source['name']}: {head}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path("configs/data/public_sources.yaml"))
    parser.add_argument("--lock", type=Path, default=Path("data/manifests/public_sources.lock.json"))
    parser.add_argument("--output-root", type=Path, default=Path("external/public"))
    parser.add_argument("--source", action="append", default=[])
    args = parser.parse_args()

    registry = load_source_registry(args.registry)
    lock = load_source_lock(args.lock)
    selected = set(args.source)
    registry_by_name = {source["name"]: source for source in registry["sources"]}
    lock_by_name = {source["name"]: source for source in lock["sources"]}
    unknown = selected - registry_by_name.keys()
    if unknown:
        raise ValueError(f"unknown public sources: {sorted(unknown)}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    for name, source in registry_by_name.items():
        if selected and name not in selected:
            continue
        destination = _checkout(source, args.output_root)
        license_path = destination / str(source["license_file"])
        if not license_path.is_file():
            raise ValueError(f"license evidence missing for {name}: {license_path}")
        locked = lock_by_name[name]
        locked["content_sha256"] = _tree_sha256(destination)
        locked["license_sha256"] = _sha256(license_path)
        locked["local_path"] = destination.as_posix()
        locked["license_file"] = source["license_file"]
        locked["status"] = (
            "verified"
            if source["allowed_usage"] != "audit_only"
            else "excluded_pending_license"
        )

    args.lock.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
