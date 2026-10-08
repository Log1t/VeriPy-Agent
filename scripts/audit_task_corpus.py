#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from veripy_agent.data import validate_split_isolation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, nargs="?", default=Path("data/manifests/task_corpus.json"))
    parser.add_argument(
        "--lock", type=Path, default=Path("configs/data/task_corpus.lock.json")
    )
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if Path(lock["manifest"]) != args.manifest:
        raise ValueError("corpus lock points to a different manifest")
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != lock["sha256"]:
        raise ValueError("corpus manifest hash mismatch")
    corpus = json.loads(args.manifest.read_text(encoding="utf-8"))
    bundle_paths = []
    for bundle in corpus["bundles"]:
        path = Path(bundle["manifest"])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != bundle["manifest_sha256"]:
            raise ValueError(f"bundle manifest hash mismatch: {path}")
        bundle_paths.append(path)
    validate_split_isolation(bundle_paths)
    print(json.dumps({"bundles": len(bundle_paths), "status": "passed"}, sort_keys=True))


if __name__ == "__main__":
    main()
