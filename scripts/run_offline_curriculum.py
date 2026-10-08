#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from veripy_agent.data.curriculum import (
    admit_curriculum_candidates,
    build_curriculum_candidates,
    curriculum_manifest,
)
from veripy_agent.evaluation.failures import cluster_failures


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _frozen_signatures(path: Path) -> tuple[set[str], set[str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return set(value.get("content_signatures", [])), set(value.get("semantic_signatures", []))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    clusters = cluster_failures(_read_jsonl(args.failures))
    args.output_root.mkdir(parents=True, exist_ok=True)
    frozen_content, frozen_semantic = _frozen_signatures(args.validation)
    for strategy in ("adaptive", "random"):
        generated = build_curriculum_candidates(
            clusters,
            count=args.count,
            base_seed=args.seed,
            strategy=strategy,
        )
        candidates, rejected = admit_curriculum_candidates(
            generated,
            forbidden_content_signatures=frozen_content,
            forbidden_semantic_signatures=frozen_semantic,
        )
        candidates_path = args.output_root / f"{strategy}.jsonl"
        candidates_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidates),
            encoding="utf-8",
        )
        rejected_path = args.output_root / f"{strategy}.rejected.jsonl"
        rejected_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rejected),
            encoding="utf-8",
        )
        manifest = curriculum_manifest(
            candidates,
            strategy=strategy,
            corpus_sha256=_sha256(args.corpus_manifest),
            failure_records_sha256=_sha256(args.failures),
            frozen_validation_sha256=_sha256(args.validation),
        )
        manifest["candidates_path"] = candidates_path.name
        manifest["rejected_path"] = rejected_path.name
        manifest["generated_count"] = len(generated)
        manifest["accepted_count"] = len(candidates)
        manifest["rejected_count"] = len(rejected)
        (args.output_root / f"{strategy}.manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
