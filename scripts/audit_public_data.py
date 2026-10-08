#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from veripy_agent.data.public_adapters import (
    ADAPTERS,
    audit_cross_split_contamination,
    audit_examples,
)
from veripy_agent.data.public_sources import admitted_training_sources, load_source_lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=Path("data/manifests/public_sources.lock.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Host-local checkout root containing one directory per locked source.",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    lock = load_source_lock(args.lock)
    report: dict[str, Any] = {"schema": "veripy.public_data.audit", "sources": {}}
    examples_for_contamination: dict[tuple[str, str], Any] = {}
    for source in admitted_training_sources(lock):
        name = source["name"]
        adapter = ADAPTERS.get(name)
        if adapter is None:
            report["sources"][name] = {"status": "no_adapter"}
            continue
        source_root = (
            (args.source_root / name).resolve()
            if args.source_root is not None
            else Path(str(source["local_path"])).resolve()
        )
        if not source_root.is_dir():
            report["sources"][name] = {"status": "missing_checkout", "path": str(source_root)}
            continue
        split_reports: dict[str, Any] = {}
        for split in source["selected_splits"]:
            examples = adapter(source_root, split)
            if args.limit:
                from itertools import islice

                examples = islice(examples, args.limit)
            materialized = list(examples)
            examples_for_contamination[(name, split)] = materialized
            split_reports[split] = audit_examples(materialized)
        report["sources"][name] = {"status": "audited", "splits": split_reports}
    contamination = audit_cross_split_contamination(examples_for_contamination)
    report["contamination"] = contamination
    if not contamination["passed"]:
        report["status"] = "rejected_contamination"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise SystemExit("public data contamination audit failed")
    report["status"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
