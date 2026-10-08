#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from veripy_agent.verl_adapter import build_rollout_rows_from_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/manifests/task_corpus.json"))
    parser.add_argument(
        "--corpus-lock", type=Path, default=Path("configs/data/task_corpus.lock.json")
    )
    parser.add_argument("--split", choices=("rl_train", "validation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/generated"),
        help="Root used to encode portable, dataset-relative episode paths.",
    )
    args = parser.parse_args()
    lock = json.loads(args.corpus_lock.read_text(encoding="utf-8"))
    if Path(lock["manifest"]) != args.corpus:
        raise ValueError("corpus lock points to a different manifest")
    rows = build_rollout_rows_from_corpus(
        args.corpus,
        args.split,
        expected_sha256=lock["sha256"],
        dataset_root=args.dataset_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    print(json.dumps({"output": str(args.output), "rows": len(rows), "split": args.split}))


if __name__ == "__main__":
    main()
