"""Failure-record normalization and deterministic clustering."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


def failure_cluster_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    source = str(record.get("source", "synthetic"))
    if source in {"benchmark", "ds1000", "datascibench", "dabench"}:
        raise ValueError("benchmark feedback cannot drive curriculum generation")
    family = str(record.get("family") or record.get("ability") or "unknown")
    failure = str(record.get("failure_code") or "success")
    structure = record.get("structure") or {}
    score = int(structure.get("difficulty_score", record.get("difficulty_score", 0)))
    bucket = "low" if score <= 4 else "medium" if score <= 8 else "high"
    return family, failure, bucket


def cluster_failures(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    task_ids: dict[tuple[str, str, str], set[str]] = {}
    for record in records:
        key = failure_cluster_key(record)
        counts[key] += 1
        task_ids.setdefault(key, set()).add(str(record.get("task_id", "")))
    return [
        {
            "family": family,
            "failure_code": failure,
            "difficulty_bucket": bucket,
            "count": count,
            "task_ids": sorted(task_ids[(family, failure, bucket)] - {""}),
        }
        for (family, failure, bucket), count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]
