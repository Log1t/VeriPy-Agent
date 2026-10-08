"""Fail-closed registry for audited public datasets.

The registry records provenance and allowed usage. It deliberately does not
download or admit a source when content and license hashes are missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_source_registry(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "veripy.public_sources":
        raise ValueError("invalid public source registry schema")
    sources = value.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("public source registry must contain sources")
    names: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("public source entry must be an object")
        required = {"name", "kind", "source", "revision", "license", "allowed_usage", "selected_splits"}
        if not required.issubset(source):
            raise ValueError(f"public source missing fields: {sorted(required - set(source))}")
        name = str(source["name"])
        if name in names:
            raise ValueError(f"duplicate public source: {name}")
        names.add(name)
    return value


def load_source_lock(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != "veripy.public_sources.lock":
        raise ValueError("invalid public source lock schema")
    for source in value.get("sources", []):
        if source.get("status") not in {
            "source_pinned_content_not_downloaded",
            "evaluation_only_not_downloaded",
            "excluded_pending_license",
            "verified",
        }:
            raise ValueError(f"unknown source lock status: {source.get('status')}")
        if source.get("status") == "verified" and not source.get("content_sha256"):
            raise ValueError(f"verified source has no content hash: {source.get('name')}")
    return value


def admitted_training_sources(lock: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only content- and license-hashed sources allowed for training."""

    admitted: list[dict[str, Any]] = []
    for source in lock.get("sources", []):
        if source.get("allowed_usage") not in {"train_dev_only", "train_and_dev"}:
            continue
        if source.get("status") != "verified":
            continue
        if not source.get("content_sha256") or not source.get("license_sha256"):
            continue
        admitted.append(source)
    return admitted
