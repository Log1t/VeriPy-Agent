"""Local-only benchmark adapters."""

import json
from pathlib import Path


def load_local_manifest(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"local manifest required: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("invalid local manifest") from exc
    if not isinstance(data, dict) or not data.get("sha256") or "task_count" not in data:
        raise ValueError("manifest missing required fields")
    return data


def load_benchmark(name: str, manifest_path):
    if name == "local":
        return {"name": name, "manifest": load_local_manifest(manifest_path)}
    if name not in {"DS-1000", "DataSciBench"}:
        raise ValueError(f"unsupported benchmark: {name}")
    return {"name": name, "manifest": load_local_manifest(manifest_path)}
