from __future__ import annotations

import hashlib
import json
from pathlib import Path


def result_manifest(
    results, *, benchmark: str, seed: int, output: str | Path | None = None
) -> dict:
    payload = json.dumps(results, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    out = {
        "benchmark": benchmark,
        "seed": seed,
        "count": len(results),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
    if output is not None:
        Path(output).write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return out
