#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from veripy_agent.data import FAMILIES, generate_tasks, validate_split_isolation, write_tasks


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _family_counts(total: int) -> dict[str, int]:
    base, remainder = divmod(total, len(FAMILIES))
    return {
        family: base + int(index < remainder)
        for index, family in enumerate(FAMILIES)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data/task_corpus.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"])
    manifest_path = Path(config["manifest_path"])
    lock_path = Path(config["lock_path"])
    base_seed = int(config["base_seed"])
    bundle_paths = []
    bundles = []
    for split_index, (split, total) in enumerate(config["splits"].items()):
        for family_index, (family, count) in enumerate(_family_counts(int(total)).items()):
            seed = base_seed + split_index * 100 + family_index
            output = output_root / split / f"{family}.jsonl"
            output.parent.mkdir(parents=True, exist_ok=True)
            write_tasks(
                generate_tasks(family, seed=seed, count=count, split=split),
                output,
                split=split,
                seed=seed,
            )
            bundle_path = output.with_suffix(output.suffix + ".manifest.json")
            bundle_paths.append(bundle_path)
            bundles.append(
                {
                    "split": split,
                    "family": family,
                    "count": count,
                    "seed": seed,
                    "manifest": bundle_path.as_posix(),
                    "manifest_sha256": _sha256(bundle_path),
                }
            )
    validate_split_isolation(bundle_paths)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "generator": "veripy_agent.data.generate_tasks",
                "config": args.config.as_posix(),
                "config_sha256": _sha256(args.config),
                "base_seed": base_seed,
                "split_totals": config["splits"],
                "bundles": bundles,
                "license": "self_generated",
                "self_generated": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {"manifest": manifest_path.as_posix(), "sha256": _sha256(manifest_path)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
