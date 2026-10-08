#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from veripy_agent.training.sft import AdmittedTrajectory, validate_trajectory_corpus


def validate(
    config_path: Path,
    *,
    audit_path: Path | None = None,
    expected_count: int | None = None,
    expected_provider: str | None = None,
) -> dict[str, int | str]:
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_args, data_args, training_args, _, _ = get_train_args(config)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(
        template,
        model_args,
        data_args,
        training_args,
        stage="sft",
        **tokenizer_module,
    )
    train = dataset_module["train_dataset"]
    evaluation = dataset_module.get("eval_dataset")
    if len(train) == 0:
        raise RuntimeError("SFT preprocessing produced an empty train split")
    trainable_tokens = [sum(token != -100 for token in row["labels"]) for row in train]
    if min(trainable_tokens) <= 0:
        raise RuntimeError("SFT preprocessing produced an example without trainable labels")
    result: dict[str, int | str] = {
        "dataset": config["dataset"],
        "template": config["template"],
        "train_examples": len(train),
        "eval_examples": len(evaluation) if evaluation is not None else 0,
        "min_trainable_tokens": min(trainable_tokens),
        "max_sequence_tokens": max(len(row["input_ids"]) for row in train),
    }
    if audit_path is not None:
        if expected_count is None or expected_provider is None:
            raise ValueError("audit validation requires expected_count and expected_provider")
        trajectories = [
            AdmittedTrajectory.model_validate_json(line)
            for line in audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        mix = validate_trajectory_corpus(
            trajectories,
            expected_count=expected_count,
            expected_provider=expected_provider,
        )
        result["admitted_examples"] = len(trajectories)
        result["direct_success"] = mix.get("direct_success", 0)
        result["recoverable_execution_error"] = mix.get("recoverable_execution_error", 0)
        result["plan_correction"] = mix.get("plan_correction", 0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--expected-provider")
    args = parser.parse_args()
    print(
        json.dumps(
            validate(
                args.config,
                audit_path=args.audit,
                expected_count=args.expected_count,
                expected_provider=args.expected_provider,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
