#!/usr/bin/env python3
import argparse
from pathlib import Path

from veripy_agent.data.generator import FAMILIES, generate_tasks, write_tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("family", choices=FAMILIES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--count", type=int, default=10)
    p.add_argument(
        "--split",
        choices=("sft_train", "rl_train", "validation", "iid_test", "ood_test"),
        default="sft_train",
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    write_tasks(
        generate_tasks(a.family, seed=a.seed, count=a.count, split=a.split),
        a.output,
        split=a.split,
        seed=a.seed,
    )


if __name__ == "__main__":
    main()
