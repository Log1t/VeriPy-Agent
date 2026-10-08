#!/usr/bin/env python3
import argparse
import json

from veripy_agent.evaluation import load_benchmark, result_manifest, summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--benchmark", default="local")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--manifest")
    p.add_argument("--output")
    a = p.parse_args()
    if a.manifest:
        load_benchmark(a.benchmark, a.manifest)
    values = json.loads(open(a.results, encoding="utf-8").read())
    print(
        json.dumps(
            {
                "summary": summarize(values),
                "manifest": result_manifest(
                    values, benchmark=a.benchmark, seed=a.seed, output=a.output
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
