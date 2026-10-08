import json
from pathlib import Path

from veripy_agent.contracts import Submission, TaskSpec, VerifyResult


def schemas():
    return {x.__name__: x.model_json_schema() for x in (TaskSpec, Submission, VerifyResult)}


def export(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(schemas(), indent=2) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/manifests/contracts.schema.json"),
    )
    args = parser.parse_args()
    print(export(args.output))
