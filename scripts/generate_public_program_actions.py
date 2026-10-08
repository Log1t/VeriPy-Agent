#!/usr/bin/env python3
"""Convert a small, explicitly labelled subset of public FinQA programs to actions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from veripy_agent.training.public_replay import load_public_bundle

_CALL = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<args>.*)\)$")


def _split_args(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _number(value: str) -> float:
    token = value.strip()
    if token.startswith("const_"):
        token = token.removeprefix("const_")
    if token.endswith("%"):
        return float(token[:-1]) / 100.0
    return float(token)


def _table_reduce_local(table: list[list[str]], label: str, operation: str) -> float:
    wanted = str(label).strip().lower()
    for row in table:
        if row and row[0].strip().lower() == wanted:
            values: list[float] = []
            for raw in row[1:]:
                text = str(raw).strip().replace(",", "")
                percent = "%" in text
                match = re.search(r"-?\d+(?:\.\d+)?", text)
                if match is None:
                    raise ValueError(f"non-numeric table cell: {raw!r}")
                value = float(match.group(0))
                values.append(value / 100.0 if percent else value)
            if not values:
                raise ValueError(f"table row has no numeric cells: {label!r}")
            if operation == "average":
                return sum(values) / len(values)
            if operation == "sum":
                return sum(values)
            if operation == "min":
                return min(values)
            if operation == "max":
                return max(values)
            raise ValueError(f"unsupported table operation: {operation}")
    raise ValueError(f"table row not found: {label!r}")


def finqa_numeric_program(
    program: str, table: list[list[str]] | None = None
) -> tuple[float | bool | str, str]:
    """Evaluate and render supported numeric/table FinQA programs.

    Supported table reductions are recomputed from /inputs/table.csv; unsupported selectors and programs are rejected rather than
    silently approximated.  The returned Python source recomputes the value in
    the sandbox and does not contain a hidden expected answer field.
    """

    calls = re.findall(r"[A-Za-z_][A-Za-z0-9_]*\([^)]*\)", program)
    if not calls or ",".join(calls).replace(",", "") == "":
        raise ValueError("program contains no supported calls")
    results: list[float | bool | str] = []
    lines = [
        "import csv",
        "import re",
        "results = []",
        "def _cell_number(raw):",
        "    text = str(raw).strip().replace(',', '')",
        "    percent = '%' in text",
        "    match = re.search(r'-?\\d+(?:\\.\\d+)?', text)",
        "    if match is None:",
        "        raise ValueError(f'non-numeric table cell: {raw!r}')",
        "    value = float(match.group(0))",
        "    return value / 100.0 if percent else value",
        "def _table_values(label):",
        "    wanted = str(label).strip().lower()",
        "    with open('/inputs/table.csv', newline='', encoding='utf-8') as stream:",
        "        for row in csv.reader(stream):",
        "            if row and row[0].strip().lower() == wanted:",
        "                return [_cell_number(cell) for cell in row[1:] if str(cell).strip()]",
        "    raise ValueError(f'table row not found: {label!r}')",
        "def _table_reduce(label, operation):",
        "    values = _table_values(label)",
        "    if not values:",
        "        raise ValueError(f'table row has no numeric cells: {label!r}')",
        "    if operation == 'average':",
        "        return sum(values) / len(values)",
        "    if operation == 'sum':",
        "        return sum(values)",
        "    if operation == 'min':",
        "        return min(values)",
        "    if operation == 'max':",
        "        return max(values)",
        "    raise ValueError(f'unsupported table operation: {operation}')",
    ]
    for _index, call in enumerate(calls):
        match = _CALL.match(call.strip())
        if match is None:
            raise ValueError(f"unsupported program call: {call}")
        name = match.group("name")
        args = _split_args(match.group("args"))
        supported_arithmetic = {"add", "subtract", "multiply", "divide", "greater", "exp"}
        supported_table = {"table_average", "table_sum", "table_min", "table_max"}
        if name not in supported_arithmetic | supported_table:
            raise ValueError(f"unsupported FinQA operation: {name}")
        if len(args) != 2:
            raise ValueError(f"operation requires two arguments: {call}")
        if name in supported_table:
            if args[1].lower() != "none":
                raise ValueError(f"unsupported table selector: {call}")
            label = args[0]
            operation = name.removeprefix("table_")
            if table is None:
                raise ValueError("table input is required for table operation")
            results.append(_table_reduce_local(table, label, operation))
            lines.append(f"results.append(_table_reduce({label!r}, {operation!r}))")
            continue
        expressions: list[str] = []
        values: list[Any] = []
        for arg in args:
            if arg.startswith("#"):
                reference = int(arg[1:])
                if reference < 0 or reference >= len(results):
                    raise ValueError(f"invalid intermediate reference: {arg}")
                values.append(results[reference])
                expressions.append(f"results[{reference}]")
            else:
                values.append(_number(arg))
                expressions.append(repr(_number(arg)))
        left, right = values
        if name == "add":
            result = left + right
            operator = "+"
        elif name == "subtract":
            result = left - right
            operator = "-"
        elif name == "multiply":
            result = left * right
            operator = "*"
        elif name == "divide":
            if right == 0:
                raise ValueError("division by zero")
            result = left / right
            operator = "/"
        elif name == "exp":
            result = left**right
            operator = "**"
        else:
            result = "yes" if left > right else "no"
            results.append(result)
            lines.append(
                f"results.append('yes' if {expressions[0]} > {expressions[1]} else 'no')"
            )
            continue
        results.append(result)
        lines.append(f"results.append({expressions[0]} {operator} {expressions[1]})")
    lines.append("answer = results[-1]")
    lines.append("print(json.dumps(answer))")
    return results[-1], "import json\n" + "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    if freeze.get("schema") != "veripy.public_corpus.freeze" or freeze.get("status") != "passed":
        raise ValueError("public freeze manifest is not admitted")
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for source in freeze["sources"]:
        if source["name"] != "finqa":
            continue
        for split in source["splits"].values():
            bundle_root = (args.freeze_manifest.parent / split["bundle"]).parent
            for task in load_public_bundle(bundle_root):
                if args.limit and len(records) >= args.limit:
                    break
                try:
                    if task.program is None:
                        raise ValueError("missing public program")
                    table_path = task.input_root / "table.csv"
                    with table_path.open(encoding="utf-8", newline="") as stream:
                        table = [[str(cell) for cell in row] for row in csv.reader(stream)]
                    answer, code = finqa_numeric_program(task.program, table)
                    provenance = {
                        "provider": "public_program_replay",
                        "model": "public_annotation",
                        "config_sha256": hashlib.sha256(task.program.encode()).hexdigest(),
                        "request_sha256": hashlib.sha256(task.program.encode()).hexdigest(),
                        "response_sha256": hashlib.sha256(code.encode()).hexdigest(),
                    }
                    records.append(
                        {
                            "task_id": task.task.task_id,
                            "actions": [
                                {"tool": "execute_python", "arguments": {"code": code}},
                                {
                                    "tool": "submit_answer",
                                    "arguments": {
                                        "answer": answer,
                                        "evidence": {
                                            "columns": ["table.csv"],
                                            "filters": [],
                                            "intermediate_values": {"answer": answer},
                                        },
                                    },
                                },
                            ],
                            "provenance": provenance,
                        }
                    )
                except Exception as exc:
                    rejected.append({"task_id": task.task.task_id, "error": str(exc)})
            if args.limit and len(records) >= args.limit:
                break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8"
    )
    args.output.with_suffix(".rejected.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rejected), encoding="utf-8"
    )
    print(json.dumps({"accepted_candidates": len(records), "rejected_candidates": len(rejected)}))


if __name__ == "__main__":
    main()
