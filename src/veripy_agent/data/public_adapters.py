"""Adapters and quality audits for pinned public table-reasoning datasets."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from veripy_agent.contracts import TaskSpec


@dataclass(frozen=True)
class PublicExample:
    source: str
    split: str
    source_id: str
    instruction: str
    table: list[list[str]]
    context: list[str]
    answer: Any
    answer_type: str
    program: str | None = None

    @property
    def task_id(self) -> str:
        digest = hashlib.sha256(
            f"{self.source}|{self.split}|{self.source_id}".encode()
        ).hexdigest()[:20]
        return f"public-{self.source}-{digest}"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_finqa(root: str | Path, split: str) -> Iterable[PublicExample]:
    path = Path(root) / "dataset" / f"{split}.json"
    for index, record in enumerate(_json(path)):
        qa = record["qa"]
        source_id = str(record.get("id") or f"{record.get('filename', 'unknown')}:{index}")
        answer = qa["exe_ans"]
        yield PublicExample(
            source="finqa",
            split=split,
            source_id=source_id,
            instruction=str(qa["question"]),
            table=[[str(cell) for cell in row] for row in record.get("table", record.get("table_ori", []))],
            context=[str(item) for item in record.get("pre_text", []) + record.get("post_text", [])],
            answer=answer,
            answer_type=type(answer).__name__,
            program=str(qa.get("program") or "") or None,
        )


def iter_tatqa(root: str | Path, split: str) -> Iterable[PublicExample]:
    path = Path(root) / "dataset_raw" / f"tatqa_dataset_{split}.json"
    for record in _json(path):
        table = [[str(cell) for cell in row] for row in record["table"]["table"]]
        context = [str(item["text"]) for item in record.get("paragraphs", [])]
        for question in record.get("questions", []):
            answer = question.get("answer")
            yield PublicExample(
                source="tatqa",
                split=split,
                source_id=str(question["uid"]),
                instruction=str(question["question"]),
                table=table,
                context=context,
                answer=answer,
                answer_type=str(question.get("answer_type") or type(answer).__name__),
                program=str(question.get("derivation") or "") or None,
            )


def _safe_table_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"table path escapes dataset root: {relative}")
    if not path.is_file():
        raise ValueError(f"table path does not exist: {relative}")
    return path


def iter_wikitablequestions(root: str | Path, split: str) -> Iterable[PublicExample]:
    root = Path(root)
    path = root / "data" / f"{split}.tsv"
    with path.open(encoding="utf-8", newline="") as stream:
        for record in csv.DictReader(stream, delimiter="\t"):
            table_path = _safe_table_path(root, str(record["context"]))
            with table_path.open(encoding="utf-8", newline="") as table_stream:
                table = [[str(cell) for cell in row] for row in csv.reader(table_stream)]
            yield PublicExample(
                source="wikitablequestions",
                split=split,
                source_id=str(record["id"]),
                instruction=str(record["utterance"]),
                table=table,
                context=[],
                answer=str(record["targetValue"]),
                answer_type="target_value",
            )


ADAPTERS = {
    "finqa": iter_finqa,
    "tatqa": iter_tatqa,
    "wikitablequestions": iter_wikitablequestions,
}


def audit_examples(examples: Iterable[PublicExample]) -> dict[str, Any]:
    rows = list(examples)
    normalized = [" ".join(item.instruction.lower().split()) for item in rows]
    table_shapes = [
        (max(0, len(item.table) - 1), max((len(row) for row in item.table), default=0))
        for item in rows
    ]
    return {
        "sample_count": len(rows),
        "unique_source_ids": len({item.source_id for item in rows}),
        "duplicate_instruction_count": len(normalized) - len(set(normalized)),
        "answer_types": dict(sorted(Counter(item.answer_type for item in rows).items())),
        "program_coverage": (
            sum(item.program is not None for item in rows) / len(rows) if rows else 0.0
        ),
        "nonempty_answer_rate": (
            sum(item.answer not in (None, "", []) for item in rows) / len(rows) if rows else 0.0
        ),
        "table_row_min": min((shape[0] for shape in table_shapes), default=0),
        "table_row_max": max((shape[0] for shape in table_shapes), default=0),
        "table_column_min": min((shape[1] for shape in table_shapes), default=0),
        "table_column_max": max((shape[1] for shape in table_shapes), default=0),
        "context_coverage": (
            sum(bool(item.context) for item in rows) / len(rows) if rows else 0.0
        ),
    }


def audit_cross_split_contamination(
    examples_by_source_split: dict[tuple[str, str], Iterable[PublicExample]],
) -> dict[str, Any]:
    """Audit identity/content overlap before public data is admitted.

    The audit is independent of answer/program fields and therefore catches
    leakage that can survive source-specific IDs, such as the same normalized
    question or table appearing in two splits or sources.
    """

    def normalized_instruction(value: str) -> str:
        return " ".join(value.lower().split())

    def table_digest(table: list[list[str]]) -> str:
        raw = json.dumps(table, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    records: list[dict[str, str]] = []
    for (source, split), examples in sorted(examples_by_source_split.items()):
        for example in examples:
            records.append(
                {
                    "source": source,
                    "split": split,
                    "source_id": example.source_id,
                    "instruction": normalized_instruction(example.instruction),
                    "table": table_digest(example.table),
                }
            )

    def overlaps(field: str, *, require_cross_location: bool) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, str]]] = {}
        for record in records:
            grouped.setdefault(record[field], []).append(record)
        result: list[dict[str, Any]] = []
        for value, matches in sorted(grouped.items()):
            locations = sorted({f"{item['source']}:{item['split']}" for item in matches})
            if len(matches) > 1 and (
                not require_cross_location or len(locations) > 1
            ):
                result.append(
                    {
                        "value_sha256": hashlib.sha256(value.encode()).hexdigest(),
                        "locations": locations,
                        "count": len(matches),
                    }
                )
        return result

    source_ids = overlaps("source_id", require_cross_location=False)
    instructions = overlaps("instruction", require_cross_location=True)
    tables = overlaps("table", require_cross_location=True)
    return {
        "record_count": len(records),
        "source_split_count": len(examples_by_source_split),
        "source_id_overlap": source_ids,
        "instruction_overlap": instructions,
        "table_overlap": tables,
        "passed": not (source_ids or instructions or tables),
    }


def public_task_spec(example: PublicExample, files: list[str]) -> TaskSpec:
    numeric = example.answer_type in {"float", "arithmetic", "number", "numeric"}
    return TaskSpec(
        task_id=example.task_id,
        family=f"public_{example.source}",
        instruction=example.instruction,
        files=files,
        split=example.split,
        difficulty=None,
        answer_schema={"source_answer_type": example.answer_type},
        verifier_spec={
            "kind": "private_oracle",
            "source": example.source,
            "source_id": example.source_id,
            **(
                {"answer_kind": "numeric", "abs_tol": 1e-4, "rel_tol": 1e-4}
                if numeric
                else {"answer_kind": "exact"}
            ),
        },
    )


def materialize_public_examples(
    examples: Iterable[PublicExample], output_root: str | Path
) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    public_records: list[dict[str, Any]] = []
    oracle_records: list[dict[str, Any]] = []
    file_hashes: dict[str, str] = {}
    for example in examples:
        task_root = output_root / example.task_id
        task_root.mkdir(parents=True, exist_ok=True)
        table_path = task_root / "table.csv"
        with table_path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerows(example.table)
        files = [table_path.relative_to(task_root).as_posix()]
        if example.context:
            context_path = task_root / "context.json"
            context_path.write_text(
                json.dumps(example.context, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            files.append(context_path.relative_to(task_root).as_posix())
        for relative in files:
            bundle_relative = (Path(example.task_id) / relative).as_posix()
            file_hashes[bundle_relative] = hashlib.sha256(
                (task_root / relative).read_bytes()
            ).hexdigest()
        spec = public_task_spec(example, files)
        public_records.append(
            {
                "task_id": example.task_id,
                "source": example.source,
                "split": example.split,
                "input_root": example.task_id,
                "task_spec": spec.model_dump(mode="json"),
            }
        )
        oracle_records.append(
            {
                "task_id": example.task_id,
                "answer": example.answer,
                "answer_type": example.answer_type,
                "program": example.program,
            }
        )
    public_path = output_root / "tasks.jsonl"
    oracle_path = output_root / "tasks.oracle.jsonl"
    public_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in public_records),
        encoding="utf-8",
    )
    oracle_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in oracle_records),
        encoding="utf-8",
    )
    file_hashes[public_path.name] = hashlib.sha256(public_path.read_bytes()).hexdigest()
    file_hashes[oracle_path.name] = hashlib.sha256(oracle_path.read_bytes()).hexdigest()
    manifest = {
        "schema": "veripy.public_task_bundle",
        "task_count": len(public_records),
        "task_ids": [row["task_id"] for row in public_records],
        "public_records": public_path.name,
        "private_oracle": oracle_path.name,
        "files": file_hashes,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
