import json

from veripy_agent.contracts import (
    Submission,
    TaskSpec,
    ToolResult,
    VerifyResult,
    export_json_schemas,
)


def test_contracts_validate_and_schema(tmp_path):
    t = TaskSpec(
        task_id="t1", family="math", instruction="do it", answer_schema={}, verifier_spec={}
    )
    tr = ToolResult(status="ok")
    sub = Submission(answer=2, evidence={})
    vr = VerifyResult(passed=True, score=1)
    assert t.task_id == "t1" and tr.status == "ok" and sub.answer == 2 and vr.passed
    export_json_schemas(tmp_path)
    assert (tmp_path / "TaskSpec.json").exists()
    assert json.loads((tmp_path / "Submission.json").read_text())["title"] == "Submission"


def test_contracts_reject_bad_score():
    import pytest

    with pytest.raises(Exception):
        VerifyResult(passed=False, score=1.2)


def test_contracts_use_enums_and_complete_fields():
    from veripy_agent.contracts import ToolStatus

    assert ToolStatus.OK.value == "ok"
    assert set(TaskSpec.model_fields) >= {
        "task_id",
        "family",
        "instruction",
        "answer_schema",
        "verifier_spec",
    }


def test_schema_is_stable(tmp_path):
    first = export_json_schemas(tmp_path)
    snapshots = {k: p.read_text() for k, p in first.items()}
    second = export_json_schemas(tmp_path)
    assert snapshots == {k: p.read_text() for k, p in second.items()}
