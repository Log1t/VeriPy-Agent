import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from veripy_agent.data import teacher_curriculum as teacher


def context():
    return {
        "source_split": "rl_train",
        "source_launch_sha256": "a" * 64,
        "candidate_count": 32,
        "clusters": [{"cluster_id": "cluster-000", "allowed_designs": [{"generator_index": 1}]}],
    }


def decision(count=32, index=1):
    return {
        "diagnosis": "Actual training failures need a bounded curriculum.",
        "allocations": [
            {
                "cluster_id": "cluster-000",
                "count": count,
                "generator_index": index,
                "reason": "Use the supported real input design.",
            }
        ],
        "next_round_focus": "Learn from new training failures only.",
    }


class Response:
    def __init__(self, content, finish="stop"):
        self.raw = json.dumps(
            {
                "model": "test-teacher",
                "usage": {},
                "choices": [{"message": {"content": json.dumps(content)}, "finish_reason": finish}],
            }
        ).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def read(self):
        return self.raw


def mock_requests(monkeypatch, sequence):
    config = SimpleNamespace(
        model="test-teacher",
        provider="test",
        base_url="https://teacher.invalid/v1",
        resolve_api_key=lambda _env: "private-test-key",
    )
    monkeypatch.setattr(teacher, "load_teacher_secret_file", lambda _p: (config, {}))
    monkeypatch.setattr(teacher.time, "sleep", lambda _n: None)
    calls = []
    items = iter(sequence)

    def urlopen(request, timeout):
        assert timeout == 120
        calls.append(json.loads(request.data))
        item = next(items)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(teacher.urllib.request, "urlopen", urlopen)
    return calls


@pytest.mark.parametrize("total", [27, 38, 31])
def test_rejected_quota_is_sent_to_teacher_and_never_rebalanced(tmp_path, monkeypatch, total):
    # 38 must be composed of individually legal allocations, as in the captured failure.
    bad = decision(total) if total <= 32 else decision(19)
    if total > 32:
        bad["allocations"].append(dict(bad["allocations"][0]))
    calls = mock_requests(monkeypatch, [Response(bad), Response(decision())])
    result = teacher.request_curriculum(
        context(), secret=Path("not-read"), output=tmp_path / "teacher"
    )
    assert sum(a.count for a in result.allocations) == 32
    retry = calls[1]["messages"][-1]["content"]
    assert "quota_mismatch" in retry and str(total) in retry and "32" in retry
    assert calls[0] != calls[1]
    assert calls[1]["messages"][-2]["role"] == "assistant"
    assert json.loads(calls[1]["messages"][-2]["content"]) == bad
    saved = json.loads((tmp_path / "teacher/rejection_1.json").read_text())
    assert saved["code"] == "quota_mismatch"
    assert saved["details"]["actual_total"] == total
    assert (tmp_path / "teacher/request_attempt_2.json").exists()


def test_unavailable_design_retry_is_actionable(tmp_path, monkeypatch):
    calls = mock_requests(monkeypatch, [Response(decision(index=2)), Response(decision())])
    teacher.request_curriculum(context(), secret=Path("not-read"), output=tmp_path / "teacher")
    assert "unavailable_design" in calls[1]["messages"][-1]["content"]
    assert "generator_index" in calls[1]["messages"][-1]["content"]


@pytest.mark.parametrize("change", ["extra", "string_count", "truncated"])
def test_schema_and_terminal_failures_remain_rejected(tmp_path, monkeypatch, change):
    bad = decision()
    if change == "extra":
        bad["hidden_answer"] = "not allowed"
    if change == "string_count":
        bad["allocations"][0]["count"] = "32"
    first = Response(bad, finish="length" if change == "truncated" else "stop")
    calls = mock_requests(monkeypatch, [first, Response(decision())])
    teacher.request_curriculum(context(), secret=Path("not-read"), output=tmp_path / "teacher")
    assert (
        "schema_rejected" in calls[1]["messages"][-1]["content"]
        or "nonterminal_response" in calls[1]["messages"][-1]["content"]
    )


def test_transport_error_never_logs_sensitive_exception(tmp_path, monkeypatch):
    calls = mock_requests(
        monkeypatch, [urllib.error.URLError("private-test-key URL-body"), Response(decision())]
    )
    teacher.request_curriculum(context(), secret=Path("not-read"), output=tmp_path / "teacher")
    artifacts = "\n".join(p.read_text() for p in (tmp_path / "teacher").glob("*.json"))
    assert "private-test-key" not in artifacts and "URL-body" not in artifacts
    assert calls[1]["messages"] == calls[0]["messages"]
    assert (
        json.loads((tmp_path / "teacher/rejection_1.json").read_text())["code"] == "transport_error"
    )


def test_bad_context_rejected_before_network(tmp_path, monkeypatch):
    calls = mock_requests(monkeypatch, [])
    value = context()
    value["source_split"] = "validation"
    with pytest.raises(ValueError, match="rl_train"):
        teacher.request_curriculum(value, secret=Path("not-read"), output=tmp_path / "teacher")
    assert not calls and not (tmp_path / "teacher").exists()


def test_three_rejections_preserve_receipts_and_no_decision(tmp_path, monkeypatch):
    mock_requests(monkeypatch, [Response(decision(31))] * 3)
    with pytest.raises(RuntimeError, match="teacher curriculum failed"):
        teacher.request_curriculum(context(), secret=Path("not-read"), output=tmp_path / "teacher")
    assert len(list((tmp_path / "teacher").glob("rejection_*.json"))) == 3
    assert not (tmp_path / "teacher/decision.json").exists()
    assert len(list((tmp_path / "teacher").glob("request_attempt_*.json"))) == 3


def test_existing_output_is_not_overwritten(tmp_path, monkeypatch):
    calls = mock_requests(monkeypatch, [])
    output = tmp_path / "teacher"
    output.mkdir()
    (output / "historical.json").write_text("historical")
    with pytest.raises(FileExistsError):
        teacher.request_curriculum(context(), secret=Path("not-read"), output=output)
    assert not calls and (output / "historical.json").read_text() == "historical"


@pytest.mark.parametrize("content", [[], {}, None])
def test_invalid_response_envelope_is_not_replayed_as_an_assistant_message(
    tmp_path, monkeypatch, content
):
    bad = Response(decision())
    envelope = json.loads(bad.raw)
    envelope["choices"][0]["message"]["content"] = content
    bad.raw = json.dumps(envelope).encode()
    calls = mock_requests(monkeypatch, [bad, Response(decision())])
    teacher.request_curriculum(context(), secret=Path("not-read"), output=tmp_path / "teacher")
    assert calls[1]["messages"] == calls[0]["messages"]
    rejection = json.loads((tmp_path / "teacher/rejection_1.json").read_text())
    assert rejection["code"] == "response_envelope_invalid"
