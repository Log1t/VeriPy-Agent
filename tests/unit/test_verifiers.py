from veripy_agent.verifiers import (
    verify_exact,
    verify_numeric,
    verify_submission,
    verify_table,
    verify_unordered,
)


def test_verifier_modes():
    assert verify_exact("a", "a").passed
    assert not verify_exact("a", "b").passed
    assert verify_numeric(1.0, 1.001, abs_tol=0.01).passed
    assert verify_unordered(["a", "b"], ["b", "a"]).passed
    assert verify_table([{"a": 1}], [{"a": 1}]).passed


def test_structured_verifiers_report_partial_credit_without_oracle_diagnostics():
    exact = verify_exact({"a": 1, "b": 0}, {"a": 1, "b": 2})
    assert not exact.passed
    assert exact.score == 0.5
    assert "expected" not in exact.diagnostics

    exact_with_extra = verify_exact(
        {"a": 1, "b": 2, "extra": 3}, {"a": 1, "b": 2}
    )
    assert not exact_with_extra.passed
    assert exact_with_extra.score == 2 / 3

    table = verify_table([{"a": 1, "b": 0}], [{"a": 1, "b": 2}], ordered=True)
    assert not table.passed
    assert table.score == 0.5

    extra = verify_table([{"a": 1}, {"a": 2}], [{"a": 1}], ordered=True)
    assert not extra.passed
    assert extra.score == 0.5

    numeric = verify_numeric(1, 2)
    assert numeric.score == 0
    assert "expected" not in numeric.diagnostics


def test_evidence_comparison():
    r = verify_table(
        [{"claim": "x", "evidence": ["doc:1"]}], [{"claim": "x", "evidence": ["doc:1"]}]
    )
    assert r.passed
    assert not verify_table(
        [{"claim": "wrong", "evidence": ["doc:1"]}], [{"claim": "x", "evidence": ["doc:1"]}]
    ).passed


def test_table_default_unordered_and_order_can_be_required():
    rows = [{"a": 1}, {"a": 2}]
    assert verify_table(rows[::-1], rows).passed
    assert not verify_table(rows[::-1], rows, ordered=True).passed


def test_unordered_is_multiset_semantics():
    assert not verify_unordered(["a", "a"], ["a"]).passed
    assert verify_unordered(["a", "b", "a"], ["a", "a", "b"]).passed


def test_submission_checks_evidence_only_after_answer():
    calls = []

    def recompute(evidence):
        calls.append(evidence)
        return True

    submission = {
        "answer": 1,
        "evidence": {"columns": ["value"], "filters": [], "intermediate_values": {}},
    }
    assert verify_submission(submission, 1, recompute).passed
    assert calls[0].columns == ["value"]
    calls.clear()
    assert not verify_submission({**submission, "answer": 2}, 1, recompute).passed
    assert calls == []
