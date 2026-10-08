from collections.abc import Callable
from math import isclose
from typing import Any

from veripy_agent.contracts import Evidence, FailureCode, Submission, VerifyResult


def _fail(code, details=None):
    return VerifyResult(passed=False, score=0, failure_code=code, diagnostics=details or {})


def _structured_score(actual: Any, expected: Any) -> float:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or not expected:
            return float(actual == expected)
        matched = sum(
            _structured_score(actual.get(key), value) for key, value in expected.items()
        )
        return matched / max(len(actual), len(expected))
    return float(actual == expected)


def verify_exact(actual: Any, expected: Any) -> VerifyResult:
    if actual == expected:
        return VerifyResult(passed=True, score=1)
    score = _structured_score(actual, expected)
    return VerifyResult(passed=False, score=score, failure_code=FailureCode.INCORRECT)


def verify_numeric(actual: Any, expected: Any, *, abs_tol=1e-6, rel_tol=1e-6) -> VerifyResult:
    try:
        ok = isclose(float(actual), float(expected), abs_tol=abs_tol, rel_tol=rel_tol)
    except (TypeError, ValueError):
        return _fail("invalid_format")
    return (
        VerifyResult(passed=True, score=1, diagnostics={"abs_tol": abs_tol, "rel_tol": rel_tol})
        if ok
        else _fail("incorrect", {"abs_tol": abs_tol, "rel_tol": rel_tol})
    )


def verify_unordered(actual, expected) -> VerifyResult:
    if not isinstance(actual, list) or not isinstance(expected, list):
        return _fail("invalid_format")
    if len(actual) != len(expected):
        return _fail("incorrect")
    remaining = list(expected)
    for item in actual:
        try:
            i = remaining.index(item)
        except ValueError:
            return _fail("incorrect")
        remaining.pop(i)
    return VerifyResult(passed=True, score=1)


def verify_table(actual, expected, *, evidence_key="evidence", ordered=False) -> VerifyResult:
    if not isinstance(actual, list) or not isinstance(expected, list):
        return _fail("invalid_format")
    if not all(isinstance(row, dict) for row in actual + expected):
        return _fail("invalid_format")
    if not expected:
        return VerifyResult(passed=not actual, score=float(not actual))
    if ordered:
        matched = list(zip(actual[: len(expected)], expected, strict=False))
    else:
        remaining = list(expected)
        matched = []
        for a in actual:
            if not remaining:
                break
            scores = [_structured_score(a, item) for item in remaining]
            i = max(range(len(scores)), key=scores.__getitem__)
            matched.append((a, remaining.pop(i)))
    matched_score = sum(_structured_score(a, e) for a, e in matched)
    score = matched_score / max(len(actual), len(expected))
    passed = len(actual) == len(expected) and score == 1.0
    if not passed:
        return VerifyResult(passed=False, score=score, failure_code=FailureCode.INCORRECT)
    return VerifyResult(
        passed=True, score=1, diagnostics={"evidence_key": evidence_key, "ordered": ordered}
    )


def verify_submission(
    submission: Submission | dict[str, Any],
    expected_answer: Any,
    evidence_recompute: Callable[[Evidence], bool],
) -> VerifyResult:
    parsed = (
        submission if isinstance(submission, Submission) else Submission.model_validate(submission)
    )
    result = (
        expected_answer(parsed.answer)
        if callable(expected_answer)
        else verify_exact(parsed.answer, expected_answer)
    )
    if not result.passed:
        return result
    try:
        evidence_ok = bool(evidence_recompute(parsed.evidence))
    except Exception as exc:
        return _fail("verifier_error", {"error": str(exc)})
    return result if evidence_ok else _fail("incorrect", {"evidence_consistent": False})
