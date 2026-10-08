from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from veripy_agent.contracts import FailureCode, VerifyResult


def replay_and_verify(
    steps: Iterable[Any], replay: Callable[[Any], Any], verifier: Callable[[Any], VerifyResult]
) -> VerifyResult:
    """Replay frozen steps and require verifier success; exceptions fail closed."""
    try:
        result = None
        count = 0
        for step in steps:
            result = replay(step)
            count += 1
        if count == 0:
            return VerifyResult(
                passed=False,
                score=0,
                failure_code=FailureCode.REPLAY_FAILED,
                diagnostics={"error": "empty replay"},
            )
        checked = verifier(result)
        return (
            checked
            if isinstance(checked, VerifyResult)
            else VerifyResult(
                passed=False,
                score=0,
                failure_code=FailureCode.VERIFIER_ERROR,
                diagnostics={"error": "invalid verifier result"},
            )
        )
    except Exception as e:
        return VerifyResult(
            passed=False,
            score=0,
            failure_code=FailureCode.REPLAY_FAILED,
            diagnostics={"error": str(e)},
        )
