"""Fail-closed acceptance gate for offline curriculum candidates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

IDENTITY_FIELDS = ("model_sha256", "environment_sha256", "frozen_validation_sha256")


def evaluate_curriculum_acceptance(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    minimum_success_gain: float = 0.01,
) -> dict[str, Any]:
    missing = [key for key in IDENTITY_FIELDS if not baseline.get(key) or not candidate.get(key)]
    if missing:
        raise ValueError(f"acceptance identity fields missing: {missing}")
    mismatched = [key for key in IDENTITY_FIELDS if baseline[key] != candidate[key]]
    if mismatched:
        raise ValueError(f"acceptance identity mismatch: {mismatched}")
    checks = {
        "success_gain": float(candidate["task_success"]) - float(baseline["task_success"])
        >= minimum_success_gain,
        "format_not_regressed": float(candidate["format_valid_rate"])
        >= float(baseline["format_valid_rate"]),
        "security_not_regressed": int(candidate.get("security_violations", 0))
        <= int(baseline.get("security_violations", 0)),
        "benchmark_feedback_unused": not bool(candidate.get("benchmark_feedback_used", False)),
    }
    return {"passed": all(checks.values()), "checks": checks}
