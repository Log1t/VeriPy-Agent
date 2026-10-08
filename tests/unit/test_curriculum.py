import pytest

from veripy_agent.data.curriculum import build_curriculum_candidates, curriculum_manifest
from veripy_agent.evaluation.acceptance import evaluate_curriculum_acceptance
from veripy_agent.evaluation.failures import cluster_failures, failure_cluster_key


def test_failure_clustering_is_deterministic_and_rejects_benchmark_feedback():
    rows = [
        {"task_id": "a", "family": "missing_values", "failure_code": "timeout", "structure": {"difficulty_score": 6}},
        {"task_id": "b", "family": "missing_values", "failure_code": "timeout", "structure": {"difficulty_score": 7}},
        {"task_id": "c", "family": "filter_aggregate", "failure_code": "incorrect", "structure": {"difficulty_score": 3}},
    ]
    clusters = cluster_failures(reversed(rows))
    assert clusters[0]["count"] == 2
    assert clusters[0]["task_ids"] == ["a", "b"]
    with pytest.raises(ValueError, match="benchmark feedback"):
        failure_cluster_key({"source": "ds1000"})


def test_adaptive_and_random_candidates_are_equal_size_and_deterministic():
    clusters = [{"family": "missing_values", "failure_code": "timeout", "count": 3}]
    adaptive = build_curriculum_candidates(clusters, count=4, base_seed=9, strategy="adaptive")
    repeated = build_curriculum_candidates(clusters, count=4, base_seed=9, strategy="adaptive")
    random_baseline = build_curriculum_candidates(clusters, count=4, base_seed=9, strategy="random")
    assert adaptive == repeated
    assert len(adaptive) == len(random_baseline) == 4
    assert {row["task"]["family"] for row in adaptive} == {"missing_values"}
    manifest = curriculum_manifest(
        adaptive,
        strategy="adaptive",
        corpus_sha256="a" * 64,
        failure_records_sha256="b" * 64,
        frozen_validation_sha256="c" * 64,
    )
    assert manifest["benchmark_feedback_used"] is False


def test_acceptance_requires_identity_and_success_without_format_regression():
    baseline = {
        "model_sha256": "a",
        "environment_sha256": "b",
        "frozen_validation_sha256": "c",
        "task_success": 0.20,
        "format_valid_rate": 0.90,
        "security_violations": 0,
    }
    candidate = {**baseline, "task_success": 0.22, "format_valid_rate": 0.91}
    assert evaluate_curriculum_acceptance(baseline, candidate)["passed"]
    regressed = {**candidate, "format_valid_rate": 0.89}
    assert not evaluate_curriculum_acceptance(baseline, regressed)["passed"]
    with pytest.raises(ValueError, match="identity mismatch"):
        evaluate_curriculum_acceptance(baseline, {**candidate, "model_sha256": "other"})
