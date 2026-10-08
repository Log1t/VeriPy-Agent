from veripy_agent.evaluation.metrics import aggregate, bootstrap_ci, pass_at_k


def test_metrics():
    xs = [
        {"task_id": "a", "score": 1, "passed": True},
        {"task_id": "a", "score": 0, "passed": False},
    ]
    assert aggregate(xs)["count"] == 2
    assert pass_at_k(xs, 1) == 1.0
    assert len(bootstrap_ci([0, 1], n_boot=20)) == 2
