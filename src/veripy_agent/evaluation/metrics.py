from __future__ import annotations

import random


def aggregate(results):
    xs = list(results)
    scores = [float(x.get("score", x) if isinstance(x, dict) else x) for x in xs]
    return {
        "count": len(scores),
        "mean": sum(scores) / len(scores) if scores else 0.0,
        "passed": sum(x >= 1 for x in scores),
    }


def bootstrap_ci(scores, *, n_boot=2000, alpha=0.05, seed=0):
    xs = [float(x) for x in scores]
    if not xs:
        return (0.0, 0.0)
    if n_boot <= 0 or not 0 < alpha < 1:
        raise ValueError("invalid bootstrap parameters")
    r = random.Random(seed)
    means = sorted(sum(r.choice(xs) for _ in xs) / len(xs) for _ in range(n_boot))
    return (means[int(alpha * n_boot / 2)], means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))])


def pass_at_k(results, k=1):
    if k <= 0:
        raise ValueError("k must be positive")
    groups: dict[str, list[bool]] = {}
    for x in results:
        if isinstance(x, dict):
            groups.setdefault(x.get("task_id", "_"), []).append(
                bool(x.get("passed", x.get("score", 0) >= 1))
            )
    return sum(any(v[:k]) for v in groups.values()) / len(groups) if groups else 0.0


def summarize(results, **kwargs):
    xs = list(results)
    out = aggregate(xs)
    scores = [float(x.get("score", x) if isinstance(x, dict) else x) for x in xs]
    out["ci95"] = bootstrap_ci(scores, **kwargs)
    out["pass_at_1"] = pass_at_k(xs, 1)
    return out
