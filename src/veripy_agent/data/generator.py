"""Deterministic, file-backed benchmark task materialization."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from veripy_agent.contracts import TaskSpec

FAMILIES = (
    "filter_aggregate",
    "group_rank_topk",
    "multi_table_join",
    "datetime_timeseries",
    "missing_values",
    "outliers_robust_stats",
    "correlation_inference",
    "multi_constraint_decision",
)
SUPPORTED_SPLITS = ("sft_train", "rl_train", "validation", "iid_test", "ood_test")


@dataclass(frozen=True)
class GeneratedTask:
    task_id: str
    family: str
    prompt: str
    input: Any
    oracle: Any
    seed: int
    template_version: str
    schema_version: str
    verifier_spec: dict[str, Any]
    task_spec: TaskSpec
    content_signature: str
    semantic_signature: str
    input_path: str | None = None

    def to_dict(self):
        value = asdict(self)
        value["task_spec"] = self.task_spec.model_dump()
        return value


def _task(
    seed,
    family,
    i,
    prompt,
    data,
    oracle,
    template,
    schema,
    split,
    *,
    verifier_kind="exact",
    answer_schema=None,
    verifier_options=None,
    structure=None,
    evidence_filters=None,
):
    structure = dict(structure or {})
    difficulty_score = (
        int(structure.get("operation_count", 1))
        + int(structure.get("file_count", 1))
        + int(structure.get("join_depth", 0)) * 2
        + max(0, int(structure.get("schema_width", 1)) - 2) // 2
        + max(0, int(structure.get("minimum_tool_turns", 1)) - 1)
        + int(structure.get("row_count", 0)) // 25
        + int(structure.get("constraint_count", 0))
    )
    difficulty = "easy" if difficulty_score <= 4 else "medium" if difficulty_score <= 8 else "hard"
    structure["difficulty_score"] = difficulty_score
    structure["structural_difficulty"] = difficulty
    operation_sequence = structure.get("operation_sequence", [])
    semantic_signature = hashlib.sha256(
        json.dumps(
            {
                "family": family,
                "template": template,
                "schema": schema,
                "operation_sequence": operation_sequence,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    content_signature = hashlib.sha256(
        json.dumps(
            {"family": family, "prompt": prompt, "input": data},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    tid = f"{split}-{family}-{hashlib.sha256(f'{split}|{family}|{seed}|{template}|{schema}|{i}'.encode()).hexdigest()[:16]}"
    public_verifier = {
        "kind": verifier_kind,
        "generator_index": i,
        "structure": structure,
        "semantic_signature": semantic_signature,
        "evidence_filters": evidence_filters or [],
        **(verifier_options or {}),
    }
    spec = TaskSpec(
        task_id=tid,
        family=family,
        instruction=prompt,
        files=[],
        seed=seed,
        split=split,
        difficulty=difficulty,
        answer_schema=answer_schema or {},
        verifier_spec=public_verifier,
    )
    return GeneratedTask(
        tid,
        family,
        prompt,
        data,
        oracle,
        seed,
        template,
        schema,
        {**public_verifier, "expected": oracle},
        spec,
        content_signature,
        semantic_signature,
    )


def _filter_aggregate(seed, n, split="sft_train"):
    r = random.Random(seed)
    out = []
    for i in range(n):
        ood = split == "ood_test"
        size = 24 if ood else 8 + i % 5
        categories = ["A", "B", "C"]
        rows: list[dict[str, Any]] = [
            {
                "category": r.choice(categories),
                "value": r.randint(1, 30),
                **({"segment": r.choice(["retail", "enterprise"]), "weight": r.randint(1, 3)} if ood else {}),
            }
            for _ in range(size)
        ]
        c = r.choice(categories)
        oracle: Any
        threshold: int | None = None
        if ood:
            segment = r.choice(["retail", "enterprise"])
            oracle = sum(
                row["value"] * row["weight"]
                for row in rows
                if row["category"] == c and row["segment"] == segment
            )
            prompt = (
                f"Filter rows where category is {c} and segment is {segment}, then sum value * weight."
            )
            template = "weighted_segment_aggregate"
            schema = "category_segment_value_weight_rows"
            operations = ["filter_category", "filter_segment", "multiply", "sum"]
            required_columns = ["category", "segment", "value", "weight"]
        elif i % 2:
            threshold = r.randint(8, 18)
            oracle = sum(1 for row in rows if row["category"] == c and row["value"] >= threshold)
            prompt = f"Count rows where category is {c} and value is at least {threshold}."
            template = "threshold_count"
            schema = "category_value_rows"
            operations = ["filter_category", "filter_threshold", "count"]
            required_columns = ["category", "value"]
        else:
            threshold = None
            oracle = sum(row["value"] for row in rows if row["category"] == c)
            prompt = f"Filter rows where category is {c} and sum value."
            template = "category_sum"
            schema = "category_value_rows"
            operations = ["filter_category", "sum"]
            required_columns = ["category", "value"]
        payload: dict[str, Any] = {"rows": rows, "category": c, "operation": template}
        if ood:
            payload["segment"] = segment
        elif threshold is not None:
            payload["threshold"] = threshold
        out.append(
            _task(
                seed,
                "filter_aggregate",
                i,
                prompt,
                payload,
                oracle,
                template,
                schema,
                split,
                answer_schema={"type": "integer"},
                evidence_filters=(
                    [{"category": c}, {"segment": segment}]
                    if ood
                    else [{"category": c}] + ([{"threshold": threshold}] if threshold is not None else [])
                ),
                structure={
                    "operation_sequence": operations,
                    "operation_count": len(operations),
                    "file_count": 1,
                    "join_depth": 0,
                    "schema_width": len(required_columns),
                    "row_count": len(rows),
                    "minimum_tool_turns": 2,
                    "required_columns": required_columns,
                },
            )
        )
    return out


def _group_rank_topk(seed, n, split="sft_train"):
    r = random.Random(seed)
    out = []
    for i in range(n):
        ood = split == "ood_test"
        rows: list[dict[str, Any]] = [
            {
                "group": r.choice(["A", "B", "C", "D"]),
                "score": r.randint(1, 99),
                **({"region": r.choice(["north", "south"])} if ood else {}),
            }
            for _ in range(22 if ood else 10 + i % 4)
        ]
        target_region = r.choice(["north", "south"]) if ood else None
        selected = [row for row in rows if not ood or row["region"] == target_region]
        g: dict[str, list[int]] = {}
        for x in selected:
            score = x["score"]
            if not isinstance(score, int):
                raise TypeError("generated score must be an integer")
            g.setdefault(str(x["group"]), []).append(score)
        if ood:
            metric = {key: round(sum(values) / len(values), 6) for key, values in g.items()}
            template = "regional_mean_topk"
            schema = "group_region_score_rows"
            k = 3
            operations = ["filter_region", "group", "mean", "rank", "topk"]
            prompt = f"Within region {target_region}, rank groups by mean score and return top {k}."
        elif i % 2:
            metric = {key: sum(values) for key, values in g.items()}
            template = "group_sum_topk"
            schema = "group_score_rows"
            k = 3
            operations = ["group", "sum", "rank", "topk"]
            prompt = f"Rank groups by total score and return top {k}."
        else:
            metric = {key: max(values) for key, values in g.items()}
            template = "group_max_topk"
            schema = "group_score_rows"
            k = 2
            operations = ["group", "max", "rank", "topk"]
            prompt = f"Rank groups by maximum score and return top {k}."
        o = [
            {"group": group, "metric": score}
            for group, score in sorted(metric.items(), key=lambda z: (-z[1], z[0]))[:k]
        ]
        payload: dict[str, Any] = {"rows": rows, "k": k, "operation": template}
        if target_region is not None:
            payload["region"] = target_region
        out.append(
            _task(
                seed,
                "group_rank_topk",
                i,
                prompt,
                payload,
                o,
                template,
                schema,
                split,
                verifier_kind="table",
                answer_schema={
                    "type": "array",
                    "items": {"type": "object", "required": ["group", "metric"]},
                },
                verifier_options={"ordered": True},
                structure={
                    "operation_sequence": operations,
                    "operation_count": len(operations),
                    "file_count": 1,
                    "join_depth": 0,
                    "schema_width": 3 if ood else 2,
                    "row_count": len(rows),
                    "minimum_tool_turns": 2,
                    "required_columns": ["group", "score"] + (["region"] if ood else []),
                },
            )
        )
    return out


def _multi_table_join(seed, n, split="sft_train"):
    r = random.Random(seed)
    out = []
    for i in range(n):
        ood = split == "ood_test"
        customer_count = 8 if ood else 4 + i % 3
        cs: list[dict[str, Any]] = [
            {"customer_id": j, "name": f"c{j}", "region_id": j % 3}
            for j in range(customer_count)
        ]
        os: list[dict[str, Any]] = [
            {"customer_id": r.randrange(customer_count), "amount": r.randint(10, 80)}
            for _ in range(20 if ood else 8 + i % 5)
        ]
        regions: list[dict[str, Any]] = [
            {"region_id": 0, "region": "north"},
            {"region_id": 1, "region": "south"},
            {"region_id": 2, "region": "west"},
        ]
        if ood:
            target_region = r.choice(["north", "south", "west"])
            region_ids = {row["region_id"] for row in regions if row["region"] == target_region}
            eligible = [customer for customer in cs if customer["region_id"] in region_ids]
            o = [
                {
                    "customer_id": customer["customer_id"],
                    "total": sum(
                        order["amount"]
                        for order in os
                        if order["customer_id"] == customer["customer_id"]
                    ),
                }
                for customer in eligible
            ]
            prompt = f"Join regions, customers, and orders; return customer totals for region {target_region}."
            template = "regional_customer_totals"
            schema = "regions_customers_orders"
            operations = ["join_regions_customers", "filter_region", "join_orders", "group_sum"]
        elif i % 2:
            target_region = None
            o = [
                {
                    "customer_id": customer["customer_id"],
                    "average": round(
                        sum(
                            order["amount"]
                            for order in os
                            if order["customer_id"] == customer["customer_id"]
                        )
                        / max(1, sum(order["customer_id"] == customer["customer_id"] for order in os)),
                        6,
                    ),
                }
                for customer in cs
            ]
            prompt = "Join customers and orders and return average order amount for every customer."
            template = "customer_order_average"
            schema = "customers_orders"
            operations = ["join_customers_orders", "group_mean"]
        else:
            target_region = None
            o = [
                {
                    "customer_id": customer["customer_id"],
                    "total": sum(
                        order["amount"]
                        for order in os
                        if order["customer_id"] == customer["customer_id"]
                    ),
                }
                for customer in cs
            ]
            prompt = "Join customers and orders and return totals."
            template = "customer_order_totals"
            schema = "customers_orders"
            operations = ["join_customers_orders", "group_sum"]
        payload: dict[str, Any] = {"customers": cs, "orders": os, "operation": template}
        if ood:
            payload.update({"regions": regions, "region": target_region})
        out.append(
            _task(
                seed,
                "multi_table_join",
                i,
                prompt,
                payload,
                o,
                template,
                schema,
                split,
                verifier_kind="table",
                answer_schema={
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["customer_id", "average" if i % 2 and not ood else "total"],
                    },
                },
                verifier_options={"ordered": True},
                structure={
                    "operation_sequence": operations,
                    "operation_count": len(operations),
                    "file_count": 3 if ood else 2,
                    "join_depth": 2 if ood else 1,
                    "schema_width": 7 if ood else 5,
                    "row_count": len(cs) + len(os) + (len(regions) if ood else 0),
                    "minimum_tool_turns": 3 if ood else 2,
                    "required_columns": ["customer_id", "amount"] + (["region"] if ood else []),
                },
            )
        )
    return out


def _datetime_timeseries(seed, n, split="sft_train"):
    r = random.Random(seed)
    out = []
    for i in range(n):
        ood = split == "ood_test"
        size = 24 if ood else 10 + i % 5
        rows: list[dict[str, int | str]] = []
        for day in range(size):
            rows.append({"date": f"2025-01-{day + 1:02d}", "value": r.randint(1, 50)})
        window = 5 if ood else 3 + i % 2
        windows = [rows[j - window + 1 : j + 1] for j in range(window - 1, size)]
        metrics: list[int | float]
        if ood:
            metrics = [int(chunk[-1]["value"]) - int(chunk[0]["value"]) for chunk in windows]
            template = "rolling_growth_peak"
            schema = "dated_value_series_with_calendar"
            operations = ["sort_date", "rolling_window", "difference", "argmax", "tie_break"]
            prompt = (
                f"Find the {window}-day window with the largest end-to-start growth; "
                "break ties by earliest end date."
            )
        elif i % 2:
            metrics = [round(sum(int(row["value"]) for row in chunk) / window, 6) for chunk in windows]
            template = "rolling_mean_peak"
            schema = "dated_value_series"
            operations = ["sort_date", "rolling_window", "mean", "argmax", "tie_break"]
            prompt = (
                f"Find the {window}-day window with the largest rolling mean; "
                "break ties by earliest end date."
            )
        else:
            metrics = [sum(int(row["value"]) for row in chunk) for chunk in windows]
            template = "rolling_sum_peak"
            schema = "dated_value_series"
            operations = ["sort_date", "rolling_window", "sum", "argmax", "tie_break"]
            prompt = (
                f"Find the {window}-day window with the largest rolling sum; "
                "break ties by earliest end date."
            )
        best_offset = max(range(len(metrics)), key=lambda j: (metrics[j], -j))
        best = best_offset + window - 1
        oracle = {"end_date": rows[best]["date"], "metric": metrics[best_offset]}
        out.append(
            _task(
                seed,
                "datetime_timeseries",
                i,
                prompt,
                {"rows": rows, "window": window, "operation": template},
                oracle,
                template,
                schema,
                split,
                answer_schema={"type": "object", "required": ["end_date", "metric"]},
                structure={
                    "operation_sequence": operations,
                    "operation_count": len(operations),
                    "file_count": 1,
                    "join_depth": 0,
                    "schema_width": 2,
                    "row_count": len(rows),
                    "minimum_tool_turns": 2,
                    "window_size": window,
                    "required_columns": ["date", "value"],
                },
            )
        )
    return out


def _missing_values(seed, n, split="sft_train"):
    r = random.Random(seed)
    out = []
    for i in range(n):
        ood = split == "ood_test"
        size = 24 if ood else 9 + i % 5
        values: list[int | None] = [r.randint(2, 40) for _ in range(size)]
        for index in r.sample(range(size), k=5 if ood else 2 + i % 2):
            values[index] = None
        if ood:
            groups = ["A" if index % 2 == 0 else "B" for index in range(size)]
            rows = [{"group": group, "value": value} for group, value in zip(groups, values, strict=True)]
            imputed: list[float] = []
            for group in ("A", "B"):
                group_values = [
                    int(row["value"])
                    for row in rows
                    if row["group"] == group and row["value"] is not None
                ]
                ordered = sorted(group_values)
                middle = len(ordered) // 2
                median = (
                    float(ordered[middle])
                    if len(ordered) % 2
                    else (ordered[middle - 1] + ordered[middle]) / 2
                )
                imputed.extend(
                    median if row["value"] is None else float(row["value"])
                    for row in rows
                    if row["group"] == group
                )
            answer = sum(imputed)
            payload: dict[str, Any] = {
                "rows": rows,
                "strategy": "group_median",
                "operation": "group_median_imputation_sum",
            }
            prompt = "Impute missing values with each group's observed median and return the total sum."
            template = "group_median_imputation_sum"
            schema = "group_nullable_value_rows"
            operations = ["group", "drop_missing", "median", "impute", "sum"]
            required_columns = ["group", "value"]
        else:
            observed = [value for value in values if value is not None]
            if i % 2:
                fill = sum(observed) / len(observed)
                strategy = "mean"
                template = "mean_imputation_sum"
                operations = ["drop_missing", "mean", "impute", "sum"]
            else:
                ordered = sorted(observed)
                middle = len(ordered) // 2
                fill = (
                    float(ordered[middle])
                    if len(ordered) % 2
                    else (ordered[middle - 1] + ordered[middle]) / 2
                )
                strategy = "median"
                template = "median_imputation_sum"
                operations = ["drop_missing", "median", "impute", "sum"]
            answer = sum(observed) + values.count(None) * fill
            payload = {"values": values, "strategy": strategy, "operation": template}
            prompt = f"Impute missing values with the {strategy} of observed values and return the resulting sum."
            schema = "nullable_numeric_vector"
            required_columns = ["value"]
        out.append(
            _task(
                seed,
                "missing_values",
                i,
                prompt,
                payload,
                answer,
                template,
                schema,
                split,
                verifier_kind="numeric",
                answer_schema={"type": "number"},
                verifier_options={"abs_tol": 1e-8, "rel_tol": 1e-8},
                evidence_filters=[{"strategy": payload["strategy"]}],
                structure={
                    "operation_sequence": operations,
                    "operation_count": len(operations),
                    "file_count": 1,
                    "join_depth": 0,
                    "schema_width": len(required_columns),
                    "row_count": size,
                    "minimum_tool_turns": 2,
                    "null_rate": values.count(None) / len(values),
                    "required_columns": required_columns,
                },
            )
        )
    return out


def _outliers_robust_stats(seed, n, split="sft_train"):
    r = random.Random(seed)
    out = []
    for i in range(n):
        ood = split == "ood_test"
        size = 25 if ood else 11 + 2 * (i % 3)
        values = [r.randint(10, 30) for _ in range(size - 1)] + [r.randint(120, 200)]
        ordered = sorted(values)
        if ood:
            trim = max(1, int(len(ordered) * 0.1))
            trimmed = ordered[trim:-trim]
            oracle: Any = round(sum(trimmed) / len(trimmed), 6)
            prompt = "Remove the lowest and highest 10 percent of values and return the trimmed mean."
            template = "trimmed_mean"
            schema = "numeric_vector_with_multiple_outliers"
            operations = ["sort", "trim_tails", "mean"]
            answer_schema: dict[str, Any] = {"type": "number"}
            verifier_kind = "numeric"
        elif i % 2:
            midpoint = len(ordered) // 2
            lower = ordered[:midpoint]
            upper = ordered[midpoint + 1 :]
            q1 = float(lower[len(lower) // 2])
            q3 = float(upper[len(upper) // 2])
            oracle = {"q1": q1, "q3": q3, "iqr": q3 - q1}
            prompt = "Return Q1, Q3, and the interquartile range (IQR) of the values."
            template = "quartiles_iqr"
            schema = "numeric_vector_with_outlier"
            operations = ["sort", "quartiles", "subtract"]
            answer_schema = {"type": "object", "required": ["q1", "q3", "iqr"]}
            verifier_kind = "exact"
        else:
            median = float(ordered[len(ordered) // 2])
            deviations = sorted(abs(value - median) for value in values)
            mad = float(deviations[len(deviations) // 2])
            oracle = {"median": median, "mad": mad}
            prompt = "Return the median and median absolute deviation (MAD) of the values."
            template = "median_mad"
            schema = "numeric_vector_with_outlier"
            operations = ["median", "absolute_deviation", "median"]
            answer_schema = {"type": "object", "required": ["median", "mad"]}
            verifier_kind = "exact"
        out.append(
            _task(
                seed,
                "outliers_robust_stats",
                i,
                prompt,
                {"values": values, "operation": template},
                oracle,
                template,
                schema,
                split,
                verifier_kind=verifier_kind,
                answer_schema=answer_schema,
                verifier_options={"abs_tol": 1e-8, "rel_tol": 1e-8} if verifier_kind == "numeric" else None,
                structure={
                    "operation_sequence": operations,
                    "operation_count": len(operations),
                    "file_count": 1,
                    "join_depth": 0,
                    "schema_width": 1,
                    "row_count": len(values),
                    "minimum_tool_turns": 2,
                    "outlier_count": 1,
                    "required_columns": ["value"],
                },
            )
        )
    return out


def _correlation_inference(seed, n, split="sft_train"):
    r = random.Random(seed)
    out = []
    for i in range(n):
        ood = split == "ood_test"
        size = 24 if ood else 12 + i % 5
        xs = list(range(1, size + 1))
        slope = r.choice([-3, -2, 2, 3])
        ys = [slope * x + r.randint(-2, 2) for x in xs]
        mean_x = sum(xs) / size
        mean_y = sum(ys) / size
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
        denominator = (
            sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
        ) ** 0.5
        if ood:
            rank_x = {value: rank for rank, value in enumerate(sorted(xs), start=1)}
            ordered_y = sorted((value, index) for index, value in enumerate(ys))
            ranks_y = [0] * size
            for rank, (_, index) in enumerate(ordered_y, start=1):
                ranks_y[index] = rank
            d2 = sum((rank_x[x] - rank_y) ** 2 for x, rank_y in zip(xs, ranks_y, strict=True))
            correlation = 1 - 6 * d2 / (size * (size * size - 1))
            template = "spearman_rank_direction"
            schema = "paired_numeric_rows_with_rank_noise"
            operations = ["rank_x", "rank_y", "rank_difference", "spearman", "classify_direction"]
            prompt = "Compute Spearman rank correlation rounded to 6 decimals and report its direction."
        elif i % 2:
            covariance = numerator / size
            oracle = {
                "covariance": round(covariance, 6),
                "direction": "positive" if covariance > 0 else "negative",
            }
            template = "population_covariance_direction"
            schema = "paired_numeric_rows"
            operations = ["mean_x", "mean_y", "center", "covariance", "classify_direction"]
            prompt = "Compute population covariance rounded to 6 decimals and report its direction."
            correlation = None
        else:
            correlation = numerator / denominator
            template = "pearson_direction"
            schema = "paired_numeric_rows"
            operations = ["mean_x", "mean_y", "center", "normalize", "classify_direction"]
            prompt = "Compute Pearson correlation rounded to 6 decimals and report its direction."
        if correlation is not None:
            oracle = {
                "correlation": round(correlation, 6),
                "direction": "positive" if correlation > 0 else "negative",
            }
        out.append(
            _task(
                seed,
                "correlation_inference",
                i,
                prompt,
                {
                    "rows": [{"x": x, "y": y} for x, y in zip(xs, ys, strict=True)],
                    "operation": template,
                },
                oracle,
                template,
                schema,
                split,
                answer_schema={
                    "type": "object",
                    "required": ["covariance" if template == "population_covariance_direction" else "correlation", "direction"],
                },
                structure={
                    "operation_sequence": operations,
                    "operation_count": len(operations),
                    "file_count": 1,
                    "join_depth": 0,
                    "schema_width": 2,
                    "row_count": size,
                    "minimum_tool_turns": 2,
                    "required_columns": ["x", "y"],
                },
            )
        )
    return out


def _multi_constraint_decision(seed, n, split="sft_train"):
    r = random.Random(seed)
    out = []
    for i in range(n):
        ood = split == "ood_test"
        size = 24 if ood else 7 + i % 5
        rows: list[dict[str, Any]] = [
            {
                "name": f"option_{j}",
                "cost": r.randint(20, 100),
                "quality": r.randint(50, 100),
                "risk": r.randint(1, 10),
            }
            for j in range(size)
        ]
        if ood:
            max_cost, min_quality, max_risk = (80, 68, 6)
            min_margin = 20
            for row in rows:
                row["benefit"] = r.randint(10, 50)
            template = "multi_constraint_margin_rank"
            schema = "option_cost_quality_risk_benefit"
            operations = ["filter_cost", "filter_quality", "filter_risk", "compute_margin", "rank", "topk"]
            prompt = "Select up to 3 options satisfying all constraints, ranked by quality plus benefit minus cost."
        else:
            max_cost, min_quality, max_risk = (75, 72, 6)
            min_margin = 0
            if i % 2:
                template = "constraint_quality_cost_rank"
                schema = "option_cost_quality_risk"
                operations = ["filter_cost", "filter_quality", "filter_risk", "rank", "topk"]
                prompt = "Return up to 3 eligible options ranked by quality descending, cost ascending, then name."
            else:
                template = "constraint_quality_rank"
                schema = "option_cost_quality_risk"
                operations = ["filter_cost", "filter_quality", "filter_risk", "rank", "topk"]
                prompt = "Return up to 3 eligible options ranked by quality descending, then name."
        eligible = [
            row
            for row in rows
            if int(row["cost"]) <= max_cost
            and int(row["quality"]) >= min_quality
            and int(row["risk"]) <= max_risk
            and (not ood or int(row["quality"]) + int(row["benefit"]) - int(row["cost"]) >= min_margin)
        ]
        if ood:
            oracle_rows: list[dict[str, Any]] = [
                {
                    "name": str(row["name"]),
                    "quality": int(row["quality"]),
                    "cost": int(row["cost"]),
                    "margin": int(row["quality"]) + int(row["benefit"]) - int(row["cost"]),
                }
                for row in eligible
            ]
            oracle = sorted(
                oracle_rows,
                key=lambda row: (-row["margin"], -row["quality"], row["name"]),
            )[:3]
        else:
            oracle_rows = [
                {
                    "name": str(row["name"]),
                    "quality": int(row["quality"]),
                    "cost": int(row["cost"]),
                }
                for row in eligible
            ]
            oracle = sorted(
                oracle_rows,
                key=lambda row: (-row["quality"], row["cost"] if i % 2 else 0, row["name"]),
            )[:3]
        payload: dict[str, Any] = {
            "rows": rows,
            "constraints": {"max_cost": max_cost, "min_quality": min_quality, "max_risk": max_risk},
            "operation": template,
        }
        if ood:
            payload["min_margin"] = min_margin
        out.append(
            _task(
                seed,
                "multi_constraint_decision",
                i,
                prompt,
                payload,
                oracle,
                template,
                schema,
                split,
                verifier_kind="table",
                answer_schema={
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "quality", "cost"] + (["margin"] if ood else []),
                    },
                },
                verifier_options={"ordered": True},
                evidence_filters=[payload["constraints"]]
                + ([{"min_margin": min_margin}] if ood else []),
                structure={
                    "operation_sequence": operations,
                    "operation_count": len(operations),
                    "file_count": 1,
                    "join_depth": 0,
                    "schema_width": 5 if ood else 4,
                    "row_count": len(rows),
                    "minimum_tool_turns": 3 if ood else 2,
                    "constraint_count": 4 if ood else 3,
                    "required_columns": ["name", "cost", "quality", "risk"] + (["benefit"] if ood else []),
                },
            )
        )
    return out


GENERATORS: dict[str, Callable] = {
    "filter_aggregate": _filter_aggregate,
    "group_rank_topk": _group_rank_topk,
    "multi_table_join": _multi_table_join,
    "datetime_timeseries": _datetime_timeseries,
    "missing_values": _missing_values,
    "outliers_robust_stats": _outliers_robust_stats,
    "correlation_inference": _correlation_inference,
    "multi_constraint_decision": _multi_constraint_decision,
}


PUBLIC_CONTRACT_VERSION = "explicit-v1"


def _explicit_public_contract(task: GeneratedTask) -> GeneratedTask:
    """Clarify conventions only in new identities; keep old oracle/scoring unchanged."""
    convention = ""
    if task.family == "group_rank_topk":
        convention = " Rank by metric descending, then group name ascending for equal metrics."
        if task.template_version == "regional_mean_topk":
            convention += " Round each mean to 6 decimals using Python round before ranking."
    elif task.family == "multi_table_join":
        convention = " Include every eligible customer, including customers with no orders; "
        convention += "use 0 for their total or average. Return rows by customer_id ascending."
        if task.template_version == "customer_order_average":
            convention += " Round averages to 6 decimals using Python round."
    elif task.template_version == "quartiles_iqr":
        convention = (
            " Use this benchmark convention: sort values ascending, exclude the central value, "
            "and split into lower and upper halves. Q1 and Q3 are each half's element at "
            "zero-based index len(half)//2 (upper middle when a half has even length), "
            "not an interpolated percentile or averaged median. IQR = Q3 - Q1."
        )
    elif task.template_version == "rolling_mean_peak":
        convention = " Round each rolling mean to 6 decimals using Python round before ranking."
    prompt = task.prompt + convention
    # Same underlying operation signature, separate public-contract namespace.
    tid = (
        task.task_id.rsplit("-", 1)[0]
        + "-"
        + hashlib.sha256(f"{task.task_id}|{PUBLIC_CONTRACT_VERSION}".encode()).hexdigest()[:16]
    )
    content = hashlib.sha256(
        json.dumps(
            {
                "family": task.family,
                "prompt": prompt,
                "input": task.input,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    public = {**task.task_spec.verifier_spec, "public_contract_version": PUBLIC_CONTRACT_VERSION}
    spec = task.task_spec.model_copy(
        update={"task_id": tid, "instruction": prompt, "verifier_spec": public}
    )
    return replace(
        task,
        task_id=tid,
        prompt=prompt,
        template_version=f"{task.template_version}@{PUBLIC_CONTRACT_VERSION}",
        task_spec=spec,
        content_signature=content,
        verifier_spec={**task.verifier_spec, "public_contract_version": PUBLIC_CONTRACT_VERSION},
    )


def generate_tasks(family, *, seed=0, count=10, split="sft_train", public_contract_version=None):
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family}")
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"split must be one of {SUPPORTED_SPLITS}")
    if count < 0:
        raise ValueError("count must be non-negative")
    if public_contract_version not in {None, PUBLIC_CONTRACT_VERSION}:
        raise ValueError("unsupported public contract version")
    tasks = GENERATORS[family](int(seed), count, split)
    return [_explicit_public_contract(task) for task in tasks] if public_contract_version else tasks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_task_bundle(manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    base = manifest_path.parent
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected_hash in value.get("files", {}).items():
        path = base / relative
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"task bundle file hash mismatch: {relative}")
    records_path = base / value["records_path"]
    oracle_path = base / value["oracle_manifest"]["path"]
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    oracles = [json.loads(line) for line in oracle_path.read_text(encoding="utf-8").splitlines()]
    if len(records) != value["task_count"] or len(oracles) != value["task_count"]:
        raise ValueError("task bundle record count mismatch")
    ids = [record["task_id"] for record in records]
    if ids != value["task_ids"] or ids != [record["task_id"] for record in oracles]:
        raise ValueError("task bundle task ids are inconsistent")
    for record in records:
        if "oracle" in record or "expected" in record.get("verifier_spec", {}):
            raise ValueError("public task record leaks oracle data")
        spec = TaskSpec.model_validate(record["task_spec"])
        if spec.task_id != record["task_id"] or spec.family != record["family"]:
            raise ValueError("public TaskSpec is inconsistent with its task record")
        for relative in spec.files:
            if relative not in value["files"] or not (base / relative).is_file():
                raise ValueError(f"TaskSpec references an untracked input: {relative}")
    if sorted({record["seed"] for record in records}) != value["seeds"]:
        raise ValueError("task bundle seeds are inconsistent with the manifest")
    if sorted({record["template_version"] for record in records}) != value["templates"]:
        raise ValueError("task bundle templates are inconsistent with the manifest")
    if sorted({record["schema_version"] for record in records}) != value["schemas"]:
        raise ValueError("task bundle schemas are inconsistent with the manifest")
    if sorted({record["content_signature"] for record in records}) != value["content_signatures"]:
        raise ValueError("task bundle content signatures are inconsistent with the manifest")
    if sorted({record["semantic_signature"] for record in records}) != value["semantic_signatures"]:
        raise ValueError("task bundle semantic signatures are inconsistent with the manifest")
    value["_records"] = records
    return value


def validate_split_isolation(manifests):
    seen: dict[str, dict[str, set[object]]] = {}
    for item in manifests:
        m = validate_task_bundle(item) if isinstance(item, (str, Path)) else item
        if "_records" not in m:
            raise ValueError("split isolation requires a validated task bundle")
        split = m["split"]
        records = m["_records"]
        actual = {
            "task_ids": {record["task_id"] for record in records},
            "seeds": {record["seed"] for record in records},
            "content_signatures": {record["content_signature"] for record in records},
            "semantic_signatures": {record["semantic_signature"] for record in records},
            "schemas": {record["schema_version"] for record in records},
        }
        for key, values in actual.items():
            if values != set(m.get(key, ())):
                raise ValueError(f"validated bundle {key} are inconsistent with its records")
        prior = seen.setdefault(
            split,
            {
                "task_ids": set(),
                "seeds": set(),
                "content_signatures": set(),
                "semantic_signatures": set(),
                "schemas": set(),
            },
        )
        for key in ("task_ids", "seeds", "content_signatures"):
            vals = actual[key]
            overlap = prior[key] & vals
            if overlap:
                raise ValueError(f"duplicate {key} in split {split}: {sorted(map(str, overlap))}")
            prior[key].update(vals)
        prior["semantic_signatures"].update(actual["semantic_signatures"])
        prior["schemas"].update(actual["schemas"])
    splits = list(seen)
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            for key in ("task_ids", "seeds", "content_signatures"):
                overlap = seen[left][key] & seen[right][key]
                if overlap:
                    raise ValueError(
                        f"split isolation violation ({key}): {left} ∩ {right} = {sorted(map(str, overlap))}"
                    )
            if "ood_test" in {left, right}:
                for key in ("semantic_signatures", "schemas"):
                    overlap = seen[left][key] & seen[right][key]
                    if overlap:
                        raise ValueError(
                            f"OOD isolation violation ({key}): {left} ∩ {right} = {sorted(map(str, overlap))}"
                        )
    return True


def manifest(
    tasks,
    *,
    split,
    seed,
    template_version=None,
    schema_version=None,
    files=None,
    generator_parameters=None,
):
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"split must be one of {SUPPORTED_SPLITS}")
    if any(t.seed != seed for t in tasks):
        raise ValueError("task seed differs from manifest seed")
    ts = sorted({t.template_version for t in tasks})
    ss = sorted({t.schema_version for t in tasks})
    template = template_version or (ts[0] if len(ts) == 1 else "mixed")
    schema = schema_version or (ss[0] if len(ss) == 1 else "mixed")
    return {
        "split": split,
        "seed": seed,
        "seeds": sorted({t.seed for t in tasks}),
        "template": template,
        "schema": schema,
        "templates": ts,
        "schemas": ss,
        "content_signatures": sorted({t.content_signature for t in tasks}),
        "semantic_signatures": sorted({t.semantic_signature for t in tasks}),
        "template_version": template,
        "schema_version": schema,
        "task_count": len(tasks),
        "task_ids": [t.task_id for t in tasks],
        "families": sorted({t.family for t in tasks}),
        "files": files or {},
        "generator_parameters": generator_parameters or {"seed": seed, "count": len(tasks)},
        "license": "self_generated",
        "self_generated": True,
    }


def write_tasks(tasks, output, *, split, seed):
    output = Path(output)
    if any(task.task_spec.split != split for task in tasks):
        raise ValueError("task split differs from output split")
    if any(task.seed != seed for task in tasks):
        raise ValueError("task seed differs from output seed")
    root = output.parent / (output.stem + "_episodes")
    root.mkdir(parents=True, exist_ok=True)
    previous_manifest = output.with_suffix(output.suffix + ".manifest.json")
    current_ids = {task.task_id for task in tasks}
    if previous_manifest.is_file():
        previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
        for task_id in set(previous.get("task_ids", ())) - current_ids:
            stale_root = (root / task_id).resolve()
            if stale_root.parent == root.resolve() and stale_root.is_dir():
                shutil.rmtree(stale_root)
    for stale in root.iterdir():
        if stale.is_file() and stale.suffix in {".csv", ".json"}:
            stale.unlink()
    records = []
    hashes = {}
    oracle_records = []
    for t in tasks:
        ext = ".csv" if t.family == "filter_aggregate" else ".json"
        episode_root = root / t.task_id
        episode_root.mkdir(exist_ok=True)
        for stale in episode_root.iterdir():
            if stale.is_file() and stale.suffix in {".csv", ".json"}:
                stale.unlink()
        path = episode_root / ("input" + ext)
        if ext == ".csv":
            rows = t.input["rows"]
            fieldnames = list(rows[0]) if rows else []
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            raw = stream.getvalue().encode()
        else:
            raw = (json.dumps(t.input, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
        path.write_bytes(raw)
        rel = str(path.relative_to(output.parent))
        hashes[rel] = hashlib.sha256(raw).hexdigest()
        public = t.to_dict()
        public.pop("oracle", None)
        public["verifier_spec"] = {k: v for k, v in t.verifier_spec.items() if k != "expected"}
        spec = t.task_spec.model_copy(update={"files": [rel]})
        public["task_spec"] = spec.model_dump()
        public["input"] = {"path": rel, "format": ext[1:]}
        public["input_path"] = rel
        records.append(public)
        oracle_records.append(
            {
                "task_id": t.task_id,
                "family": t.family,
                "seed": t.seed,
                "split": split,
                "verifier_spec": t.verifier_spec,
                "oracle": t.oracle,
            }
        )
    raw = b"".join(
        (json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n").encode() for x in records
    )
    output.write_bytes(raw)
    hashes[output.name] = hashlib.sha256(raw).hexdigest()
    oracle_path = output.with_suffix(output.suffix + ".oracle.jsonl")
    oracle_raw = b"".join(
        (json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n").encode() for x in oracle_records
    )
    oracle_path.write_bytes(oracle_raw)
    hashes[oracle_path.name] = hashlib.sha256(oracle_raw).hexdigest()
    m = manifest(tasks, split=split, seed=seed, files=hashes)
    m["records_path"] = output.name
    m["record_sha256"] = hashlib.sha256(raw).hexdigest()
    m["sha256"] = m["record_sha256"]
    m["oracle_manifest"] = {"path": oracle_path.name, "sha256": hashes[oracle_path.name]}
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_task_bundle(manifest_path)
    return m
