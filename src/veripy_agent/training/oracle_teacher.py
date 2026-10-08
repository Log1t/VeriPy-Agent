from __future__ import annotations

import hashlib
import json
from typing import Any

from veripy_agent.contracts import Submission, TaskSpec
from veripy_agent.data import generate_tasks
from veripy_agent.rollout import AgentAction, ToolName
from veripy_agent.verifiers.dispatcher import REQUIRED_COLUMNS, recompute_oracle

from .sft import TeacherProvenance, TrajectoryKind


def _generated_task(task: TaskSpec):
    index = int(task.verifier_spec["generator_index"])
    if task.seed is None or task.split is None:
        raise ValueError("oracle smoke requires task seed and split")
    return generate_tasks(task.family, seed=task.seed, count=index + 1, split=task.split)[index]


def _analysis_code(task: TaskSpec) -> str:
    family = task.family
    if len(task.files) != 1:
        raise ValueError("reference analysis requires one mounted JSON input")
    from pathlib import PurePosixPath

    relative = PurePosixPath(task.files[0])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError("reference input path must stay inside /inputs")
    header = f"import json\nwith open({str(PurePosixPath('/inputs') / relative)!r}) as handle:\n    data = json.load(handle)\n"
    snippets = {
        "filter_aggregate": """operation = data['operation']
if operation == 'weighted_segment_aggregate':
    answer = sum(row['value'] * row['weight'] for row in data['rows'] if row['category'] == data['category'] and row['segment'] == data['segment'])
elif operation == 'threshold_count':
    answer = sum(1 for row in data['rows'] if row['category'] == data['category'] and row['value'] >= data['threshold'])
else:
    answer = sum(row['value'] for row in data['rows'] if row['category'] == data['category'])""",
        "group_rank_topk": """operation = data['operation']
rows = [row for row in data['rows'] if operation != 'regional_mean_topk' or row['region'] == data['region']]
grouped = {}
for row in rows:
    grouped.setdefault(row['group'], []).append(row['score'])
if operation == 'group_max_topk':
    metrics = {group: max(values) for group, values in grouped.items()}
elif operation == 'group_sum_topk':
    metrics = {group: sum(values) for group, values in grouped.items()}
else:
    metrics = {group: round(sum(values) / len(values), 6) for group, values in grouped.items()}
answer = [{'group': group, 'metric': metric} for group, metric in sorted(metrics.items(), key=lambda item: (-item[1], item[0]))[:data['k']]]""",
        "multi_table_join": """operation = data['operation']
customers = data['customers']
if operation == 'regional_customer_totals':
    region_ids = {row['region_id'] for row in data['regions'] if row['region'] == data['region']}
    customers = [customer for customer in customers if customer['region_id'] in region_ids]
if operation == 'customer_order_average':
    answer = [
        {'customer_id': customer['customer_id'], 'average': round(sum(order['amount'] for order in data['orders'] if order['customer_id'] == customer['customer_id']) / max(1, sum(order['customer_id'] == customer['customer_id'] for order in data['orders'])), 6)}
        for customer in customers
    ]
else:
    answer = [
        {'customer_id': customer['customer_id'], 'total': sum(order['amount'] for order in data['orders'] if order['customer_id'] == customer['customer_id'])}
        for customer in customers
    ]""",
        "datetime_timeseries": """window = data['window']
chunks = [data['rows'][i-window+1:i+1] for i in range(window - 1, len(data['rows']))]
if data['operation'] == 'rolling_growth_peak':
    metrics = [chunk[-1]['value'] - chunk[0]['value'] for chunk in chunks]
elif data['operation'] == 'rolling_mean_peak':
    metrics = [round(sum(row['value'] for row in chunk) / window, 6) for chunk in chunks]
else:
    metrics = [sum(row['value'] for row in chunk) for chunk in chunks]
offset = max(range(len(metrics)), key=lambda i: (metrics[i], -i))
best = offset + window - 1
answer = {'end_date': data['rows'][best]['date'], 'metric': metrics[offset]}""",
        "missing_values": """import statistics
if data['operation'] == 'group_median_imputation_sum':
    answer = 0.0
    for group in ('A', 'B'):
        rows = [row for row in data['rows'] if row['group'] == group]
        observed = [row['value'] for row in rows if row['value'] is not None]
        fill = statistics.median(observed)
        answer += sum(fill if row['value'] is None else row['value'] for row in rows)
else:
    observed = [value for value in data['values'] if value is not None]
    fill = statistics.mean(observed) if data['strategy'] == 'mean' else statistics.median(observed)
    answer = sum(observed) + data['values'].count(None) * fill""",
        "outliers_robust_stats": """import statistics
ordered = sorted(data['values'])
if data['operation'] == 'trimmed_mean':
    trim = max(1, int(len(ordered) * 0.1))
    answer = round(statistics.mean(ordered[trim:-trim]), 6)
elif data['operation'] == 'quartiles_iqr':
    midpoint = len(ordered) // 2
    q1 = float(ordered[:midpoint][len(ordered[:midpoint]) // 2])
    q3 = float(ordered[midpoint + 1:][len(ordered[midpoint + 1:]) // 2])
    answer = {'q1': q1, 'q3': q3, 'iqr': q3 - q1}
else:
    median = statistics.median(ordered)
    answer = {'median': float(median), 'mad': float(statistics.median(abs(value - median) for value in ordered))}""",
        "correlation_inference": """xs = [row['x'] for row in data['rows']]
ys = [row['y'] for row in data['rows']]
mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
numerator = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
if data['operation'] == 'population_covariance_direction':
    value = numerator / len(xs)
    answer = {'covariance': round(value, 6), 'direction': 'positive' if value > 0 else 'negative'}
elif data['operation'] == 'spearman_rank_direction':
    rank_x = {value: rank for rank, value in enumerate(sorted(xs), start=1)}
    ordered_y = sorted((value, index) for index, value in enumerate(ys))
    ranks_y = [0] * len(ys)
    for rank, (_, index) in enumerate(ordered_y, start=1):
        ranks_y[index] = rank
    d2 = sum((rank_x[x] - rank_y) ** 2 for x, rank_y in zip(xs, ranks_y))
    value = 1 - 6 * d2 / (len(xs) * (len(xs) ** 2 - 1))
    answer = {'correlation': round(value, 6), 'direction': 'positive' if value > 0 else 'negative'}
else:
    value = numerator / (sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys)) ** 0.5
    answer = {'correlation': round(value, 6), 'direction': 'positive' if value > 0 else 'negative'}""",
        "multi_constraint_decision": """constraints = data['constraints']
eligible = [row for row in data['rows'] if row['cost'] <= constraints['max_cost'] and row['quality'] >= constraints['min_quality'] and row['risk'] <= constraints['max_risk']]
if data['operation'] == 'multi_constraint_margin_rank':
    eligible = [row for row in eligible if row['quality'] + row['benefit'] - row['cost'] >= data['min_margin']]
    answer = [{'name': row['name'], 'quality': row['quality'], 'cost': row['cost'], 'margin': row['quality'] + row['benefit'] - row['cost']} for row in sorted(eligible, key=lambda row: (-(row['quality'] + row['benefit'] - row['cost']), -row['quality'], row['name']))[:3]]
else:
    use_cost = data['operation'] == 'constraint_quality_cost_rank'
    answer = [{'name': row['name'], 'quality': row['quality'], 'cost': row['cost']} for row in sorted(eligible, key=lambda row: (-row['quality'], row['cost'] if use_cost else 0, row['name']))[:3]]""",
    }
    try:
        body = snippets[family]
    except KeyError as exc:
        raise ValueError(f"unsupported task family: {family}") from exc
    return f"""{header}{body}
print(json.dumps(answer, ensure_ascii=False))
"""


def oracle_submission(task: TaskSpec) -> Submission:
    answer = recompute_oracle(task)
    columns = sorted(REQUIRED_COLUMNS[task.family])
    filters: list[dict[str, Any]] = list(task.verifier_spec.get("evidence_filters", []))
    return Submission.model_validate(
        {
            "answer": answer,
            "evidence": {
                "columns": columns,
                "filters": filters,
                "intermediate_values": {"answer": answer},
            },
        }
    )


def bind_oracle_submission(task: TaskSpec, actions: list[AgentAction]) -> list[AgentAction]:
    if not actions or actions[-1].tool is not ToolName.SUBMIT_ANSWER:
        raise ValueError("teacher trajectory must finish with submit_answer")
    bound = list(actions)
    bound[-1] = AgentAction(
        tool=ToolName.SUBMIT_ANSWER,
        arguments=oracle_submission(task).model_dump(mode="json"),
    )
    return bound


def bind_sandbox_input_paths(task: TaskSpec, actions: list[AgentAction]) -> list[AgentAction]:
    bound: list[AgentAction] = []
    for action in actions:
        if action.tool is not ToolName.EXECUTE_PYTHON:
            bound.append(action)
            continue
        code = str(action.arguments.get("code", ""))
        for relative in task.files:
            mounted = f"/inputs/{relative}"
            code = code.replace(repr(relative), repr(mounted))
            code = code.replace(json.dumps(relative), json.dumps(mounted))
        bound.append(action.model_copy(update={"arguments": {"code": code}}))
    return bound


def bind_oracle_python_analysis(task: TaskSpec, actions: list[AgentAction]) -> list[AgentAction]:
    python_indexes = [
        index for index, action in enumerate(actions) if action.tool is ToolName.EXECUTE_PYTHON
    ]
    if not python_indexes:
        raise ValueError("teacher trajectory has no Python analysis action")
    bound = list(actions)
    final_python = python_indexes[-1]
    bound[final_python] = AgentAction(
        tool=ToolName.EXECUTE_PYTHON,
        arguments={"code": _analysis_code(task)},
    )
    return bound


def bind_trajectory_kind(
    task: TaskSpec, kind: TrajectoryKind, actions: list[AgentAction]
) -> list[AgentAction]:
    if kind is TrajectoryKind.PLAN_CORRECTION:
        has_invalid_path = any(
            action.tool is ToolName.INSPECT_TABLE
            and str(action.arguments.get("path", "")) not in task.files
            for action in actions
        )
        if has_invalid_path:
            return actions
        if len(actions) >= 6:
            actions = [
                action
                for index, action in enumerate(actions)
                if not (index == 0 and action.tool is ToolName.INSPECT_FILES)
            ]
        if len(actions) >= 6:
            raise ValueError("cannot inject plan correction into a six-action teacher trajectory")
        bound = list(actions)
        bound.insert(
            1,
            AgentAction(
                tool=ToolName.INSPECT_TABLE,
                arguments={"path": "missing-input.csv", "rows": 5},
            ),
        )
        return bound
    if kind is not TrajectoryKind.RECOVERABLE_EXECUTION_ERROR:
        return actions
    has_explicit_error = any(
        action.tool is ToolName.EXECUTE_PYTHON and "raise " in str(action.arguments.get("code", ""))
        for action in actions
    )
    if has_explicit_error:
        return actions
    if len(actions) >= 6:
        actions = [
            action
            for index, action in enumerate(actions)
            if not (index == 0 and action.tool is ToolName.INSPECT_FILES)
        ]
    if len(actions) >= 6:
        raise ValueError("cannot inject recovery error into a six-action teacher trajectory")
    first_python = next(
        (index for index, action in enumerate(actions) if action.tool is ToolName.EXECUTE_PYTHON),
        None,
    )
    if first_python is None:
        raise ValueError("teacher recovery trajectory has no Python analysis to recover into")
    bound = list(actions)
    bound.insert(
        first_python,
        AgentAction(
            tool=ToolName.EXECUTE_PYTHON,
            arguments={"code": "raise ValueError('recoverable teacher error')"},
        ),
    )
    return bound


def oracle_smoke_actions(task: TaskSpec, kind: TrajectoryKind) -> list[AgentAction]:
    submission = oracle_submission(task)
    actions = [AgentAction(tool=ToolName.INSPECT_FILES)]
    if kind is TrajectoryKind.PLAN_CORRECTION:
        actions.append(
            AgentAction(tool=ToolName.INSPECT_TABLE, arguments={"path": "missing-input.csv"})
        )
    actions.append(
        AgentAction(tool=ToolName.INSPECT_TABLE, arguments={"path": task.files[0], "rows": 5})
    )
    if kind is TrajectoryKind.RECOVERABLE_EXECUTION_ERROR:
        actions.append(
            AgentAction(
                tool=ToolName.EXECUTE_PYTHON,
                arguments={"code": "raise ValueError('recoverable teacher error')"},
            )
        )
    actions.extend(
        [
            AgentAction(tool=ToolName.EXECUTE_PYTHON, arguments={"code": _analysis_code(task)}),
            AgentAction(
                tool=ToolName.SUBMIT_ANSWER,
                arguments=submission.model_dump(mode="json"),
            ),
        ]
    )
    return actions


def oracle_smoke_provenance(task: TaskSpec, actions: list[AgentAction]) -> TeacherProvenance:
    request = json.dumps(task.model_dump(mode="json"), sort_keys=True).encode()
    response = json.dumps(
        [action.model_dump(mode="json") for action in actions], sort_keys=True
    ).encode()
    config = b'{"provider":"oracle_smoke","model":"deterministic_oracle"}'
    return TeacherProvenance(
        provider="oracle_smoke",
        model="deterministic_oracle",
        config_sha256=hashlib.sha256(config).hexdigest(),
        request_sha256=hashlib.sha256(request).hexdigest(),
        response_sha256=hashlib.sha256(response).hexdigest(),
    )
