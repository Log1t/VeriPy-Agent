"""Semantic checks for SFT targets; these do not change online rewards."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


def answers_equal(actual: Any, expected: Any) -> bool:
    if type(actual) in (int, float) and type(expected) in (int, float):
        return math.isfinite(actual) and math.isfinite(expected) and math.isclose(
            actual, expected, rel_tol=1e-8, abs_tol=1e-8
        )
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            answers_equal(actual[k], value) for k, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            answers_equal(a, b) for a, b in zip(actual, expected, strict=True)
        )
    return actual == expected


def parse_answer_stdout(stdout: str) -> Any:
    """Require the final nonempty line to be JSON, not an earlier lucky print."""
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError('stdout has no answer')
    try:
        value = json.loads(lines[-1])
    except ValueError as exc:
        raise ValueError('stdout final line is not a JSON answer') from exc
    if isinstance(value, dict) and set(value) == {'answer'}:
        value = value['answer']
    return value


def require_stdout_answer(stdout: str, expected: Any) -> Any:
    actual = parse_answer_stdout(stdout)
    if not answers_equal(actual, expected):
        raise ValueError('stdout answer does not match verified submission')
    return actual


def require_input_sensitive_program(task: Any, code: str, expected: Any) -> list[dict[str, Any]]:
    """Replay unmodified code on two generated, different-answer training inputs.

    Counterfactuals are private admission evidence, never appended to model history.
    This is a bounded check, not a proof of generalization or policy recovery.
    """
    from veripy_agent.data import generate_tasks
    from veripy_agent.environment import HardenedProcessWorker, SandboxPaths
    from veripy_agent.verifiers.dispatcher import recompute_oracle

    if len(task.files) != 1 or task.seed is None or task.split is None:
        raise ValueError('input replay requires a generated single-file task')
    relative = PurePosixPath(task.files[0])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('input replay path must remain inside /inputs')
    index = int(task.verifier_spec['generator_index'])
    receipts: list[dict[str, Any]] = []
    for ordinal in range(128):
        seed = int(hashlib.sha256(f'sft-admission-v1:{task.task_id}:{ordinal}'.encode()).hexdigest()[:8], 16)
        generated = generate_tasks(task.family, seed=seed, count=index+1,
                                   split='sft_train', public_contract_version='explicit-v1')[index]
        answer = recompute_oracle(generated.task_spec)
        if answers_equal(answer, expected) or any(answers_equal(answer, r['answer']) for r in receipts):
            continue
        with tempfile.TemporaryDirectory(prefix='veripy-sft-replay-', dir='/tmp') as tmp:
            root = Path(tmp)
            root.chmod(0o755)
            inputs = root / 'inputs'
            inputs.mkdir(mode=0o755)
            path = inputs / task.files[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(generated.input))
            worker = HardenedProcessWorker(SandboxPaths(inputs, root/'scratch'))
            result = worker.run(code)
            if result.status.value != 'ok':
                raise ValueError('input replay Python execution failed')
            try:
                require_stdout_answer(result.stdout, answer)
            except ValueError as exc:
                raise ValueError('input replay stdout does not compute new answer') from exc
            receipts.append({'task_id': generated.task_id, 'seed': seed, 'answer': answer,
                             'input_fingerprint': hashlib.sha256(json.dumps(generated.input, sort_keys=True,
                                separators=(',', ':')).encode()).hexdigest(),
                             'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                             'result': result.model_dump(mode='json')})
        if len(receipts) == 2:
            return receipts
    raise ValueError('could not generate two distinct-answer admission probes')


def positive_sft_prefixes(item: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Failed actions stay in history; positive Python targets must compute the terminal answer.

    This validates observed output/submission consistency, not independent historical
    correctness. New preparation must first independently admit/replay each trajectory.
    """
    messages = item['conversations']
    terminal = None
    for m in reversed(messages):
        if m['from'] == 'function_call':
            try:
                call = json.loads(m['value'])
                if call['name'] == 'submit_answer':
                    terminal = call['arguments']
                    break
            except (ValueError, KeyError, TypeError):
                pass
    records, rejected = [], []
    latest_stdout = None
    for index, message in enumerate(messages):
        if message['from'] not in {'gpt', 'function_call'}:
            continue
        name = None
        try:
            if message['from'] != 'function_call':
                raise ValueError('non_tool_target')
            call = json.loads(message['value'])
            if not isinstance(call, dict) or not isinstance(call.get('arguments'), dict):
                raise ValueError('invalid_tool_call_shape')
            name, arguments = call['name'], call['arguments']
            if name == 'execute_python':
                latest_stdout = None
            if terminal is None or 'answer' not in terminal:
                raise ValueError('missing_terminal_answer')
            if name == 'submit_answer':
                if not answers_equal(arguments['answer'], terminal['answer']):
                    raise ValueError('nonterminal_submission')
                evidence_record = arguments.get('evidence', {})
                if not isinstance(evidence_record, dict):
                    raise ValueError('invalid_evidence_shape')
                evidence = evidence_record.get('intermediate_values', {})
                if not isinstance(evidence, dict):
                    raise ValueError('invalid_intermediate_values_shape')
                if not answers_equal(evidence.get('answer'), arguments['answer']):
                    raise ValueError('submission_evidence_mismatch')
                require_stdout_answer(latest_stdout or '', arguments['answer'])
            else:
                feedback = messages[index + 1]
                if feedback['from'] != 'observation':
                    raise ValueError('missing_tool_feedback')
                result = json.loads(feedback['value'])
                if not isinstance(result, dict):
                    raise ValueError('invalid_feedback_shape')
                if result.get('status') != 'ok':
                    raise ValueError('failed_tool_action')
                if name == 'execute_python':
                    latest_stdout = result.get('stdout', '')
                    require_stdout_answer(latest_stdout, terminal['answer'])
                    if any(isinstance(n, ast.Raise) for n in ast.walk(ast.parse(arguments['code']))):
                        raise ValueError('explicit_raise_target')
                elif name not in {'inspect_files', 'inspect_table'}:
                    raise ValueError('unknown_tool')
        except (ValueError, KeyError, IndexError, TypeError, SyntaxError) as exc:
            if name == 'execute_python':
                latest_stdout = None
            rejected.append({'message_index': index, 'reason': str(exc)})
        else:
            records.append({'conversations': messages[:index+1], 'tools': item['tools']})
    return records, rejected
