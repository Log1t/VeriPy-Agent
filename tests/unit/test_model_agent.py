import json

import pytest

from veripy_agent.contracts import FailureCode, StepRecord, ToolResult, ToolStatus
from veripy_agent.data import generate_tasks
from veripy_agent.evaluation.model_agent import episode_messages, parse_tool_call
from veripy_agent.rollout import ToolName, run_episode


def test_parse_qwen_tool_call():
    action = parse_tool_call(
        '<tool_call>\n{"name":"inspect_files","arguments":{}}\n</tool_call><|im_end|>'
    )
    assert action.tool is ToolName.INSPECT_FILES


def test_parse_tool_call_rejects_non_json():
    with pytest.raises(json.JSONDecodeError):
        parse_tool_call("I will inspect the files.")


def test_parse_tool_call_accepts_canonical_json_action():
    action = parse_tool_call('{"tool":"execute_python","arguments":{"code":"print(1)"}}')
    assert action.tool is ToolName.EXECUTE_PYTHON


def test_parse_tool_call_rejects_truncated_json():
    with pytest.raises(json.JSONDecodeError):
        parse_tool_call('{"name":"submit_answer","arguments":{"answer":')


def test_parse_tool_call_accepts_nested_arguments():
    action = parse_tool_call(
        '<think>\n\n</think>\n<tool_call>\n'
        '{"name":"submit_answer","arguments":{"answer":{"value":1},'
        '"evidence":{"columns":[],"filters":[],"intermediate_values":{}}}}'
        '\n</tool_call><|im_end|>'
    )
    assert action.tool is ToolName.SUBMIT_ANSWER
    assert action.arguments["answer"] == {"value": 1}


def test_parse_tool_call_rejects_explanation_around_json():
    with pytest.raises(json.JSONDecodeError):
        parse_tool_call('I will call {"name":"inspect_files","arguments":{}} now.')


def test_parse_tool_call_rejects_extra_fields():
    with pytest.raises(ValueError, match="contain only"):
        parse_tool_call('{"name":"inspect_files","arguments":{},"comment":"guess"}')


def test_episode_messages_begin_with_hidden_task_observation():
    generated = generate_tasks("missing_values", seed=7, count=1, split="validation")[0]
    messages = episode_messages(generated.task_spec, [])
    payload = json.loads(messages[1]["content"])
    assert payload == {
        "instruction": generated.task_spec.instruction,
        "files": generated.task_spec.files,
        "answer_schema": generated.task_spec.answer_schema,
    }


def test_episode_messages_preserve_invalid_output_and_feedback():
    generated = generate_tasks("datetime_timeseries", seed=7, count=1, split="validation")[0]
    raw_output = '{"name":"submit_answer","arguments":{"answer":1}'
    result = ToolResult(
        status=ToolStatus.ERROR,
        stderr="model output contains an incomplete JSON object",
        failure_code=FailureCode.INVALID_REQUEST,
    )
    step = StepRecord(
        observation={"instruction": generated.task_spec.instruction},
        assistant_action={"raw_output": raw_output},
        tool_result=result,
        reward_components={"invalid_call": -0.05},
        next_observation=result.model_dump(mode="json"),
    )

    messages = episode_messages(generated.task_spec, [step])

    assert messages[-2] == {"role": "assistant", "content": raw_output}
    assert messages[-1]["role"] == "user"
    assert "invalid_request" in messages[-1]["content"]
    assert "complete tool call" in messages[-1]["content"]


def test_invalid_raw_action_feedback_preserves_json_parse_error():
    task = generate_tasks("datetime_timeseries", seed=7, count=1, split="validation")[0].task_spec
    raw_output = '{"name":"submit_answer","arguments":{"answer":{}}}}<|im_end|>'

    def agent(_task, *, observation, steps):
        del observation, steps
        try:
            parse_tool_call(raw_output)
        except json.JSONDecodeError as exc:
            return {"raw_output": raw_output, "parse_error": str(exc)}
        raise AssertionError("expected a malformed tool call")

    result = run_episode(task, agent, lambda *_: (_ for _ in ()).throw(AssertionError()), max_turns=1)

    assert result.steps[0].tool_call is None
    assert "Extra data" in result.steps[0].tool_result.stderr
    assert "Field required" not in result.steps[0].tool_result.stderr
