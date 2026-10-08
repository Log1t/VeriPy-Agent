"""Conservative posthoc computation audit; never edits code or model observations."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from veripy_agent.contracts import TaskSpec, ToolStatus
from veripy_agent.environment import HardenedProcessWorker, SandboxPaths
from veripy_agent.verifiers.dispatcher import verify_answer


def computation_prefix(record):
    """Return original attempted Python actions BEFORE the first accepted submission."""
    if not record.get("correct") or not record.get("submitted"):
        return []
    codes: list[str] = []
    successful = False
    for step in record["steps"]:
        call = step.get("tool_call") or {}
        result = step.get("tool_result") or {}
        tool = call.get("tool", call.get("name"))
        if tool == "submit_answer" and result.get("status") == "ok":
            return codes if successful else []
        if tool == "execute_python":
            code = call.get("arguments", {}).get("code")
            if not isinstance(code, str):
                return []
            codes.append(code)
            successful |= result.get("status") == "ok"
    return []


def stdout_correct(task: TaskSpec, stdout: str) -> bool:
    # Deliberately narrow: documented numeric stdout, not regex extraction from prose.
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and set(value) == {"answer"}:
            value = value["answer"]
        if type(value) not in (int, float):
            continue
        if verify_answer(task, value).passed:
            return True
    return False


def audit_computation(record, task: TaskSpec, input_file: Path, probes: list[dict]):
    """Original output + original and two different-input unmodified-code replays."""
    codes = computation_prefix(record)
    if not codes:
        return {"passed": False, "reason": "no_successful_python_before_correct_submission"}
    if not record.get("evidence_consistent"):
        return {"passed": False, "reason": "submission_evidence_inconsistent"}
    observed = False
    for step in record["steps"]:
        call = step.get("tool_call") or {}
        result = step.get("tool_result") or {}
        tool = call.get("tool", call.get("name"))
        if tool == "submit_answer" and result.get("status") == "ok":
            break
        if tool == "execute_python" and result.get("status") == "ok":
            observed |= stdout_correct(task, result.get("stdout", ""))
    if not observed:
        return {"passed": False, "reason": "no_verifier_correct_numeric_stdout_in_original"}
    if task.family != "missing_values" or len(task.files) != 1 or len(probes) != 2:
        raise ValueError("computation audit is scoped to missing-value original plus two probes")
    tasks = [(task, input_file)]
    for probe in probes:
        sister = TaskSpec.model_validate(probe["task"])
        if sister.verifier_spec["generator_index"] != task.verifier_spec["generator_index"]:
            raise ValueError("offline probe changed generator index")
        if sister.verifier_spec["structure"] != task.verifier_spec["structure"]:
            raise ValueError("offline probe changed structural contract")
        tasks.append((sister, Path(probe["input_file"])))
    replays = []
    for sister, path in tasks:
        with tempfile.TemporaryDirectory(prefix="veripy-computation-", dir="/tmp") as tmp:
            root = Path(tmp)
            root.chmod(0o755)
            inputs = root / "inputs"
            inputs.mkdir(mode=0o755)
            # Keep filenames stable. Only actual mounted bytes differ between probes.
            shutil.copyfile(path, inputs / task.files[0])
            worker = HardenedProcessWorker(SandboxPaths(inputs, root / "scratch"))
            steps = []
            produced_correct = False
            for code in codes:
                result = worker.run(code)
                steps.append(result.model_dump(mode="json"))
                if result.status == ToolStatus.SECURITY_VIOLATION:
                    produced_correct = False
                    break
                if result.status == ToolStatus.OK:
                    produced_correct |= stdout_correct(sister, result.stdout)
        replays.append(
            {
                "task_id": sister.task_id,
                "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "correct_numeric_output": produced_correct,
                "steps": steps,
            }
        )
    return {
        "passed": all(r["correct_numeric_output"] for r in replays),
        "reason": "passed"
        if all(r["correct_numeric_output"] for r in replays)
        else "input_replay_failed",
        "code_sha256": [hashlib.sha256(c.encode()).hexdigest() for c in codes],
        "replays": replays,
        "program_repaired": False,
        "scope": "posthoc_input_sensitive_computation_not_policy_recovery",
    }
