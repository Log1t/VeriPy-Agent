import json

from veripy_agent.contracts import FailureCode, Submission, ToolResult, ToolStatus
from veripy_agent.environment.worker import PythonWorker


def _ok(stdout="", artifacts=None):
    return ToolResult(status=ToolStatus.OK, stdout=stdout, artifacts=artifacts or [])


def _fail(code, message):
    return ToolResult(status=ToolStatus.ERROR, stderr=message, failure_code=code)


def inspect_files(paths):
    try:
        return _ok(
            artifacts=sorted(
                str(p.relative_to(paths.inputs)) for p in paths.inputs.rglob("*") if p.is_file()
            )
        )
    except Exception as e:
        return _fail(FailureCode.INVALID_PATH, str(e))


def inspect_table(paths, path, rows=5):
    try:
        p = paths.resolve(path)
        if not p.is_file():
            return _fail(FailureCode.INVALID_PATH, "not a file")
        import pandas as pd

        if p.suffix.lower() == ".csv":
            data = pd.read_csv(p).head(rows).to_json(orient="records")
        elif p.suffix.lower() == ".json":
            obj = json.loads(p.read_text())
            data = json.dumps(obj[:rows] if isinstance(obj, list) else obj, ensure_ascii=False)
        else:
            return _fail(FailureCode.INVALID_FORMAT, "only csv and json are supported")
        return _ok(stdout=data)
    except Exception as e:
        return _fail(FailureCode.INVALID_PATH, str(e))


def execute_python(paths, code, worker: PythonWorker | None = None, **limits):
    if worker is None:
        return _fail(
            FailureCode.SECURITY_VIOLATION,
            "execute_python requires an explicit sandbox worker; host fallback is disabled",
        )
    return worker.run(code)


def submit_answer(payload):
    try:
        Submission.model_validate(payload)
        return _ok(stdout=Submission.model_validate(payload).model_dump_json())
    except Exception as e:
        return _fail(FailureCode.INVALID_FORMAT, str(e))
