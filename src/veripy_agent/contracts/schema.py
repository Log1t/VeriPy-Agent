import json
from pathlib import Path

from .models import Evidence, StepRecord, Submission, TaskSpec, ToolResult, VerifyResult


def export_json_schemas(directory: str | Path) -> dict[str, Path]:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    models = (TaskSpec, StepRecord, ToolResult, Submission, VerifyResult, Evidence)
    result = {}
    for model in models:
        p = d / f"{model.__name__}.json"
        p.write_text(json.dumps(model.model_json_schema(), indent=2), encoding="utf-8")
        result[model.__name__] = p
    return result
