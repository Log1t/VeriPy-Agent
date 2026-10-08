"""Future-only public conventions, exercised through real CPU I/O."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from veripy_agent.contracts import TaskSpec, ToolStatus
from veripy_agent.data.curriculum import admit_curriculum_candidates
from veripy_agent.data.generator import FAMILIES, generate_tasks, validate_task_bundle, write_tasks
from veripy_agent.data.teacher_curriculum import CurriculumDecision, generate_teacher_candidates
from veripy_agent.environment import HardenedProcessWorker, SandboxPaths
from veripy_agent.rollout.messages import decode_message_observation
from veripy_agent.verifiers.dispatcher import verify_answer, verify_task_submission
from veripy_agent.verl_adapter.data import build_rollout_row


def test_legacy_snapshot_is_unchanged():
    records = [
        t.to_dict()
        for f in FAMILIES
        for t in generate_tasks(f, seed=20261005, count=4, split="rl_train")
    ]
    assert hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest() == (
        "96cf0f6f0f938bca3ace01652b3e77fe7a1b2778a2e84f3e70b51658b68d8eb5"
    )


def test_content_signature_cannot_hide_identical_inputs_behind_version_namespace():
    old = generate_tasks("missing_values", seed=4, count=1, split="rl_train")[0]
    new = generate_tasks(
        "missing_values", seed=4, count=1, split="rl_train", public_contract_version="explicit-v1"
    )[0]
    assert new.task_id != old.task_id
    assert new.input == old.input and new.prompt == old.prompt
    assert new.content_signature == old.content_signature


@pytest.mark.parametrize(
    "family,seed,index,required",
    [
        ("group_rank_topk", 24, 0, "group name ascending"),
        ("multi_table_join", 4, 1, "no orders"),
        ("outliers_robust_stats", 7, 1, "upper middle"),
    ],
)
def test_explicit_conventions_have_separate_identity_and_cpu_submission(
    family, seed, index, required
):
    old = generate_tasks(family, seed=seed, count=index + 1, split="rl_train")[index]
    new = generate_tasks(
        family, seed=seed, count=index + 1, split="rl_train", public_contract_version="explicit-v1"
    )[index]
    assert required in new.prompt
    assert new.task_id != old.task_id and new.content_signature != old.content_signature
    assert new.input == old.input and new.oracle == old.oracle
    assert new.task_spec.verifier_spec["public_contract_version"] == "explicit-v1"
    assert "expected" not in new.task_spec.verifier_spec
    with tempfile.TemporaryDirectory(prefix="veripy-public-contract-", dir="/tmp") as tmp:
        root = Path(tmp)
        root.chmod(0o755)
        write_tasks([new], root / "tasks.jsonl", split="rl_train", seed=seed)
        records = validate_task_bundle(root / "tasks.jsonl.manifest.json")["_records"]
        spec = TaskSpec.model_validate(records[0]["task_spec"])
        input_root = (root / spec.files[0]).parent
        spec = spec.model_copy(update={"files": [(root / spec.files[0]).name]})
        row = build_rollout_row(spec, input_root=str(input_root))
        messages = decode_message_observation(row["prompt"][0]["content"])
        assert json.loads(messages[1]["content"])["instruction"] == new.prompt
        worker = HardenedProcessWorker(SandboxPaths(input_root, root / "scratch"))
        # Independent reference calculates from mounted input, not generator oracle.
        result = worker.run(REFERENCE)
        assert result.status == ToolStatus.OK, result.stderr
        answer = json.loads(result.stdout)
        assert verify_answer(spec, answer).passed
        submission = {
            "answer": answer,
            "evidence": {
                "columns": spec.verifier_spec["structure"]["required_columns"],
                "filters": [],
                "intermediate_values": {"answer": answer},
            },
        }
        verified, evidence_ok = verify_task_submission(spec, submission)
        assert verified.passed and evidence_ok
        assert not verify_answer(spec, {"wrong_field": 0}).passed
        bad = json.loads(json.dumps(answer))
        if family == "group_rank_topk":
            bad.reverse()  # tied scores, wrong lexical order
        elif family == "multi_table_join":
            bad[1]["average"] = 104 / 3  # unrounded value must remain rejected
        else:
            bad["q1"] += 0.5
        assert not verify_answer(spec, bad).passed
        with pytest.raises(ValueError, match="instruction mismatch"):
            verify_answer(spec.model_copy(update={"instruction": old.prompt}), answer)
        # A profile attached to an old identity must not sneak through verification.
        with pytest.raises(ValueError, match="identity mismatch"):
            verify_answer(
                old.task_spec.model_copy(
                    update={
                        "verifier_spec": {
                            **old.task_spec.verifier_spec,
                            "public_contract_version": "explicit-v1",
                        }
                    }
                ),
                answer,
            )


def test_unknown_public_contract_fails_closed():
    with pytest.raises(ValueError, match="public contract"):
        generate_tasks("group_rank_topk", public_contract_version="typo")


def test_teacher_profile_propagates_through_admission_and_materialization(tmp_path):
    base = generate_tasks("multi_table_join", seed=4, count=2, split="rl_train")[1]
    structure = base.task_spec.verifier_spec["structure"]
    score = structure["difficulty_score"]
    cluster = {
        "cluster_id": "cluster-000",
        "family": base.family,
        "allowed_designs": [{"generator_index": 1}],
        "operation_sequence": structure["operation_sequence"],
        "semantic_signature": base.semantic_signature,
        "difficulty_bucket": "low" if score <= 4 else "medium" if score <= 8 else "high",
        "task_ids": [base.task_id],
        "failure_code": "incorrect",
    }
    context = {
        "source_split": "rl_train",
        "candidate_count": 2,
        "public_contract_version": "explicit-v1",
        "clusters": [cluster],
    }
    decision = CurriculumDecision(
        diagnosis="Actual interface regression fixture only.",
        allocations=[
            {
                "cluster_id": "cluster-000",
                "count": 2,
                "generator_index": 1,
                "reason": "CPU input/output compatibility test only.",
            }
        ],
        next_round_focus="No model learning is tested here.",
    )
    generated = generate_teacher_candidates(context, decision, base_seed=20261006)
    accepted, rejected = admit_curriculum_candidates(generated)
    assert len(accepted) == 2 and not rejected
    path = Path(__file__).resolve().parents[2] / "scripts/prepare_curriculum_training.py"
    module_spec = importlib.util.spec_from_file_location("prepare_profile_fixture", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    rows = module._materialize(accepted, output_root=tmp_path, dataset_root=tmp_path)
    assert len(rows) == 2
    for row in rows:
        spec = TaskSpec.model_validate(row["env_kwargs"]["task"])
        assert spec.verifier_spec["public_contract_version"] == "explicit-v1"
        assert "no orders" in spec.instruction and "6 decimals" in spec.instruction


@pytest.mark.parametrize("profile", [None, "explicit-v1"])
def test_real_curriculum_audit_requires_explicit_quartile_convention(profile):
    script = Path(__file__).resolve().parents[2] / "scripts/audit_curriculum_inputs.py"
    with tempfile.TemporaryDirectory(prefix="veripy-quartile-audit-", dir="/tmp") as tmp:
        root = Path(tmp)
        root.chmod(0o755)
        task = generate_tasks(
            "outliers_robust_stats",
            seed=7,
            count=2,
            split="rl_train",
            public_contract_version=profile,
        )[1]
        write_tasks([task], root / "tasks.jsonl", split="rl_train", seed=7)
        spec = TaskSpec.model_validate(
            validate_task_bundle(root / "tasks.jsonl.manifest.json")["_records"][0]["task_spec"]
        )
        input_path = root / spec.files[0]
        spec = spec.model_copy(update={"files": [input_path.name]})
        row = build_rollout_row(spec, input_root=str(input_path.parent.relative_to(root)))
        parquet = root / "train.parquet"
        pd.DataFrame([row]).to_parquet(parquet, index=False)
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "train_parquet": str(parquet),
                    "train_parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
                    "added_task_ids": [task.task_id],
                }
            )
        )
        evidence = root / "audit.json"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--manifest",
                str(manifest),
                "--dataset-root",
                str(root),
                "--output",
                str(evidence),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=os.environ.copy(),
        )
        if profile:
            assert result.returncode == 0, result.stderr
            assert json.loads(evidence.read_text())["passed"]
        else:
            assert result.returncode != 0 and "candidate blocked" in result.stderr
            assert not evidence.exists()


REFERENCE = r"""
import json, statistics
from pathlib import Path
d = json.loads(next(Path('/inputs').iterdir()).read_text())
op = d['operation']
if op == 'group_max_topk':
    groups = sorted({r['group'] for r in d['rows']})
    rows = [{'group': g, 'metric': max(r['score'] for r in d['rows'] if r['group']==g)} for g in groups]
    answer = sorted(rows, key=lambda r: (-r['metric'], r['group']))[:d['k']]
elif op == 'customer_order_average':
    answer = []
    for c in sorted(d['customers'], key=lambda r: r['customer_id']):
        values = [r['amount'] for r in d['orders'] if r['customer_id']==c['customer_id']]
        answer.append({'customer_id': c['customer_id'], 'average': round(statistics.mean(values), 6) if values else 0})
else:
    values = sorted(d['values'])
    midpoint = len(values)//2
    lower, upper = values[:midpoint], values[midpoint+1:]
    q1, q3 = lower[len(lower)//2], upper[len(upper)//2]
    answer = {'q1': q1, 'q3': q3, 'iqr': q3-q1}
print(json.dumps(answer))
"""
