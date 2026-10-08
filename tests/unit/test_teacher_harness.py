import copy
import hashlib
import json

import pytest

from veripy_agent.data.generator import generate_tasks
from veripy_agent.data.teacher_harness import attach_harness
from veripy_agent.training.agentic import sha


def evidence_fixture(tmp_path):
    task = generate_tasks(
        "missing_values", seed=41, count=1, split="rl_train", public_contract_version="explicit-v1"
    )[0]
    paths = []
    for i, won in enumerate([False, True, True, False]):
        path = tmp_path / f"{i}.json"
        receipt = {
            "schema": "veripy.train_rollout_receipt",
            "source": "train_rollout",
            "split": "rl_train",
            "task_id": task.task_id,
            "family": task.family,
            "structure": task.task_spec.verifier_spec["structure"],
            "failure_code": "success" if won else "incorrect",
            "won": won,
            "submitted": True,
            "episode_reward": float(won),
            "steps": [
                {
                    "turn": 1,
                    "action": "a" * 7001,
                    "tool": "execute_python",
                    "status": "ok",
                    "result": {
                        "stdout": "real stdout",
                        "stderr": "",
                        "hidden_diagnostic": "must not leak",
                    },
                }
            ],
        }
        path.write_text(json.dumps(receipt))
        paths.append(path)
    spec = task.task_spec.model_dump()
    rows = [{"extra_info": {"task_id": task.task_id}, "env_kwargs": {"task": spec}}]
    context = {
        "source_launch_sha256": "a" * 64,
        "source_split": "rl_train",
        "behavior_targeted": True,
        "public_contract_version": "explicit-v1",
        "candidate_count": 1,
        "clusters": [
            {
                "cluster_id": "c0",
                "task_ids": [task.task_id],
                "family": task.family,
                "allowed_designs": [],
                "real_failure_examples": [
                    {
                        "task_id": task.task_id,
                        "receipt_sha256": sha(paths[0]),
                        "successful_siblings": [
                            {"task_id": task.task_id, "receipt_sha256": sha(paths[1])}
                        ],
                    }
                ],
            }
        ],
    }
    binding = {
        "source_launch_sha256": "a" * 64,
        "source_train_sha256": "b" * 64,
        "source_checkpoint": "/historical/checkpoint",
        "purpose": "historical_cpu_integration",
    }
    aggregate = hashlib.sha256()
    for p in paths:
        aggregate.update(p.name.encode() + b"\0" + p.read_bytes())
    binding["source_receipts_sha256"] = aggregate.hexdigest()
    return context, rows, paths, binding


def test_full_public_critical_traces_bound_to_train_source(tmp_path):
    context, rows, paths, binding = evidence_fixture(tmp_path)
    result = attach_harness(context, rows, tmp_path, binding=binding)
    packet = result["teacher_harness"]
    assert packet["schema"] == "veripy.teacher_harness.v1"
    assert packet["source_binding"] == binding
    episode = packet["episodes"][sha(paths[0])]
    assert len(episode["steps"][0]["action"]) == 7001
    assert "hidden_diagnostic" not in json.dumps(packet)
    assert episode["steps"][0]["stdout"] == "real stdout"
    assert "teacher_harness" not in context
    assert packet["student_learning_verified"] is False


@pytest.mark.parametrize("change", ["heldout", "tamper", "missing", "nan", "cross_task", "budget"])
def test_source_contamination_tamper_and_overflow_fail_closed(tmp_path, change):
    context, rows, paths, binding = evidence_fixture(tmp_path)
    if change == "missing":
        paths[3].unlink()
    elif change == "budget":
        with pytest.raises(ValueError, match="budget"):
            attach_harness(context, rows, tmp_path, binding=binding, max_context_bytes=100)
        return
    else:
        r = json.loads(paths[0].read_text())
        if change == "heldout":
            r["split"] = "validation"
        elif change == "nan":
            r["episode_reward"] = float("nan")
        elif change == "cross_task":
            r["task_id"] = "different-task"
        else:
            r["steps"][0]["action"] = "tampered"
        paths[0].write_text(json.dumps(r))
    with pytest.raises(ValueError):
        attach_harness(context, rows, tmp_path, binding=binding)


def proposal_fixture(tmp_path):
    from veripy_agent.data.teacher_curriculum import bucket

    context, rows, paths, binding = evidence_fixture(tmp_path)
    task = generate_tasks(
        "missing_values", seed=41, count=1, split="rl_train", public_contract_version="explicit-v1"
    )[0]
    c = context["clusters"][0]
    c.update(
        operation_sequence=task.task_spec.verifier_spec["structure"]["operation_sequence"],
        difficulty_bucket=bucket(task.task_spec.verifier_spec["structure"]["difficulty_score"]),
        semantic_signature=task.semantic_signature,
        failure_code="incorrect",
    )
    c["allowed_designs"] = [{"generator_index": 0, "input_conditions": ["ordinary_input"]}]
    context = attach_harness(context, rows, tmp_path, binding=binding)
    proposal = {
        "diagnosis": "A hypothesis about computation, not a verified causal root cause.",
        "next_round_focus": "Inspect input and calculate before submitting.",
        "budget_summary": {"generated": 1, "retrieved": 0, "total": 1},
        "assessments": [
            {
                "cluster_id": "c0",
                "disposition": "focus",
                "gap": "input_contract",
                "evidence_receipt_sha256": [sha(paths[0]), sha(paths[1])],
                "observed_fact_ids": [sha(paths[0]) + ":task_correct"],
                "hypothesis": "Input reading could account for this observed failure.",
                "alternative_hypothesis": "The algorithm could instead be implemented incorrectly.",
                "falsification": "Matched training candidates still fail after correct input reading.",
            }
        ],
        "allocations": [
            {
                "cluster_id": "c0",
                "count": 1,
                "generator_index": 0,
                "reason": "Test the input handling hypothesis on matched tasks.",
                "input_condition": "ordinary_input",
                "expected_observable": "Read declared input and compute before a valid submission.",
                "measurement": "computation_and_submit",
                "candidate_source": "generated",
                "retrieval_task_id": None,
            }
        ],
    }
    return context, proposal


def test_structured_observations_and_budget_are_not_teacher_inventions(tmp_path):
    from veripy_agent.data.teacher_curriculum import validate_curriculum_decision

    context, proposal = proposal_fixture(tmp_path)
    assert validate_curriculum_decision(context, json.dumps(proposal)).allocations[0].count == 1
    invalid = copy.deepcopy(proposal)
    invalid["assessments"][0]["observed_fact_ids"] = ["made-up-fact"]
    with pytest.raises(ValueError, match="harness_fact_reference"):
        validate_curriculum_decision(context, json.dumps(invalid))
    invalid = copy.deepcopy(proposal)
    invalid["budget_summary"]["total"] = 2
    with pytest.raises(ValueError, match="harness_budget_summary"):
        validate_curriculum_decision(context, json.dumps(invalid))


def test_retrieval_matches_skill_and_excludes_concrete_heldout_inputs(tmp_path):
    from veripy_agent.data.teacher_curriculum import (
        generate_teacher_candidates,
        validate_curriculum_decision,
    )
    from veripy_agent.data.teacher_harness import attach_retrieval, input_digest

    context, proposal = proposal_fixture(tmp_path)
    task = generate_tasks(
        "missing_values",
        seed=92837,
        count=1,
        split="rl_train",
        public_contract_version="explicit-v1",
    )[0]
    pool = [
        {
            "extra_info": {"task_id": task.task_id},
            "env_kwargs": {"task": task.task_spec.model_dump()},
        }
    ]
    exclusions = {"input_fingerprints": [], "seeds": [], "task_ids": []}
    packet = attach_retrieval(context, pool, exclusions=exclusions)
    assert packet["teacher_harness"]["retrieval_catalog"][0]["task_id"] == task.task_id
    proposal["allocations"][0].update(candidate_source="retrieved", retrieval_task_id=task.task_id)
    proposal["budget_summary"] = {"generated": 0, "retrieved": 1, "total": 1}
    decision = validate_curriculum_decision(packet, json.dumps(proposal))
    candidate = generate_teacher_candidates(packet, decision, base_seed=778)[0]
    assert candidate["task"]["task_id"] == task.task_id
    assert candidate["candidate_source"] == "retrieved"
    exclusions["input_fingerprints"] = [input_digest(task.input)]
    assert not attach_retrieval(context, pool, exclusions=exclusions)["teacher_harness"][
        "retrieval_catalog"
    ]


def test_unavailable_or_duplicate_retrieval_is_rejected(tmp_path):
    from veripy_agent.data.teacher_curriculum import validate_curriculum_decision

    context, proposal = proposal_fixture(tmp_path)
    proposal["allocations"][0].update(candidate_source="retrieved", retrieval_task_id="unseen")
    proposal["budget_summary"] = {"generated": 0, "retrieved": 1, "total": 1}
    with pytest.raises(ValueError, match="harness_retrieval_conflict"):
        validate_curriculum_decision(context, json.dumps(proposal))


def test_sampling_report_never_confuses_reward_python_and_primary(tmp_path):
    from veripy_agent.data.teacher_harness import summarize_sampling

    context, rows, paths, binding = evidence_fixture(tmp_path)
    packet = attach_harness(context, rows, tmp_path, binding=binding)
    report = summarize_sampling(packet, rows, tmp_path)
    assert report["mixed_task_success_groups"] == 1
    assert report["nonzero_episode_reward_variance_groups"] == 1
    assert report["successful_python_episodes"] == 4
    assert report["input_sensitive_primary"] is None
    assert report["turn_weighted_grpo_advantages"] is None
    assert report["training_authorized"] is False
    r = json.loads(paths[0].read_text())
    r["episode_reward"] = 100
    paths[0].write_text(json.dumps(r))
    with pytest.raises(ValueError, match="changed"):
        summarize_sampling(packet, rows, tmp_path)


@pytest.mark.parametrize("outcome", [True, False])
def test_uniform_sampling_is_reported_not_called_learnable(tmp_path, outcome):
    from veripy_agent.data.teacher_harness import summarize_sampling

    context, rows, paths, binding = evidence_fixture(tmp_path)
    for p in paths:
        r = json.loads(p.read_text())
        r.update(
            won=outcome,
            failure_code="success" if outcome else "incorrect",
            episode_reward=float(outcome),
        )
        p.write_text(json.dumps(r))
    aggregate = hashlib.sha256()
    for p in paths:
        aggregate.update(p.name.encode() + b"\0" + p.read_bytes())
    binding["source_receipts_sha256"] = aggregate.hexdigest()
    context["clusters"] = []
    packet = attach_harness(context, rows, tmp_path, binding=binding)
    report = summarize_sampling(packet, rows, tmp_path)
    assert report["mixed_task_success_groups"] == 0
    assert report["nonzero_episode_reward_variance_groups"] == 0
    assert report["sampling_signal_observed"] is False
    assert report["training_authorized"] is False


@pytest.mark.parametrize("source_kind", ["train_rollout", "no_update_student_sampling"])
def test_http_teacher_receives_bound_full_evidence_and_rejected_facts_are_not_repaired(
    tmp_path, monkeypatch, source_kind
):
    from pathlib import Path
    from types import SimpleNamespace

    from veripy_agent.data import teacher_curriculum as teacher

    context, proposal = proposal_fixture(tmp_path)
    if source_kind == "no_update_student_sampling":
        context["source_plan_sha256"] = context.pop("source_launch_sha256")
        context["teacher_harness"]["source_binding"] = {
            "source_kind": source_kind,
            "source_plan_sha256": context["source_plan_sha256"],
            "optimizer_updates": 0,
        }
    invalid = copy.deepcopy(proposal)
    invalid["assessments"][0]["observed_fact_ids"] = ["invented"]
    outputs = iter([invalid, proposal])
    requests = []
    config = SimpleNamespace(
        model="fake",
        provider="fake",
        base_url="https://teacher.invalid/v1",
        resolve_api_key=lambda _: "private-test-key",
    )
    monkeypatch.setattr(teacher, "load_teacher_secret_file", lambda _: (config, {}))
    monkeypatch.setattr(teacher.time, "sleep", lambda _: None)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {"message": {"content": json.dumps(next(outputs))}, "finish_reason": "stop"}
                    ]
                }
            ).encode()

    def request(req, timeout):
        requests.append(json.loads(req.data))
        return Response()

    monkeypatch.setattr(teacher.urllib.request, "urlopen", request)
    output = tmp_path / "teacher"
    teacher.request_curriculum(context, secret=Path("never-read"), output=output)
    prompt = json.loads(requests[0]["messages"][1]["content"])
    assert (
        len(next(iter(prompt["teacher_harness"]["episodes"].values()))["steps"][0]["action"])
        == 7001
    )
    assert "harness_fact_reference" in requests[1]["messages"][-1]["content"]
    assert json.loads(requests[1]["messages"][-2]["content"]) == invalid
    assert json.loads((output / "decision.json").read_text()) == proposal
    assert "private-test-key" not in "".join(p.read_text() for p in output.glob("*.json"))
    provenance = json.loads((output / "provenance.json").read_text())
    if source_kind == "no_update_student_sampling":
        assert provenance["source_plan_sha256"] == context["source_plan_sha256"]
        assert provenance["source_kind"] == source_kind
        assert "source_launch_sha256" not in provenance
    else:
        assert provenance["source_launch_sha256"] == context["source_launch_sha256"]


def test_retrieval_rejects_nontrain_and_modified_public_contract(tmp_path):
    from veripy_agent.data.teacher_harness import attach_retrieval

    context, _ = proposal_fixture(tmp_path)
    task = generate_tasks(
        "missing_values",
        seed=99871,
        count=1,
        split="rl_train",
        public_contract_version="explicit-v1",
    )[0]
    row = {
        "extra_info": {"task_id": task.task_id},
        "env_kwargs": {"task": task.task_spec.model_dump()},
    }
    exclusions = {"input_fingerprints": [], "seeds": [], "task_ids": []}
    for field, value in [("split", "validation"), ("instruction", "modified malicious input")]:
        mutated = copy.deepcopy(row)
        mutated["env_kwargs"]["task"][field] = value
        with pytest.raises(ValueError):
            attach_retrieval(context, [mutated], exclusions=exclusions)


def test_impossible_operation_quota_rejected_before_credentials_or_network(tmp_path, monkeypatch):
    from pathlib import Path

    from veripy_agent.data import teacher_curriculum as teacher

    context, _ = proposal_fixture(tmp_path)
    context["operation_quotas"] = {"wrong-operation-name": 1}
    monkeypatch.setattr(
        teacher, "load_teacher_secret_file", lambda _: pytest.fail("credentials must not be loaded")
    )
    with pytest.raises(ValueError, match="quota"):
        teacher.request_curriculum(context, secret=Path("never-read"), output=tmp_path / "teacher")
    assert not (tmp_path / "teacher").exists()


def test_checkpoint_snapshot_checks_nonempty_complete_bytes_without_tensor_load(tmp_path):
    from veripy_agent.data.teacher_harness import bind_source_checkpoint

    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({"status": "completed", "exit_code": 0, "steps": 8}))
    with pytest.raises(ValueError, match="checkpoint"):
        bind_source_checkpoint(launch)
    for name in [
        "actor/model_world_size_1_rank_0.pt",
        "actor/optim_world_size_1_rank_0.pt",
        "actor/extra_state_world_size_1_rank_0.pt",
        "data.pt",
    ]:
        p = tmp_path / "global_step_8" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fixture-bytes-not-real-tensors")
    result = bind_source_checkpoint(launch)
    assert len(result["artifacts"]) == 4
    assert (
        result["artifacts"]["data.pt"]["sha256"]
        == hashlib.sha256(b"fixture-bytes-not-real-tensors").hexdigest()
    )


def test_exact_retrieval_source_audit_rejects_relabeling_and_id_change(tmp_path):
    from veripy_agent.data.teacher_curriculum import (
        generate_teacher_candidates,
        validate_curriculum_decision,
    )
    from veripy_agent.data.teacher_harness import attach_retrieval, verify_selected_sources

    context, proposal = proposal_fixture(tmp_path)
    task = generate_tasks(
        "missing_values",
        seed=92837,
        count=1,
        split="rl_train",
        public_contract_version="explicit-v1",
    )[0]
    pool = [
        {
            "extra_info": {"task_id": task.task_id},
            "env_kwargs": {"task": task.task_spec.model_dump()},
        }
    ]
    context = attach_retrieval(
        context, pool, exclusions={"input_fingerprints": [], "seeds": [], "task_ids": []}
    )
    proposal["allocations"][0].update(candidate_source="retrieved", retrieval_task_id=task.task_id)
    proposal["budget_summary"] = {"generated": 0, "retrieved": 1, "total": 1}
    decision = validate_curriculum_decision(context, json.dumps(proposal))
    rows = generate_teacher_candidates(context, decision, base_seed=7)
    manifest = {
        "generation_seed": 7,
        "candidates": [
            {
                "task_id": r["task"]["task_id"],
                "candidate_source": r["candidate_source"],
                "retrieval_task_id": r["retrieval_task_id"],
                "mutation_seed": r["mutation_seed"],
                "mutation_index": r["mutation_index"],
            }
            for r in rows
        ],
    }
    assert (
        verify_selected_sources(context, decision, manifest)[task.task_id]["candidate_source"]
        == "retrieved"
    )
    manifest["candidates"][0]["candidate_source"] = "generated"
    with pytest.raises(ValueError, match="retrieval source"):
        verify_selected_sources(context, decision, manifest)
