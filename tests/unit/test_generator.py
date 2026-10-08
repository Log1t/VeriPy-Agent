import json

import pytest

from veripy_agent.contracts import TaskSpec
from veripy_agent.data import (
    FAMILIES,
    generate_tasks,
    validate_split_isolation,
    validate_task_bundle,
    write_tasks,
)


def test_all_families_are_deterministic_and_have_semantic_contracts():
    assert len(FAMILIES) == 8
    for family in FAMILIES:
        first = generate_tasks(family, seed=4, count=2, split="sft_train")
        repeated = generate_tasks(family, seed=4, count=2, split="sft_train")
        assert [task.to_dict() for task in first] == [task.to_dict() for task in repeated]
        assert all(task.oracle is not None for task in first)
        assert all(isinstance(task.task_spec, TaskSpec) for task in first)
        assert all(task.task_spec.verifier_spec["generator_index"] >= 0 for task in first)
        assert all(task.task_spec.difficulty in {"easy", "medium", "hard"} for task in first)
        assert all(task.task_spec.verifier_spec["structure"]["operation_sequence"] for task in first)
        assert all(task.content_signature for task in first)
        assert all(task.semantic_signature for task in first)


def test_training_families_have_multiple_semantic_templates():
    for family in FAMILIES:
        tasks = generate_tasks(family, seed=4, count=4, split="sft_train")
        assert len({task.template_version for task in tasks}) >= 2


def test_array_answer_schema_names_all_oracle_fields():
    for family in ("group_rank_topk", "multi_table_join", "multi_constraint_decision"):
        for split in ("rl_train", "ood_test"):
            for generated in generate_tasks(family, seed=41000, count=4, split=split):
                schema = generated.task_spec.answer_schema
                assert schema["type"] == "array"
                required = set(schema["items"]["required"])
                assert required
                assert all(required == set(row) for row in generated.oracle)


def test_ood_changes_semantics_and_schema_for_every_family():
    for family in FAMILIES:
        iid = generate_tasks(family, seed=9, count=2, split="iid_test")
        ood = generate_tasks(family, seed=10, count=2, split="ood_test")
        assert {task.semantic_signature for task in iid}.isdisjoint(
            {task.semantic_signature for task in ood}
        )
        assert {task.schema_version for task in iid}.isdisjoint(
            {task.schema_version for task in ood}
        )


def test_filter_aggregate_instruction_exposes_target_category():
    task = generate_tasks("filter_aggregate", seed=9, count=1, split="iid_test")[0]
    assert f"category is {task.input['category']}" in task.task_spec.instruction


def test_task_bundle_is_hash_verified_and_oracle_is_private(tmp_path):
    tasks = generate_tasks("filter_aggregate", seed=9, count=2, split="iid_test")
    manifest = write_tasks(tasks, tmp_path / "x.jsonl", split="iid_test", seed=9)
    manifest_path = tmp_path / "x.jsonl.manifest.json"
    verified = validate_task_bundle(manifest_path)
    assert verified["task_ids"] == manifest["task_ids"]
    records = [json.loads(line) for line in (tmp_path / "x.jsonl").read_text().splitlines()]
    assert all("oracle" not in record for record in records)
    assert all("expected" not in record["verifier_spec"] for record in records)
    assert all("expected" not in record["task_spec"]["verifier_spec"] for record in records)
    input_path = tmp_path / records[0]["input_path"]
    input_path.write_text("tampered\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_task_bundle(manifest_path)


def test_split_isolation_uses_validated_bundles(tmp_path):
    paths = []
    for split, seed in (("validation", 20), ("ood_test", 21)):
        output = tmp_path / f"{split}.jsonl"
        write_tasks(
            generate_tasks("missing_values", seed=seed, count=1, split=split),
            output,
            split=split,
            seed=seed,
        )
        paths.append(output.with_suffix(".jsonl.manifest.json"))
    assert validate_split_isolation(paths)
    duplicate = validate_task_bundle(paths[1])
    duplicate["seeds"] = [20]
    with pytest.raises(ValueError, match="inconsistent"):
        validate_split_isolation([validate_task_bundle(paths[0]), duplicate])


def test_iid_splits_may_share_semantics_but_not_content(tmp_path):
    paths = []
    for split, seed in (("sft_train", 30), ("validation", 31), ("iid_test", 32)):
        output = tmp_path / f"{split}.jsonl"
        write_tasks(
            generate_tasks("group_rank_topk", seed=seed, count=2, split=split),
            output,
            split=split,
            seed=seed,
        )
        paths.append(output.with_suffix(".jsonl.manifest.json"))
    assert validate_split_isolation(paths)
    manifests = [validate_task_bundle(path) for path in paths]
    assert manifests[0]["semantic_signatures"] == manifests[1]["semantic_signatures"]
    assert set(manifests[0]["content_signatures"]).isdisjoint(
        manifests[1]["content_signatures"]
    )


def test_write_tasks_removes_only_stale_episode_files(tmp_path):
    output = tmp_path / "tasks.jsonl"
    root = tmp_path / "tasks_episodes"
    root.mkdir()
    (root / "stale.json").write_text("{}")
    (root / "keep.txt").write_text("owned by user")
    write_tasks(
        generate_tasks("missing_values", seed=3, count=1),
        output,
        split="sft_train",
        seed=3,
    )
    assert not (root / "stale.json").exists()
    assert (root / "keep.txt").read_text() == "owned by user"
