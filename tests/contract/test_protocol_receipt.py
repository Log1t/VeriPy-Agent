import json

import pytest

from veripy_agent.rollout.protocol import (
    assert_no_hidden_leakage,
    protocol_manifest,
    protocol_manifest_from_resources,
    tool_schema_sha256,
)
from veripy_agent.training.sft import TOOLS


def test_protocol_manifest_is_stable_and_binds_behavioral_settings():
    first = protocol_manifest(
        TOOLS,
        tokenizer_revision="qwen-test-revision",
        chat_template_sha256="a" * 64,
    )
    repeated = protocol_manifest(
        TOOLS,
        tokenizer_revision="qwen-test-revision",
        chat_template_sha256="a" * 64,
    )
    assert first == repeated
    assert first["tools_sha256"] == tool_schema_sha256(TOOLS)
    assert first["enable_thinking"] is False
    assert first["max_turns"] == 6
    assert len(first["protocol_sha256"]) == 64


def test_recursive_hidden_oracle_leakage_is_rejected():
    assert_no_hidden_leakage({"task": {"verifier_spec": {"kind": "numeric"}}})
    with pytest.raises(ValueError, match="hidden verifier data"):
        assert_no_hidden_leakage(
            {"messages": [{"role": "tool", "content": {"verifier_diagnostics": {"expected": 3}}}]}
        )


def test_protocol_manifest_binds_pinned_tokenizer_template(tmp_path):
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    lock = tmp_path / "resources.lock.json"
    lock.write_text(
        json.dumps(
            {
                "resources": [
                    {
                        "name": "Qwen/Qwen3-1.7B",
                        "type": "model",
                        "revision": "pinned-revision",
                        "local_path": str(tokenizer),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    receipt = protocol_manifest_from_resources(TOOLS, resources_lock=lock)
    assert receipt["tokenizer_revision"] == "pinned-revision"
    assert len(receipt["chat_template_sha256"]) == 64
