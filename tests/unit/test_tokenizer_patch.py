import json
from types import SimpleNamespace

from veripy_agent.verl_adapter.tokenizer_patch import (
    install_auto_tokenizer_compat_patch,
    install_tokenizer_compat_patch,
)


def test_transformers_five_special_token_list_is_adapted(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"extra_special_tokens": ["<|im_start|>"]}), encoding="utf-8"
    )
    calls = []
    module = SimpleNamespace(
        hf_tokenizer=lambda path, **kwargs: calls.append((path, kwargs)) or "tokenizer"
    )
    install_tokenizer_compat_patch(module)
    assert module.hf_tokenizer(tmp_path, trust_remote_code=False) == "tokenizer"
    assert calls == [(tmp_path, {"trust_remote_code": False, "extra_special_tokens": {}})]


def test_existing_keyword_is_preserved(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"extra_special_tokens": ["<|im_start|>"]}), encoding="utf-8"
    )
    calls = []
    module = SimpleNamespace(hf_tokenizer=lambda path, **kwargs: calls.append(kwargs))
    install_tokenizer_compat_patch(module)
    module.hf_tokenizer(tmp_path, extra_special_tokens={"image_token": "<image>"})
    assert calls == [{"extra_special_tokens": {"image_token": "<image>"}}]


def test_auto_tokenizer_entry_point_is_adapted_and_idempotent(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"extra_special_tokens": ["<|im_start|>"]}), encoding="utf-8"
    )
    calls = []

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, path, *args, **kwargs):
            calls.append((cls, path, args, kwargs))
            return "tokenizer"

    install_auto_tokenizer_compat_patch(FakeAutoTokenizer)
    install_auto_tokenizer_compat_patch(FakeAutoTokenizer)

    assert FakeAutoTokenizer.from_pretrained(tmp_path, "arg", trust_remote_code=False) == (
        "tokenizer"
    )
    assert calls == [
        (
            FakeAutoTokenizer,
            tmp_path,
            ("arg",),
            {"trust_remote_code": False, "extra_special_tokens": {}},
        )
    ]


def test_module_patch_is_idempotent(tmp_path):
    calls = []
    module = SimpleNamespace(hf_tokenizer=lambda path, **kwargs: calls.append(kwargs))
    install_tokenizer_compat_patch(module)
    install_tokenizer_compat_patch(module)
    module.hf_tokenizer(tmp_path)
    assert calls == [{}]
