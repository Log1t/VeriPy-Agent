from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _needs_special_token_adaptation(name_or_path: str | Path) -> bool:
    config_path = Path(name_or_path) / "tokenizer_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    return isinstance(config.get("extra_special_tokens"), list)


def _adapt_kwargs(name_or_path: str | Path, kwargs: dict[str, Any]) -> None:
    if _needs_special_token_adaptation(name_or_path):
        kwargs.setdefault("extra_special_tokens", {})


def install_auto_tokenizer_compat_patch(auto_tokenizer_class: Any | None = None) -> None:
    """Patch the Transformers entry point used directly by vLLM.

    Qwen tokenizer configs exported by Transformers 5 encode
    ``extra_special_tokens`` as a list. Transformers 4.51 expects a mapping.
    The patch only changes the in-memory call arguments and never rewrites the
    model snapshot.
    """

    if auto_tokenizer_class is None:
        from transformers import AutoTokenizer

        auto_tokenizer_class = AutoTokenizer
    if getattr(auto_tokenizer_class, "_veripy_compat_installed", False):
        return

    original = auto_tokenizer_class.from_pretrained

    def patched(cls: Any, name_or_path: str | Path, *args: Any, **kwargs: Any) -> Any:
        _adapt_kwargs(name_or_path, kwargs)
        return original(name_or_path, *args, **kwargs)

    target: Any = auto_tokenizer_class
    target.from_pretrained = classmethod(patched)
    target._veripy_compat_installed = True


def install_tokenizer_compat_patch(module: Any) -> Callable[..., Any]:
    """Adapt tokenizer configs emitted by Transformers 5 for the pinned 4.51 runtime."""

    existing = getattr(module.hf_tokenizer, "_veripy_original", None)
    if existing is not None:
        return existing
    original = module.hf_tokenizer

    def patched(name_or_path: str | Path, **kwargs: Any) -> Any:
        _adapt_kwargs(name_or_path, kwargs)
        return original(name_or_path, **kwargs)

    patched._veripy_original = original  # type: ignore[attr-defined]
    module.hf_tokenizer = patched
    return original
