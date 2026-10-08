#!/usr/bin/env bash
set -euo pipefail

: "${VERIPY_MODEL_PATH:=/models/VeriPy-Agent-Qwen3-1.7B-SFT}"
: "${VERIPY_CACHE_ROOT:=/cache}"
: "${VERIPY_CHECKPOINT_ROOT:=/checkpoints}"
export HF_HOME="${VERIPY_HF_CACHE:-${VERIPY_CACHE_ROOT}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export VLLM_CACHE_ROOT="${VERIPY_VLLM_CACHE:-${VERIPY_CACHE_ROOT}/vllm}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/veripy-hf-datasets}"
export VERIPY_MODEL_PATH VERIPY_CHECKPOINT_ROOT HF_HOME TRANSFORMERS_CACHE VLLM_CACHE_ROOT HF_DATASETS_CACHE

mkdir -p "$VERIPY_CHECKPOINT_ROOT" "$HF_HOME" "$VLLM_CACHE_ROOT" "$HF_DATASETS_CACHE"

for repository in \
  /workspace/veripy-agent/external/LLaMA-Factory \
  /workspace/veripy-agent/external/verl-agent \
  /workspace/veripy-agent/external/verl; do
  if [[ -d "$repository/.git" ]]; then
    git config --global --add safe.directory "$repository"
  fi
done

if [[ "${1:-}" == "--check" ]]; then
  python - <<'PY'
import json
import os
from pathlib import Path

import flash_attn
import torch
import vllm

model = Path(os.environ["VERIPY_MODEL_PATH"])
checkpoint = Path(os.environ["VERIPY_CHECKPOINT_ROOT"])
probe = checkpoint / ".write-probe"
probe.write_text("ok", encoding="utf-8")
probe.unlink()
payload = {
    "torch": torch.__version__,
    "vllm": vllm.__version__,
    "flash_attn": flash_attn.__version__,
    "cuda_available": torch.cuda.is_available(),
    "torch_cuda": torch.version.cuda,
    "cuda_capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
    "model_readable": model.is_dir() and os.access(model, os.R_OK),
    "checkpoint_writable": os.access(checkpoint, os.W_OK),
    "attention_backend": os.environ.get("VERIPY_ATTENTION_BACKEND") or "FLASH_ATTN",
}
print(json.dumps(payload, sort_keys=True))
if not payload["model_readable"] or not payload["checkpoint_writable"]:
    raise SystemExit(2)
PY
  exit 0
fi

if [[ "$#" -eq 0 ]]; then
  set -- python scripts/run_grpo.py --mode smoke --dry-run
fi
exec "$@"
