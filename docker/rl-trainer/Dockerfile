ARG TORCH_BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel@sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56
FROM ${TORCH_BASE_IMAGE}

ARG VLLM_VERSION=0.8.5
ARG FLASH_ATTN_VERSION=2.7.4.post1
ARG VERL_AGENT_COMMIT=080965f74fb2d6390cc750cdd07928551f22c2aa

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    VLLM_CACHE_ROOT=/cache/vllm \
    VERIPY_MODEL_PATH=/models/VeriPy-Agent-Qwen3-1.7B-SFT \
    VERIPY_CHECKPOINT_ROOT=/checkpoints \
    TORCH_CUDA_ARCH_LIST="8.0;8.9" \
    MAX_JOBS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git ninja-build docker.io \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install "vllm==${VLLM_VERSION}" \
    && pip install --no-build-isolation "flash-attn==${FLASH_ATTN_VERSION}"

WORKDIR /workspace/veripy-agent
RUN pip install --no-deps "git+https://github.com/langfengQ/verl-agent.git@${VERL_AGENT_COMMIT}" \
    && pip install \
       "accelerate==1.14.0" \
       "codetiming==1.4.0" \
       "datasets==5.0.1" \
       "dill==0.4.1" \
       "hydra-core==1.3.6" \
       "pandas==2.3.3" \
       "peft==0.20.0" \
       "pyarrow==25.0.1" \
       "pybind11==3.1.0" \
       "pylatexenc==2.11" \
       "qwen-vl-utils==0.0.14" \
       "ray==2.49.2" \
       "tensordict==0.6.2" \
       "torchdata==0.11.0" \
       "transformers==4.51.1" \
       "wandb==0.19.11" \
       "opentelemetry-api==1.26.0" \
       "opentelemetry-sdk==1.26.0" \
       "opentelemetry-proto==1.26.0" \
       "opentelemetry-semantic-conventions==0.47b0" \
       "opentelemetry-exporter-otlp==1.26.0" \
       "opentelemetry-exporter-otlp-proto-common==1.26.0" \
       "opentelemetry-exporter-otlp-proto-grpc==1.26.0" \
       "opentelemetry-exporter-otlp-proto-http==1.26.0" \
       "protobuf==4.25.9" \
       "requests==2.32.3" \
    && python -m pip check \
    && python -c "import flash_attn, ray, torch, transformers, vllm, wandb; assert transformers.__version__ == '4.51.1'; assert ray.__version__ == '2.49.2'; assert wandb.__version__ == '0.19.11'"

COPY . /workspace/veripy-agent
RUN pip install --no-deps -e .

COPY docker/rl-trainer/entrypoint.sh /usr/local/bin/veripy-rl
RUN chmod 0755 /usr/local/bin/veripy-rl \
    && mkdir -p /models /cache/huggingface /cache/vllm /checkpoints

ENTRYPOINT ["/usr/local/bin/veripy-rl"]
CMD ["--help"]
