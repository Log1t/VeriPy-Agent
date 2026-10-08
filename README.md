# VeriPy-Agent

面向表格数据分析的可验证工具 Agent：让小模型通过多轮 Python 操作解决 CSV/JSON 任务，用独立验证器检查答案与计算证据，再将真实学生轨迹用于教师课程设计和有限预算的 SFT/GRPO 实验。

这是项目收尾后的**核心源码快照**，不是权重发布或一键复现全部历史实验的包。不包含凭据、私有日志、训练数据、模型权重及服务器专用运维脚本。

## 系统设计

```text
隔离任务生成 → SFT 冷启动 → 学生真实多轮轨迹
                              ↓
独立评测 ← 有限步 GRPO ← 沙箱验收 ← 教师课程与语义审查
```

- 8 个合成任务家族：聚合、分组排名、连接、时间序列、缺失值、异常值、相关性、多约束决策。公共任务与隐藏答案分离，按内容和种子隔离数据。
- 四个工具：`inspect_files`、`inspect_table`、`execute_python`、`submit_answer`。逐步保存动作、真实反馈与奖励；Docker 或显式选择的 Linux hardened-process 沙箱拒绝回退到无隔离的宿主 Python。
- 教师 harness：以可核验的学生轨迹为来源，要求结构化设计、来源绑定、输入条件与独立计算验收；验证集/测试反馈不进入课程生成，CPU 验收不代表学习成功。
- 训练适配：Qwen3、LoRA、LLaMA-Factory 与 verl-agent/veRL 接入，统一工具消息和 tokenizer 渲染，绑定数据、父权重及运行收据。
- 上下文软终止：显式启用 `VERIPY_CONTEXT_LIMIT_POLICY=terminate_episode_v1` 后，超限只结束对应 episode，保留真实动作与奖励，兄弟 episode 继续。不截断历史、不虚构提交；bootstrap 超限仍显式拒绝。

## 最终实验结果

2026-10-07：纠错 SFT 父模型，48 道教师课程（16 mean / 32 median）加 32 anchors，8 次 GRPO 更新；80 题各 4 路，共 **320 条真实训练轨迹**。训练正常完成，2 条 episode 触发软终止而未中断训练。

| 同协议评测 | 修复 SFT 父模型 | GRPO 子模型 |
| --- | ---: | ---: |
| 原样代码计算主指标，64 题 | 63/64（98.44%） | 64/64（100%） |
| 整体任务正确，32 题 | 25/32 | 25/32 |
| 有效提交，32 题 | 28/32 | 29/32 |
| 证据一致性，32 题 | 22/32 | 21/32 |

计算主指标要求学生原样代码在原输入及两个答案不同的新输入上均通过，衡量输入敏感的真实计算。两个 32 题批次分别为 31→32、32→32，总增益 **1 题 / 1.56 个百分点**。

这是单 seed、IID 的方向性观测，**不证明统计显著、OOD 泛化、教师优于匹配 fixed 课程或整体能力提高**。整体正确率未提高且证据指标略降，因此修复 SFT 保留为默认模型，子模型仅作探索性候选，不追加训练。主要产出是可审计工程链路，不能将小幅增益表述为显著自进化。

## 代码结构

| 路径（`src/veripy_agent/` 下） | 职责 |
| --- | --- |
| `contracts` | 任务、动作、轨迹、证据契约 |
| `data` | 任务生成、教师 harness、课程与来源隔离 |
| `environment`、`tools` | 沙箱和多轮工具环境 |
| `verifiers`、`rewards` | 独立答案/证据验收和奖励分解 |
| `rollout`、`verl_adapter` | 模型协议及真实 collector 接入 |
| `training`、`evaluation` | SFT admission、训练绑定及评测 |

`scripts/`、`configs/`、`tests/` 提供可复用入口、配置和核心测试。

## CPU 快速开始

需要 Linux 或 WSL、Python 3.11/3.12。Windows 原生不支持 Linux 沙箱接口。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest tests/unit/test_contracts.py tests/unit/test_verifiers.py \
  tests/unit/test_rewards.py tests/unit/test_generator.py \
  tests/unit/test_teacher_harness.py -q

# 小型数据生成，不调用 API，不训练
python scripts/generate_tasks.py filter_aggregate --split validation \
  --seed 42 --count 8 --output data/generated/validation.jsonl
```

发布整理时上述五个测试文件在现有 WSL 环境 **45 passed**，发布快照的 `pytest tests/unit -q` 为 **93 passed**，8 题数据生成成功。依赖未发布实验文件或一次性脚本的测试未纳入快照；完整测试还需要 PyTorch、PyArrow、真实 collector 与沙箱依赖，不能据此声称新机器全部测试/GPU 训练通过。历史服务器完整 CPU 结果为 371 passed、3 skipped、1 个既有冻结 manifest 哈希失败，失败没有被改写为成功。

完整合成数据可通过 `scripts/generate_task_corpus.py --config configs/data/task_corpus.yaml` 生成，输出 manifest 与新锁文件。Parquet 导出另外需要 PyArrow，参数见 `scripts/prepare_verl_dataset.py --help`。生成物不纳入 Git。

## GPU 与教师接入

GPU 栈不是基础安装的一部分。需要单独准备并核验模型、CUDA/PyTorch/vLLM、训练框架、沙箱、数据锁、环境清单和运行配置，不可仅凭静态测试启动训练。

- 资源版本见 `resources.lock.json`，模型路径用 `VERIPY_QWEN3_LOCAL_PATH` 指定；历史摘要不代表当前机器已验收。
- verl-agent pin：`080965f74fb2d6390cc750cdd07928551f22c2aa`；veRL reference pin：`0758489422e8d41a89e6c36d4c477714520f0dcc`。不可随意互换。
- `scripts/run_veripy_grpo.py` 为真实适配入口；`run_grpo.py`、`run_sft.py` 为有验收门槛的启动器，须提供自己的配置与证据。本快照不包含服务器环境或私有收据。
- 教师入口见 `scripts/prepare_curriculum_training.py --help` 和 `run_offline_curriculum.py --help`。凭据留在仓库外，不提交 `.env`、密钥或学生私有日志。
- Docker 材料位于 `docker/`；hardened-process 需要管理员单独配置专用用户、权限及 Linux 安全机制，使用前运行沙箱验收，其隔离不等同于 VM。服务器专属安装脚本未纳入发布。

## 来源与使用边界

复用 [Qwen3](https://github.com/QwenLM/Qwen3)、[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)、[verl-agent](https://github.com/langfengQ/verl-agent) 和 [veRL](https://github.com/volcengine/verl)，而非重写训练框架。版本见资源锁，公开数据候选及许可见 `configs/data/public_sources.yaml`。本仓库不分发上游模型或数据；许可未确认的数据不得自动用于训练或再分发。

目前未授予本仓库独立的开源许可证。公开可读不等于无限制使用或再分发，第三方组件仍受各自许可证约束。
