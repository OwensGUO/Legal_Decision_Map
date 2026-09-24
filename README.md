# 条件法律决策地形学习

本项目把 CAIL-small 单被告主实验和 CMDL-small 多被告扩展实验统一为可追溯的
`CaseUnit`，先抽取结构化法律因素，再生成三类反事实，最后用多任务 QLoRA 模型学习
罪名、刑罚类型、有限期刑月份和因素辅助任务。默认命令不会启动全量生成或训练。

## 1. 目录与安全边界

- 原始数据只读；构建结果默认写到 `outputs/`，不会复制进 Git。
- CAIL-small 使用 `exercise_contest/data_{train,valid,test}.json`。
- CMDL-small 使用 `small/{train,valid,test}_small.jsonl`。
- 只有同时发现 `train_big*`、`valid_big*`、`test_big*` 时，审计才会把 CMDL-big
  标记为可用；当前仅有一个 big 训练分片时会明确拒绝作为正式结果。
- 配置优先级：命令行 `--set` > `LEGAL_LANDSCAPE_...` 环境变量 > YAML。
  例如 `LEGAL_LANDSCAPE_DATA__ROOT=/new/path`。
- 生成客户端只访问 YAML 中显式给出的本地 vLLM 地址；默认是
  `http://127.0.0.1:30000`，没有任何外部模型 API。

主要目录：

```text
configs/                 数据、生成、模型、实验配置
src/legal_landscape/     数据、因素、反事实、模型、训练、评测代码
scripts/                 所有用户入口
tests/                   CPU/纯逻辑测试
docs/superpowers/        设计与实施计划
```

## 2. 服务器首次安装（Conda、Python 3.12、驱动 535）

数据处理、vLLM 反事实生成、训练和评测共用一个 Conda 环境。在项目根目录执行：

```bash
conda create -n legal-landscape python=3.12 -y
conda activate legal-landscape
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip check
```

基础环境不强制安装 Triton/FLA。只有基础 `MODE=smoke` 已通过且需要加速
Qwen3.5 训练时，才尝试：

```bash
python -m pip install -r requirements-optional.txt
CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/probe_gpu_stack.py --limit 4 --probe-fla
```

版本选择依据（`requirements.txt` 顶部注释同步说明）：

- Qwen3.5-9B 与 Qwen3.6-27B 的 `config.json` 都是 `model_type: qwen3_5`
  （线性注意力 Gated DeltaNet 与全注意力混合，附带视觉塔）。vLLM 从 0.17 起、
  Transformers 从 5.2 起才支持该架构。
- Ubuntu 20.04 的 glibc 为 2.31；vLLM 0.20 起的 wheel 需要 glibc 2.35，因此固定
  `vllm==0.19.1`，并按它的测试组合固定 `torch==2.10.0`、`transformers==5.5.3`、
  `peft==0.18.1`、`bitsandbytes==0.49.2`、`accelerate==1.13.0`。
- PyPI 上的 torch 2.10.0 使用 CUDA 12.8 运行时。驱动 535.171.04 原生对应 CUDA 12.2，
  依靠 [CUDA 12.x 次版本兼容](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
  （要求驱动 ≥ 525.60.13）运行；不要安装 cu13x wheel，它们需要 580 以上驱动。
  服务器上的 CUDA Toolkit 12.2 不参与 wheel 选择。
- `flash-linear-attention` 提供 Qwen3.5 训练时 Gated DeltaNet 的 Triton 快速路径，但
  驱动 535 对较新 PTX/JIT 内核存在兼容风险，因此只放在 `requirements-optional.txt`。
  缺少它时使用 Transformers 回退路径，速度可能较慢，但基础安装不会因此失败。

CUDA 12.x 次版本兼容不保证较新 PTX/JIT 一定可用。因此“pip 安装成功”不视为验收成功；
必须以 `probe_gpu_stack.py` 和 vLLM 的真实 JSON 生成探针结果为准。RTX 4090 不支持
vLLM 的 CUDA forward-compatibility 库，请勿设置 `VLLM_ENABLE_CUDA_COMPATIBILITY=1`。

安装后可用第 4 节命令确认 CUDA 真实可用、四卡可见、模型架构可被解析。

## 3. Conda 环境下一键运行

`run.sh` 不会创建或切换环境，必须先激活上一步建立的 Conda 环境。建议第一次先检查将要
执行的全部命令，再跑小规模闭环：

```bash
conda activate legal-landscape
bash run.sh --dry-run
MODE=smoke bash run.sh
```

小规模验证成功后，主实验一键执行：

```bash
bash run.sh
```

默认 `MODE=main` 会依次构建 CAIL-small 和 CMDL-small、以四卡张量并行启动
Qwen3.6-27B vLLM 服务、断点续生成三类反事实、停止 vLLM 释放显存、用四卡训练两套
`B3/seed-42` 和 `M/seed-42` 模型、导出预测，并执行聚类 bootstrap 及 M 对 B3 的
配对评测。输出统一写入
`outputs/`。反事实生成始终带 `--resume`。训练默认每 100 个优化器更新保存一次
`checkpoint-step-*`；若发生中断，再次执行会从时间最新的周期或最终检查点恢复。可用
`CHECKPOINT_EVERY` 调整间隔。最终状态写入 `outputs/run-summary.json`。

完整基线、消融和三随机种子矩阵会产生 78 次训练，耗时和存储开销很大，仅在主流程验证
通过后执行：

```bash
MODE=matrix bash run.sh
```

服务器目录不同可直接覆盖环境变量：

```bash
CAIL_ROOT=/data/datasets/CAIL2018 \
CMDL_ROOT=/data/datasets/CMDL \
QWEN35_PATH=/models/Qwen3.5-9B \
QWEN36_PATH=/models/Qwen3.6-27B \
OUTPUT_ROOT=/data/experiments/legal-landscape \
bash run.sh
```

默认不会修改依赖；只有显式设置 `INSTALL_DEPS=1` 才会安装 `requirements.txt`。
`INSTALL_DEPS=1 INSTALL_FLA=1` 才会额外安装并探测 FLA。`MOCK_GENERATOR=1 MODE=smoke bash run.sh` 可跳过 Qwen3.6 服务，
用于检查数据到训练的控制流；`INFER_MANAGED=0` 表示复用已运行的本地 vLLM 服务，
`INFER_HOST`、`INFER_PORT`、`INFER_GPU_MEMORY` 调整地址与显存占用比例（不使用
`VLLM_*` 前缀，因为 `VLLM_PORT` 等是 vLLM 自身的内部变量）。常用覆盖
参数可运行 `bash run.sh --help` 查看。

## 4. 环境、依赖和四卡检查

```bash
python scripts/audit_environment.py
python scripts/check_requirements.py \
  --model-path /data/chenguo/Qwen3.5-9B
CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/probe_gpu_stack.py --limit 4
```

检查输出应包含 Python 3.12、四张 RTX 4090、CUDA 可见、模型架构和全部关键包。
若只想检查 CPU 工具链：

```bash
python scripts/check_requirements.py --cpu-only
```

## 5. 审计数据

先做不会读大文件的计划检查，再做每个 split 最多 10 条的模式审计：

```bash
python scripts/audit_data.py --config configs/data/cail_small.yaml --dry-run --limit 10
python scripts/audit_data.py --config configs/data/cmdl_small.yaml --dry-run --limit 10

python scripts/audit_data.py --config configs/data/cail_small.yaml --limit 10
python scripts/audit_data.py --config configs/data/cmdl_small.yaml --limit 10
```

本地 Mac 数据路径可以不改 YAML，直接覆盖：

```bash
python scripts/audit_data.py --config configs/data/cail_small.yaml --limit 2 \
  --set data.root=/Users/gilbert/NEFU/datasets/CAIL2018
python scripts/audit_data.py --config configs/data/cmdl_small.yaml --limit 2 \
  --set data.root=/Users/gilbert/NEFU/datasets/CMDL
```

审计会计算源文件路径、字节数、SHA-256 和记录数，因而正式审计会顺序读取完整文件，
但不会写入数据目录。

## 6. 构建小规模处理数据

默认仅打印计划；必须加 `--execute` 才写输出：

```bash
python scripts/build_dataset.py --config configs/data/cail_small.yaml \
  --output-dir outputs/cail_small --limit 100 --dry-run
python scripts/build_dataset.py --config configs/data/cail_small.yaml \
  --output-dir outputs/cail_small --limit 100 --execute

python scripts/build_dataset.py --config configs/data/cmdl_small.yaml \
  --output-dir outputs/cmdl_small --limit 100 --execute
```

去掉 `--limit` 并不安全，因为 CLI 默认仍为 100。正式全量构建时显式给一个覆盖数据规模
的上限，例如 CAIL-small 训练集可使用 `--limit 200000`。输出含 raw、conservative、
strict 三种文本、屏蔽审计、因素、标签、原始路径、case/group ID、manifest 和词表。

源数据中存在少量相同案件跨官方 split 重复的情况。构建器会按 `test > valid > train`
将整个 `group_id` 只保留在最严格的留出集，防止训练集泄漏；`metadata.json` 的
`split_integrity` 会记录跨 split 组数和各 split 删除的单元数。该处理不修改源数据。

## 7. 启动本地 Qwen3.6-27B vLLM 服务

`run.sh` 会自动启动和停止服务；手动调试时在四卡服务器仅监听回环地址：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_USE_FLASHINFER_SAMPLER=0 NCCL_P2P_DISABLE=1 \
vllm serve /data/chenguo/Qwen3.6-27B \
  --served-model-name Qwen3.6-27B \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 \
  --port 30000 \
  --max-model-len 8192 \
  --language-model-only \
  --disable-custom-all-reduce \
  --enforce-eager \
  --gpu-memory-utilization 0.90
curl --fail http://127.0.0.1:30000/health
```

- `--language-model-only` 不加载 Qwen3.6 的视觉塔，本项目只输入文本。
- `VLLM_USE_FLASHINFER_SAMPLER=0` 使用 PyTorch 采样器，避免 FlashInfer 用服务器上的
  CUDA 12.2 工具链即时编译内核。
- 服务名固定为 `Qwen3.6-27B`，请求中的 `model` 字段与之相同。
  请求传入 `chat_template_kwargs.enable_thinking=false` 和 JSON response format。
- 27B BF16 权重约 54 GB，四卡张量并行后每卡约 13.5 GB，其余显存用于 KV 缓存与线性
  注意力状态。默认禁用 P2P、自定义 all-reduce 和 CUDA graph，以优先保证驱动 535
  环境可诊断运行；验证稳定后可设置 `VLLM_ENFORCE_EAGER=0` 对比吞吐。

训练加载器读取顶层及嵌套 `text_config`：纯文本 checkpoint 使用 causal-LM loader；
多模态 `ForConditionalGeneration` checkpoint 使用正确的 image-text loader，再提取语言
骨干并释放/冻结视觉模块。分类前向只返回最后一层状态，不保存所有层隐藏状态。

## 8. 反事实 dry-run、mock 与真实小批量

处理后的 `train.jsonl` 可直接作为输入，脚本会从结构化因素生成
`InterventionSpec`，再交给生成器实现：

```bash
python scripts/generate_counterfactuals.py \
  --config configs/cf/qwen36_27b.yaml \
  --input outputs/cail_small/train.jsonl \
  --output outputs/cf/cail.jsonl \
  --limit 12 --dry-run

python scripts/generate_counterfactuals.py \
  --config configs/cf/qwen36_27b.yaml \
  --input outputs/cail_small/train.jsonl \
  --output outputs/cf/cail_mock.jsonl \
  --limit 12 --mock --execute

python scripts/generate_counterfactuals.py \
  --config configs/cf/qwen36_27b.yaml \
  --input outputs/cail_small/train.jsonl \
  --output outputs/cf/cail.jsonl \
  --limit 12 --resume --execute
```

输出保存原案件 ID、类型、前后因素、changed_fields、rule_id、模型/提示版本、采样参数、
种子、原始响应、重试次数和逐项验证结果。`--resume` 按稳定 generation ID 跳过完成项。

## 9. 训练 dry-run 与 dummy 验证

```bash
python scripts/train_model.py --config configs/model/qwen35_9b_qlora.yaml \
  --experiment M --train-data outputs/cail_small/train.jsonl --limit 8 --dry-run

python scripts/train_model.py --config configs/model/qwen35_9b_qlora.yaml \
  --experiment M --dummy --limit 2 --execute
```

dummy 命令使用微型 CPU backbone 做一次真实反向传播，不读取 Qwen 权重。

## 10. 单 GPU、四 GPU与恢复训练

单 GPU：

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/train_model.py \
  --config configs/model/qwen35_9b_qlora.yaml \
  --experiment M \
  --train-data outputs/cail_small/train.jsonl \
  --counterfactual-data outputs/cf/cail.jsonl \
  --output-dir outputs/train/M-seed42 \
  --limit 200000 --execute
```

四 GPU（每个 Accelerate 进程按 `LOCAL_RANK` 加载到对应卡）：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --multi_gpu --num_processes 4 \
  --mixed_precision bf16 scripts/train_model.py \
  --config configs/model/qwen35_9b_qlora.yaml \
  --experiment M \
  --train-data outputs/cail_small/train.jsonl \
  --counterfactual-data outputs/cf/cail.jsonl \
  --output-dir outputs/train/M-seed42 \
  --limit 200000 --execute
```

恢复：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --multi_gpu --num_processes 4 \
  --mixed_precision bf16 scripts/train_model.py \
  --config configs/model/qwen35_9b_qlora.yaml \
  --experiment M \
  --train-data outputs/cail_small/train.jsonl \
  --counterfactual-data outputs/cf/cail.jsonl \
  --output-dir outputs/train/M-seed42 \
  --resume-from-checkpoint outputs/train/M-seed42/checkpoint-final \
  --limit 200000 --execute
```

通过 `--experiment B0|B1|B2|B3|B4|B5|M|A1|...|A6` 选择实验。B1/B2 必须把
`model.path` 覆盖成本地已有权重，项目不会联网下载。种子用
`--set training.seed=42`、`2026`、`3407` 分别运行。CAIL 使用
`--set model.max_length=4096`，CMDL 使用 `8192`。训练日志在 `train.jsonl` 和
`tensorboard/`，损失 NaN/Inf 或空样本非零会立即写 `loss_diagnostic.json` 并退出。

## 11. 评测并生成 JSON

评测入口消费模型推理得到的 JSONL 预测；三类格式分别对应 static、counterfactual、CMDL：

```bash
python scripts/evaluate_model.py --kind static \
  --input outputs/predictions/static.jsonl \
  --output outputs/results/static.json --limit 100000 \
  --bootstrap-iterations 2000 --bootstrap-seed 42

python scripts/evaluate_model.py --kind counterfactual \
  --input outputs/predictions/counterfactual.jsonl \
  --reference-input outputs/B3/predictions/counterfactual.jsonl \
  --output outputs/results/counterfactual.json --limit 100000 \
  --bootstrap-iterations 2000 --bootstrap-seed 42

python scripts/evaluate_model.py --kind cmdl \
  --input outputs/predictions/cmdl.jsonl \
  --output outputs/results/cmdl.json --limit 100000
```

每个评测结果都会按 `group_id` 整组重采样并输出点估计与 95% 置信区间。指定
`--reference-input` 后，额外输出配对差异、原始 p 值，以及至多五个主要终点的 Holm
校正 p 值；未指定参考模型时不会伪造比较检验。无期、死刑不会进入月份指标。

## 12. 本地验收

```bash
python -m compileall src scripts
pytest -q
PYTHONPATH=src python -m unittest tests.test_models_losses tests.test_training -v
ruff check .
```

若默认 Python 没装 PyTorch，pytest 会明确跳过 Torch 测试；应在安装了 PyTorch 的
Python 3.12 环境运行上面的 unittest 命令。每个脚本均支持 `--help`，生成、构建和训练
均需要显式 `--execute` 才发生重操作。

## 13. 尚需在服务器验证的部分

本地没有 Qwen3.5-9B/Qwen3.6-27B 和 NVIDIA GPU，因此以下内容不能由本地 CPU 验收
替代：真实 `config.json` 权重映射、bitsandbytes NF4、BF16、驱动 535 下的 Triton/PTX、
四卡显存占用、vLLM 张量并行、4096/8192 实际吞吐和多进程断点恢复。先按第 3 节检查，再各跑 `--limit 2` 的
mock/dummy/真实小批量，确认后才扩大规模。
