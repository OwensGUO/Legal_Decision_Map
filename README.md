# 条件法律决策地形学习

本项目把 CAIL-small 单被告主实验和 CMDL-small 多被告扩展实验统一为可追溯的
`CaseUnit`，先抽取结构化法律因素，再生成三类反事实，最后用多任务 QLoRA 模型学习
罪名、被告级定罪法条、刑罚类型、有限期刑月份和因素辅助任务。法条头以软罪名概率为
条件，刑期头进一步使用软法条概率；默认命令不会启动全量生成或训练。

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

## 2. 服务器首次安装（Conda、Python 3.12、CUDA 13）

数据处理、vLLM 反事实生成、训练和评测共用一个 Conda 环境。在项目根目录执行：

```bash
conda create -n legal-landscape-cu130 python=3.12 -y
conda activate legal-landscape-cu130
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip check
MODE=smoke bash run.sh
```

目标服务器为 Ubuntu 24.04 x86-64、NVIDIA 驱动 580.173.02、CUDA 13.0，默认分配
四张 RTX 4090（物理编号 `4,5,6,7`）。请新建环境，不要原地升级旧环境。

基础环境不安装可选的 `flash-linear-attention[cuda]==0.5.2`。基础 `MODE=smoke`
通过后，需要排查 FLA 兼容性时，只在独立验证环境中安装并执行以下初步检查：

```bash
python -m pip install -r requirements-optional.txt
CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/probe_gpu_stack.py --limit 4 --probe-fla
```

`requirements.txt` 是依赖的权威列表：`torch==2.13.0`、`vllm==0.30.0`、
`transformers==5.15.0`、`accelerate==1.15.0`、`peft==0.21.1`、
`bitsandbytes==0.50.0`，使用 CUDA 13 的预构建 wheel，不隐式源码编译 vLLM。
`--probe-fla` 仅导入 `fla` 并运行通用 Triton 加法内核，属于初步兼容性检查，
没有执行 FLA 注意力操作。安装成功或该检查通过都不批准项目使用 FLA。在目标 GPU
上成功执行与本项目模型相关的真实 FLA 操作之前，项目环境应保持 FLA 未安装/禁用，
使用 Transformers 回退实现；本地测试和当前自动探针均不能满足该实际内核门槛。

“pip 安装成功”不视为验收成功；必须通过精确版本检查、`pip check`、CUDA 内核、
BF16、NF4 及 vLLM 的真实非思考 JSON 生成探针。请勿设置
`VLLM_ENABLE_CUDA_COMPATIBILITY=1`。

安装后可用第 4 节命令确认 CUDA 真实可用、四卡可见、模型架构可被解析。

## 3. Conda 环境下一键运行

`run.sh` 不会创建或切换环境，必须先激活上一步建立的 Conda 环境。建议第一次先检查将要
执行的全部命令，再跑小规模闭环：

```bash
conda activate legal-landscape-cu130
bash run.sh --dry-run
MODE=smoke bash run.sh
```

`MODE=smoke` 默认将 CAIL 和 CMDL 的训练最大长度限制为 1024，以降低 Qwen3.5-9B
在普通案件、父案件和反事实案件三路前向计算时的峰值显存。显存更小的机器可进一步覆盖：

```bash
SMOKE_MAX_LENGTH=768 MODE=smoke bash run.sh
```

`MODE=main` 和 `MODE=matrix` 仍使用正式长度：CAIL 4096、CMDL 8192；
`SMOKE_MAX_LENGTH` 不会改变正式实验配置。

运行进度默认采用 `PROGRESS=auto`：`stderr` 连接 TTY 时使用动态进度，显示任务计数、
耗时、吞吐率和有观测后计算的剩余时间；将 `stderr` 重定向到日志时改为定期输出完整文本行
（约每 30 秒一次），同样包含吞吐率与预计剩余时间，
避免光标控制字符。仅重定向 `stdout` 不会关闭动态进度。
也可用 `PROGRESS=always` 强制动态显示，或用 `PROGRESS=never` 始终输出适合日志的
文本进度。设置 `NO_COLOR=1` 可保留进度显示并关闭颜色：

```bash
# 默认：stderr 是终端时动态显示，stderr 重定向到日志时定期输出文本
MODE=smoke PROGRESS=auto bash run.sh

# 强制使用适合日志的纯文本进度
MODE=smoke PROGRESS=never bash run.sh

# 保留进度显示，但关闭颜色
NO_COLOR=1 MODE=smoke bash run.sh
```

一键流程依次显示九个顶层阶段：`environment`、`validate generator provenance`、
`audit data`、`build datasets`、`start vLLM`、`generate counterfactuals`、
`stop vLLM`、`train and export predictions`、`evaluate`。每个阶段都有 `1/9` 至
`9/9` 的编号、开始消息和带耗时的完成消息。阶段内部的进度按实际工作量计数，包括
数据分组与各 split 构建、反事实请求、训练优化器更新、预测批次和 bootstrap 重采样。
反事实断点续跑会将已完成请求计入初始进度；训练从检查点恢复时，初始计数从已完成的
优化器更新开始。分布式训练和预测只由主进程显示进度，避免多卡重复输出。
吞吐率只统计本次运行新增的工作量；恢复计数不会算入速度。中间更新的动态重绘最多每秒 10 次，
任务开始、失败和完成立即显示。窄终端优先保留完整计数与状态，逐步隐藏次要字段，
极窄时使用简短文本；非 UTF-8 环境或输出流统一使用 ASCII。已结束任务的耗时和动画冻结。

普通运行的阶段消息和内部进度发往 `stderr`，各 Python 子命令的 JSON `stdout` 不会被进度条控制字符污染。
一键脚本的 `stdout` 还包含 requirement/info 消息和多个子命令输出，不能作为单个 JSON 文档解析；
需要正式结果时，应读取 `OUTPUT_ROOT` 下具体的 JSON/JSONL 文件。`--dry-run` 的阶段和模拟命令会
打印到 `stdout`，且不会使用动态控制字符。例如，纯文本日志中的顶层消息形如：

```text
[phase 1/9] environment
[phase 1/9] environment completed in 0s
[phase 2/9] validate generator provenance
```

启动 vLLM 时会持续显示已等待时间和超时上限；服务就绪时间不可预知，因此该阶段
刻意不显示虚构的完成百分比。

小规模验证成功后，主实验一键执行：

```bash
bash run.sh
```

默认 `MODE=main` 会依次构建 CAIL-small 和 CMDL-small、以四卡张量并行启动
Qwen3.8-27B BF16 vLLM 服务、断点续生成三类反事实、停止 vLLM 释放显存、用四卡训练两套
`B3/seed-42` 和 `M/seed-42` 模型、导出预测，并执行聚类 bootstrap 及 M 对 B3 的
配对评测。输出统一写入 `OUTPUT_ROOT`（默认 `outputs/`）：处理数据共享
`processed/{cail_small,cmdl_small}/`；反事实隔离为
`counterfactuals/{qwen38,qwen36}/{cail,cmdl}.jsonl`（保留历史请求的主缓存）。生成完成或续用
缓存后，脚本会原子发布 `counterfactuals/<generator>/views/<mode>/{cail,cmdl}.jsonl`，
仅包含本次请求；训练只读取该视图，不读取整个主缓存。视图报告中的 `rows/valid/excluded`
分别表示本次记录数、验证通过数、未选入的历史记录数。生成与训练使用相同的处理后父案件
行数限制，避免 CMDL 多被告展开或长文本跳过造成范围不一致。重复请求记录、损坏 JSON、
生成器/干预不一致或缺失请求会在发布前报错，不会静默筛除，也不会覆盖旧视图。
训练、检查点、预测和评测隔离为
`runs/{qwen38,qwen36}/{cail,cmdl}/{experiment}/seed-{seed}/`，避免跨生成器恢复或混用。
反事实生成始终带 `--resume`。训练默认每 100 个优化器更新保存一次
`checkpoint-step-*`；main/matrix 若发生中断，再次执行会从时间最新的周期或最终检查点恢复。可用
`CHECKPOINT_EVERY` 调整间隔。最终状态写入 `${OUTPUT_ROOT}/run-summary.json`，记录所选
生成器；该根目录文件及诊断日志代表最近一次运行，按生成器隔离的结果保存在上述目录。

smoke 结果单独写入 `runs/<generator>/smoke/length-<SMOKE_MAX_LENGTH>/`，每次从头训练，
不恢复检查点。已有同一实验/种子的结果会移到带 `.previous-<时间>-<PID>` 后缀的相邻目录，
可恢复，不删除。M 等类型化训练若过滤后没有可用反事实对，会在加载模型前明确中止。
本次修改前的 main/matrix 检查点以主缓存文件为输入，不能直接恢复到新的视图输入；首次
升级正式实验请使用新的 `OUTPUT_ROOT`，保留旧结果。新的检查点仍严格校验输入身份。

服务器上传代码后，可在 **Bash** 中运行并保存完整日志（无需服务器安装 Git）：

```bash
set -o pipefail
SMOKE_MAX_LENGTH=512 MODE=smoke PROGRESS=never bash run.sh \
  2>&1 | tee smoke-view-512.log
run_status=${PIPESTATUS[0]}
printf 'exit_status=%s\n' "$run_status"
```

`PIPESTATUS` 必须紧接管道读取。旧主缓存无需删除；若生成器身份不一致，仍应遵循下面的
来源保护要求使用新输出目录，不能绕过校验。

持久生成器身份保存在 `counterfactuals/{qwen38,qwen36}/generator-provenance.json`，
包含 schema 版本、生成器选择、实际 HTTP 模型名、解析后的本地 checkpoint 路径、配置版本、
后端和轻量指纹。启动 vLLM 前和训练前会核验该身份；仅完全一致时允许恢复。
如果同一选择器下换了 checkpoint 路径、配置版本、服务别名或指纹，或已有反事实/下游结果
却没有 manifest，脚本会中止。请使用新的 `OUTPUT_ROOT`，或在确认来源后有意识地迁移/清理
该生成器的反事实与全部下游结果；不要仅删除 manifest 或给旧数据补写当前身份。
空命名空间的 manifest 原子创建，已有 manifest 不会覆盖。

指纹对 checkpoint 顶层的 JSON 配置/索引、tokenizer 与 chat-template 文件内容做 SHA-256，
并纳入 `.safetensors` / `pytorch_model*.bin` 权重文件名和大小；元数据合计上限为 64 MiB。
缺少配置、权重、索引引用的分片或超出上限都会明确报错。它不读取大体积权重内容，
因此不是整套模型的密码学完整性证明，也无法发现大小不变的权重替换。请将 checkpoint
作为不可变输入，改变权重时使用新路径/明确版本和新的输出根目录。

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
QWEN38_PATH=/models/Qwen3.8-27B \
OUTPUT_ROOT=/data/experiments/legal-landscape \
bash run.sh
```

默认不会修改依赖；只有显式设置 `INSTALL_DEPS=1` 才会安装 `requirements.txt`。
`INSTALL_DEPS=1 INSTALL_FLA=1` 会额外安装 FLA；`INSTALL_FLA=1` 会执行上述 FLA
导入/通用 Triton 初步检查。该开关不会验证 FLA 注意力内核；在目标 GPU 的模型相关
FLA 操作成功之前，不要为项目设置此开关或安装 FLA。
`MOCK_GENERATOR=1 MODE=smoke OUTPUT_ROOT=outputs/mock bash run.sh` 可跳过生成服务，
用于检查数据到训练的控制流；`INFER_MANAGED=0` 表示复用已运行的本地 vLLM 服务，
`INFER_HOST`、`INFER_PORT`、`INFER_GPU_MEMORY` 调整地址与显存占用比例（不使用
`VLLM_*` 前缀，因为 `VLLM_PORT` 等是 vLLM 自身的内部变量）。常用覆盖
参数可运行 `bash run.sh --help` 查看。
mock 身份明确记录 `mock-v1` 与实现指纹，和真实模型不兼容，必须使用独立输出根目录。
复用已有服务时，操作者须确认它加载的 checkpoint 与所选路径一致；本地指纹无法证明
外部服务实际加载了哪些权重。

默认 `GENERATOR_MODEL=qwen38`；显式比较使用
`GENERATOR_MODEL=qwen36 MODE=smoke bash run.sh`。默认路径如下，均可用对应变量覆盖：

| 变量 | 默认路径 |
|---|---|
| `QWEN38_PATH` | `/data/cguo/Qwen3.8-27B` |
| `QWEN36_PATH` | `/data/cguo/Qwen3.6-27B` |
| `QWEN35_PATH` | `/data/cguo/Qwen3.5-9B` |
| `CAIL_ROOT` | `/data/cguo/datasets/CAIL2018` |
| `CMDL_ROOT` | `/data/cguo/datasets/CMDL` |

Qwen3.8 BF16 是主生成器，Qwen3.6 BF16 用于明确标注的比较，Qwen3.5-9B 保持 NF4
QLoRA 训练骨干。正式全量生成前，固定约 100 个父案件，用相同种子分别生成 Qwen3.6
和 Qwen3.8 结果，报告 JSON 有效率、全部验证通过率、重试、目标因素实现、非目标漂移
及近重复拒绝率。该质量门槛需要人工/实验核验，不是 `MODE=smoke` 自动完成的步骤，
也不按下游测试表现重新选择主模型。

## 4. 环境、依赖和四卡检查

```bash
python scripts/audit_environment.py
python scripts/check_requirements.py \
  --model-path /data/cguo/Qwen3.5-9B
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
  --output-dir outputs/processed/cail_small --limit 100 --dry-run
python scripts/build_dataset.py --config configs/data/cail_small.yaml \
  --output-dir outputs/processed/cail_small --limit 100 --execute

python scripts/build_dataset.py --config configs/data/cmdl_small.yaml \
  --output-dir outputs/processed/cmdl_small --limit 100 --execute
```

去掉 `--limit` 并不安全，因为 CLI 默认仍为 100。正式全量构建时显式给一个覆盖数据规模
的上限，例如 CAIL-small 训练集可使用 `--limit 200000`。输出含 raw、conservative、
strict 三种文本、屏蔽审计、因素、标签、原始路径、case/group ID、manifest 和词表。

CAIL 的定罪法条来自 `meta.relevant_articles`，CMDL 来自目标被告各项
`outcomes[].judgment[].article`，统一规范为 `criminal_law:<条>[:<款>]`。CMDL 顶层
`relevant_articles` 是案件级信息，不会复制为每个被告的监督标签。训练优先使用
`fact_strict`，避免原文中的罪名和法条引用泄漏到预测任务。

源数据中存在少量相同案件跨官方 split 重复的情况。构建器会按 `test > valid > train`
将整个 `group_id` 只保留在最严格的留出集，防止训练集泄漏；`metadata.json` 的
`split_integrity` 会记录跨 split 组数和各 split 删除的单元数。该处理不修改源数据。

## 7. 启动本地 Qwen3.8-27B BF16 vLLM 服务

`run.sh` 会自动启动和停止服务；手动调试时在四卡服务器仅监听回环地址：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_USE_FLASHINFER_SAMPLER=0 NCCL_P2P_DISABLE=1 \
vllm serve /data/cguo/Qwen3.8-27B \
  --served-model-name Qwen3.8-27B \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 \
  --port 30000 \
  --max-model-len 8192 \
  --language-model-only \
  --enable-prefix-caching \
  --disable-custom-all-reduce \
  --gpu-memory-utilization 0.90
curl --fail http://127.0.0.1:30000/health
```

- 手动示例使用安全 P2P 回退；推荐用 `run.sh` 自动应用探针决策。
- `--language-model-only` 只加载语言部分，本项目只输入文本。
- `VLLM_USE_FLASHINFER_SAMPLER=0` 使用 PyTorch 采样器，避免额外采样器 JIT 编译。
- 服务名默认 `Qwen3.8-27B`，选择 Qwen3.6 时为 `Qwen3.6-27B`，可由
  `INFER_MODEL_NAME` 覆盖；请求中的 `model` 字段必须与服务名相同。
  请求传入 `chat_template_kwargs.enable_thinking=false` 和 JSON response format。
- 27B BF16 权重约 54 GB，四卡张量并行后每卡约 13.5 GB，其余显存用于 KV 缓存与线性
  注意力状态，实际显存和吞吐需要服务器验证。
- `P2P_POLICY=auto`（默认）只信任 `gpu-probe.json` 中
  `peer_access.all_pairs_accessible` 的 JSON 布尔值 `true`；缺失、格式错误、数字、
  字符串或 `false` 都使用 `NCCL_P2P_DISABLE=1` 和 `--disable-custom-all-reduce`
  回退。`disable` 强制回退，`enable` 显式绕过自动决策启用 P2P/custom all-reduce。
  dry-run 的 auto 先显示安全回退，真实决策取决于实际 GPU 探针。
- CUDA Graph 默认开启（`VLLM_ENFORCE_EAGER=0`），前缀缓存默认开启；诊断时设置
  `VLLM_ENFORCE_EAGER=1` 添加 `--enforce-eager`。服务启动失败会退出，不自动切换配置。

训练加载器读取顶层及嵌套 `text_config`：纯文本 checkpoint 使用 causal-LM loader；
多模态 `ForConditionalGeneration` checkpoint 使用正确的 image-text loader，再提取语言
骨干并释放/冻结视觉模块。分类前向只返回最后一层状态，不保存所有层隐藏状态。

## 8. 反事实 dry-run、mock 与真实小批量

处理后的 `train.jsonl` 可直接作为输入，脚本会从结构化因素生成
`InterventionSpec`，再交给生成器实现：

```bash
python scripts/generate_counterfactuals.py \
  --config configs/cf/qwen38_27b.yaml \
  --input outputs/processed/cail_small/train.jsonl \
  --output outputs/counterfactuals/qwen38/cail.jsonl \
  --limit 12 --dry-run

python scripts/generate_counterfactuals.py \
  --config configs/cf/qwen38_27b.yaml \
  --input outputs/processed/cail_small/train.jsonl \
  --output outputs/mock/counterfactuals/qwen38/cail.jsonl \
  --artifact-root outputs/mock/runs/qwen38 \
  --limit 12 --mock --execute

python scripts/generate_counterfactuals.py \
  --config configs/cf/qwen38_27b.yaml \
  --input outputs/processed/cail_small/train.jsonl \
  --output outputs/counterfactuals/qwen38/cail.jsonl \
  --artifact-root outputs/runs/qwen38 \
  --set generator.checkpoint_path=/data/cguo/Qwen3.8-27B \
  --set generator.model_path=Qwen3.8-27B \
  --limit 12 --resume --execute
```

输出保存原案件 ID、类型、前后因素、changed_fields、rule_id、模型/提示版本、采样参数、
种子、原始响应、重试次数和逐项验证结果。每行的 `generator_identity` 保存完整生成器身份，
`model_revision` 保存非占位的 checkpoint 指纹；配置中的 `local` 只作为
`configured_revision` 保留。`--resume` 在身份核验后按稳定 generation ID 跳过完成项。
真实生成命令的 `generator.model_path` 会成为 HTTP 请求的 `model` 字段，因此必须与
第 7 节 `--served-model-name` 的别名一致；`generator.checkpoint_path` 独立记录本地目录，
两者都必须与实际服务一致。切换到 Qwen3.6 时同时改用对应配置、checkpoint、服务别名和
`qwen36` 输出目录。配置和身份均遵循 YAML → `LEGAL_LANDSCAPE_*` 环境变量 → `--set` 优先级。

手工执行时默认 manifest 位于输出 JSONL 的同一目录；`--artifact-root` 可重复，指定需要
共同保护的下游目录。`--provenance-manifest` 可以显式指定同一 manifest。只有新空目录
可初始化；`--dry-run` 不读取 checkpoint 或创建 manifest，也不证明已有输出可以安全恢复。
启动手工 vLLM 服务前，可先执行同一轻量校验：

```bash
python scripts/check_generator_provenance.py \
  --config configs/cf/qwen38_27b.yaml \
  --manifest outputs/counterfactuals/qwen38/generator-provenance.json \
  --artifact-root outputs/runs/qwen38 \
  --set generator.checkpoint_path=/data/cguo/Qwen3.8-27B \
  --set generator.model_path=Qwen3.8-27B
```

手工恢复训练/预测/评测前也应执行该校验。`run.sh` 自动完成这些门槛；直接调用训练脚本
仍通过已有训练输入文件哈希核验检查点，但不会替你推断生成器当前选择。

## 9. 训练 dry-run 与 dummy 验证

```bash
python scripts/train_model.py --config configs/model/qwen35_9b_qlora.yaml \
  --experiment M --train-data outputs/processed/cail_small/train.jsonl --limit 8 --dry-run

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
  --train-data outputs/processed/cail_small/train.jsonl \
  --counterfactual-data outputs/counterfactuals/qwen38/cail.jsonl \
  --output-dir outputs/runs/qwen38/cail/M/seed-42/training \
  --limit 200000 --execute
```

四 GPU（每个 Accelerate 进程按 `LOCAL_RANK` 加载到对应卡）：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --multi_gpu --num_processes 4 \
  --mixed_precision bf16 scripts/train_model.py \
  --config configs/model/qwen35_9b_qlora.yaml \
  --experiment M \
  --train-data outputs/processed/cail_small/train.jsonl \
  --counterfactual-data outputs/counterfactuals/qwen38/cail.jsonl \
  --output-dir outputs/runs/qwen38/cail/M/seed-42/training \
  --limit 200000 --execute
```

恢复：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --multi_gpu --num_processes 4 \
  --mixed_precision bf16 scripts/train_model.py \
  --config configs/model/qwen35_9b_qlora.yaml \
  --experiment M \
  --train-data outputs/processed/cail_small/train.jsonl \
  --counterfactual-data outputs/counterfactuals/qwen38/cail.jsonl \
  --output-dir outputs/runs/qwen38/cail/M/seed-42/training \
  --resume-from-checkpoint outputs/runs/qwen38/cail/M/seed-42/training/checkpoint-final \
  --limit 200000 --execute
```

通过 `--experiment B0|B1|B2|B3|B4|B5|M|A1|...|A6` 选择实验。B1/B2 必须把
`model.path` 覆盖成本地已有权重，项目不会联网下载。种子用
`--set training.seed=42`、`2026`、`3407` 分别运行。CAIL 使用
`--set model.max_length=4096`，CMDL 使用 `8192`。训练日志在 `train.jsonl` 和
`tensorboard/`，损失 NaN/Inf 或空样本非零会立即写 `loss_diagnostic.json` 并退出。
静态预测文件同时保存罪名和法条的真实/预测多热向量、概率与可读标签，以及刑罚类型和
连续月份预测。

## 11. 评测并生成 JSON

评测入口消费模型推理得到的 JSONL 预测；三类格式分别对应 static、counterfactual、CMDL：

```bash
python scripts/evaluate_model.py --kind static \
  --input outputs/runs/qwen38/cail/M/seed-42/predictions/static.jsonl \
  --output outputs/runs/qwen38/cail/M/seed-42/results/static.json --limit 100000 \
  --bootstrap-iterations 2000 --bootstrap-seed 42

python scripts/evaluate_model.py --kind counterfactual \
  --input outputs/runs/qwen38/cail/M/seed-42/predictions/counterfactual.jsonl \
  --reference-input outputs/runs/qwen38/cail/B3/seed-42/predictions/counterfactual.jsonl \
  --output outputs/runs/qwen38/cail/M/seed-42/results/counterfactual.json --limit 100000 \
  --bootstrap-iterations 2000 --bootstrap-seed 42

python scripts/evaluate_model.py --kind cmdl \
  --input outputs/runs/qwen38/cmdl/M/seed-42/predictions/static.jsonl \
  --output outputs/runs/qwen38/cmdl/M/seed-42/results/cmdl.json --limit 100000
```

每个评测结果都会按 `group_id` 整组重采样并输出点估计与 95% 置信区间。指定
`--reference-input` 后，额外输出配对差异、原始 p 值，以及至多五个主要终点的 Holm
校正 p 值；未指定参考模型时不会伪造比较检验。无期、死刑不会进入月份指标。

静态评测对罪名、定罪法条和最终刑期类别都强制输出 `accuracy`、
`macro_precision`、`macro_recall`、`macro_f1`，字段分别使用 `charge_`、`article_`、
`sentence_` 前缀。罪名和法条的 accuracy 是整组标签完全一致；刑期类别包括死刑、无期、
拘役、管制、免刑、未知，以及有期徒刑 `0-6`、`7-12`、`13-24`、`25-36`、`37-60`、
`61-120`、`121+` 月。评测还输出多标签 micro 指标、Hamming accuracy，以及单罪名有期
徒刑样本的 MAE、log-MAE 和三个月容差准确率。

## 12. 本地验收

```bash
python -m compileall -q src scripts
pytest -q
PYTHONPATH=src /opt/anaconda3/envs/myenv/bin/python -m unittest tests.test_models_losses tests.test_training -v
ruff check .
bash -n run.sh
MODE=smoke bash run.sh --dry-run
MODE=main bash run.sh --dry-run
MODE=matrix bash run.sh --dry-run
```

若默认 Python 没装 PyTorch，pytest 会明确跳过 Torch 测试；应在安装了 PyTorch 的
Python 3.12 环境运行上面的 unittest 命令；`/opt/anaconda3/envs/myenv/bin/python` 是本地
验收解释器，服务器可用已激活环境的 `python`。若该本地解释器缺失，必须如实记录，并用
可用项目解释器补充同一 unittest 检查。matrix dry-run 应打印 78 条训练命令。
每个脚本均支持 `--help`，生成、构建和训练
均需要显式 `--execute` 才发生重操作。

## 13. 尚需在服务器验证的部分

本地 macOS CPU 测试及 dry-run 不能证明目标 CUDA 13 硬件行为。本地没有
Qwen3.5-9B/Qwen3.6-27B/Qwen3.8-27B 和 NVIDIA GPU，以下部分必须在目标服务器
核验：wheel 安装与版本兼容、真实 `config.json` 权重映射、bitsandbytes NF4、BF16、
Triton/FLA 实际内核、四卡 P2P 与显存、CUDA Graph、vLLM TP4 JSON 生成、4096/8192
实际吞吐和多进程断点恢复。先按第 4 节检查，再运行有界 mock 和真实生成批次、
`MODE=smoke` 闭环及约 100 个父案件的生成器质量门槛，全部通过后才扩大规模。
