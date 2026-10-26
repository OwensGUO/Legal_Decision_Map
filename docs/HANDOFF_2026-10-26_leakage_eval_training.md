# 工作交接日志（2026-10-24 ～ 2026-10-26）

本日志记录数据泄露审计、修复、评测提速与训练优化的全部工作，供后续接手者（含其他模型）继续。
用户要求：全程使用简体中文回复；修改代码前遵循 TDD（先写失败测试）；改动前与用户确认方案。

---

## 0. 接手后第一件事（重要）

最新提交 `98f284f Performance improvements and data leakage fixes` **漏掉了新建文件**，而已提交的代码会
`import` 它们，服务器若通过 git 同步会直接报错。需要用户确认后补提交：

```
src/legal_landscape/data/dedup.py            # build.py 依赖
src/legal_landscape/evaluation/prepared.py   # evaluate_model.py、counterfactual_metrics.py、cmdl_case_metrics.py 依赖
scripts/export_analysis_bundle.py
tests/test_dedup.py
tests/test_evaluation_v2.py
tests/test_export_analysis_bundle.py
tests/test_training_distributed.py
tests/test_training_helpers.py
```

`.idea/legal-decision-landscape.iml` 也有改动（IDE 文件，与本工作无关，由用户决定）。

---

## 1. 环境

- 服务器：Ubuntu 20.04、驱动 580.178.04、CUDA 13.0、Python 3.12、RTX 4090（上次运行用 GPU 0-3）。
  项目路径 `/data/cguo/Projects/Legal_Decision_Map`，conda 环境名在服务器上为 `llm`（以服务器为准）。
- 本地（macOS）**没有** GPU/模型权重。本地测试解释器：项目内 `.venv/`（已 gitignore），
  由 `/opt/anaconda3/envs/myenv/bin/python -m venv --system-site-packages .venv` 创建，
  额外装了 `pytest ruff accelerate transformers tensorboard`（未改动 myenv 本身）。
- 本地验收命令：

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider   # 需在沙箱外运行才能跑 pty/DDP 测试
.venv/bin/ruff check .
for m in smoke main matrix; do MODE=$m bash run.sh --dry-run >/dev/null && echo $m ok; done
```

  当前结果：**398 passed**（含 5 个真实双进程 Gloo DDP 测试）；ruff 只剩 3 个**修改前就存在**的 RUF043
  （`tests/test_generator_provenance.py` 两处、`tests/test_training.py` 一处），未改。
- 本地数据：`/Users/gilbert/NEFU/datasets/{CAIL2018,CMDL}`（只读，可用于真实数据抽样验证）。

---

## 2. 时间线与已完成工作

### 2.1 数据泄露审计（17 个问题）——已全部修复

| # | 问题 | 修复 |
|---|---|---|
| 1 | 反事实评测用训练集反事实 | `run.sh` 另为 `test.jsonl` 生成 `counterfactuals/<gen>/heldout/`，经 `--evaluation-counterfactual-data` 只用于评测；训练前断言训练/评测/验证集 `group_id` 不重叠 |
| 2 | 反事实文本未脱敏即作输入 | 生成器只读 `fact_strict`；生成后再脱敏存 `counterfactual_strict`，训练/评测只读该字段 |
| 3 | 提示词泄露金标准、校验器过窄 | `prompts.py` v3 只给目标被告与事实改写指令；`validators.py` 计数式泄露检查（罪名/法条/结论段/刑罚词增加、`××`/`[已屏蔽]` 减少） |
| 4 | strict 脱敏依赖自身标签、括号罪名失效 | `sanitize.py` v2：训练集罪名词典（含括号展开）+ “犯/构成/涉嫌…罪”提示词，所有样本统一 |
| 5 | FACTORS 由罪名词抽取 | `extract.py` 行为线索去掉罪名词；从 strict 文本抽取；反事实用 target_factors |
| 6 | 翻转固定短语捷径 | 校验改为行为线索集合：须含目标行为、去掉原行为 |
| 7 | realized_factors 只是回显 | 从文本重新抽取因素比对 |
| 8 | 缓存键不含父文本 | generation_id 含父文本 SHA-256、提示词版本、脱敏器版本；视图发布逐条校验 |
| 9 | 公诉结论/量刑建议残留 | 从结论标记截掉该句法律结论（保留前面事实） |
| 10 | 生成器记忆 | 计数式校验 + 记录 `novel_ngram_ratio` 供人工抽检 |
| 11 | 词表含测试集 | 词表只来自训练集；未见标签记入 `label_coverage` 与 `unseen_true_*_labels`，评测作额外错误类 |
| 12 | 只做精确去重 | 规范化事实 + MinHash（5-gram，Jaccard≥0.9）跨 split 近重复（`dedup.py`） |
| 13 | A6/B5 用泄露样本 | 泄露类错误对所有实验硬过滤；B5 增强样本屏蔽刑罚/刑期损失 |
| 14 | 金额×1.01 越档 | ±1% 中选不越常见数额档位的方向，否则跳过 |
| 15 | 多罪名父案件翻转 | 只对单罪名父案件翻转 |
| 16 | CMDL 因素案件级 | 抽取时忽略只涉及其他被告的句子（`co_defendants`） |
| 17 | 文档环境版本 | README/requirements 改为 20.04 / 580.178.04 |

用户确认的三个决策：截掉公诉结论段（保留自首/退赃等客观事实）；FACTORS 保留为输入但只从 strict 文本、行为描述抽取；
留出反事实从 test 生成。

真实数据复测（本地）：CAIL 测试 strict 残留金标准罪名 10.0%→0%，量刑建议 4.7%→0%；CMDL 同案不同被告输入
不一致 3.9%→0。

### 2.2 服务器 main 运行分析（修复前代码，2026-10-19～10-25）

- 分析包已下载：`~/Desktop/analysis-bundle-20261026-120122.tar.gz`，解压于本地
  `outputs/analysis/analysis-bundle-20261026-120122/`（gitignore）。用户已**删除服务器 `outputs/`**，
  这是修复前结果唯一副本，作为“修复前基线”。
- 总耗时 164.7h：训练 55.6h；训练集反事实预测 59.4h（泄露评测）；静态预测 3.8h；评测 45.8h。
- 1000 步×8 累积×4 卡=3.2 万样本 ≈ CAIL 21% / CMDL 17% 个 epoch（严重欠训练）。
- 静态结果：CAIL B3/M 罪名 acc .456/.433；CMDL B3/M 罪名 acc .534/.398（M 明显变差）。
- 根因：BCE 对 202/263 类取平均导致梯度稀释；预测塌缩到少数类；无 LR 调度；0.5 阈值+M 多标签过度预测；
  CMDL M 过度预测翻转目标（故意伤害 预测 5525 vs 真实 1501）；旧反事实 CMDL 39.2% 含父案件罪名而父输入不含；
  invariant 损失≈0、response 第 200 步后≈0（捷径满足）。
- 输入长度：CAIL fact_strict 中位 324 字、p99 2037；CMDL 中位 703、11% >4000 字。

### 2.3 评测提速 + 口径 v2——已完成

- `evaluation/prepared.py`：行级量预计算成矩阵，`metrics(group_counts)` 按组权重加权；
  `bootstrap.py` 新增 `fast_bootstrap_metric_set`、`fast_paired_test`（所有终点共享一次抽样序列），
  抽样序列与旧逐行实现逐位一致（有等价性测试）。
- 实测：服务器旧预测全部重评 45.8h → 6.7min；CAIL M 静态+配对 7.7h → 48s，CI 与旧结果完全一致。
- 口径 v2（`metric_version: v2`）：
  1. 多标签宏平均只对完整测试集真实标签中出现过的类别（固定集合）；旧口径 `*_macro_*_all_labels`；刑期多分类也用固定类别。
  2. `flip_accuracy`：目标罪名概率 ≥ 阈值且原罪名 < 阈值；另有 `flip_top1_accuracy`、`flip_margin`。
  3. `invariant_charge_drift`（原 non_target_prediction_drift）+ `invariant_sentence_drift`。
  4. 反事实主要终点：flip_accuracy、invariant_violation_rate、sentence_pairwise_accuracy、invariant_charge_drift。
  5. 量刑方向平局记 0.5；`sentence_rank_tie_rate`、`sentence_pairwise_accuracy_margin`（>1 个月）。
  6. CMDL 增加“罪名+法条”“完整判决”两层被告级/案件级准确率与分层。
- `evaluate_model.py` 新增 `--parent-records`、`--metadata`，为旧反事实预测回填 `parent_charge_indices`。
- 旧预测 v2 重评结果在 `outputs/analysis/v2/`：CAIL M flip_accuracy 新口径 .477（top1 .744）；
  CMDL 宏F1 supported .138/.129 vs all_labels .092/.086；CMDL 完整判决案件级 B3 .112 / M .083。

### 2.4 训练优化——已完成（未在 GPU 上实测）

- 损失：`loss.multilabel_reduction: sum`（罪名、法条、boundary、invariant 对类别求和）。
- 学习率：`warmup_ratio` 线性预热 + 余弦衰减到 `min_learning_rate_ratio`；写入 `train.jsonl` 的 `learning_rate`。
- 步数：`epochs` 个完整 epoch，`max_steps: 0` 不封顶；run.sh 的 main/matrix 默认 `MAX_STEPS=0`。
- 批次：`TokenBudgetBatchSampler`（按字符估长，批大小×最长 ≤ `max_tokens_per_batch`，按批次断点恢复，
  采样器状态格式 `{"epoch","start_batch","seed"}`，与旧检查点不兼容）；推理 `inference_batches` + `shard_batches`
  （各进程批次数相同）+ `accelerate.utils.gather_object` 汇总。
- `pair_every`：每 N 个微批次算一次反事实对（各进程同一计数器，DDP 一致）。
- 验证集：`--validation-data`（run.sh 传 `processed/*/valid.jsonl`），抽 `validation_limit` 条；每个检查点按
  罪名与法条 micro-F1 均值打分写入 `training_progress.json` 的 `validation_score`；只保留最佳与最新周期检查点；
  结束后加载最佳（同分取步数多），再调罪名/法条阈值写 `predictions/thresholds.json` 并用于测试预测。
- 多被告长文本：`focus_fact` 超过 `max_length` 字符时按目标被告选句。
- 配置 `configs/model/qwen35_9b_qlora.yaml`：per_device_batch_size 16、max_tokens_per_batch 8192、
  gradient_accumulation_steps 2、inference 32/16384、lr 2e-4、warmup 0.03、min_lr 0.1、epochs 1、max_steps 0、
  checkpoint_every 200、pair_every 4、validation_limit 2000。

### 2.5 辅助脚本

- `scripts/export_analysis_bundle.py`：在服务器一键打包分析所需文件（只读、标准库、跳过检查点权重/tensorboard/
  完整反事实缓存，精简标签与反事实并抽样 3000 条，自动收集项目根目录 `*.log`）。用法：
  `python scripts/export_analysis_bundle.py [--output-root ...] [--console-log ...]`。

---

## 3. 当前状态与下一步

1. **补提交第 0 节的新文件**（需用户确认），再把项目同步到服务器（不传 `outputs/`、`.venv/`）。
2. 服务器 `outputs/` 已清空，直接用默认 `OUTPUT_ROOT`：

```bash
set -o pipefail
MODE=smoke PROGRESS=never bash run.sh 2>&1 | tee smoke-v2.log; echo "exit=${PIPESTATUS[0]}"
PROGRESS=never bash run.sh 2>&1 | tee main-v2.log; echo "exit=${PIPESTATUS[0]}"
```

3. 本次 run.sh 会重新构建数据，并为**训练集**和**测试集**生成反事实（验证集不生成，只用于静态任务的检查点选择
   与阈值）；生成量约比上次多 15%。中断后重跑会 `--resume` 续用反事实缓存。
4. 运行完成后用 `export_analysis_bundle.py` 打包下载，与 `outputs/analysis/` 中修复前基线对比。

## 4. 已知风险 / 待服务器验证

- **显存**：`max_tokens_per_batch: 8192` 依据 CMDL 单条 8192 已跑通设定，但未实测批量形态；smoke 只用 1024 长度，
  不能证明正式长度可行。OOM 时调小 `max_tokens_per_batch`（如 6144/4096）或 `inference_max_tokens_per_batch` 后重跑。
- **耗时**：整 epoch 的实际时长未知（旧配置下 CAIL M 每步 52s、CMDL M 99s，新配置批量更大、步数更少）。
- **磁盘**：每个检查点约 8.7GB（`save_state` 保存完整模型状态），现已只保留最佳+最新周期检查点+final。
- 近重复阈值 0.9、`novel_ngram_ratio` 的合理范围需在真实生成结果上校准。
- 验证集打分只用罪名与法条 micro-F1（阈值 0.5），未纳入刑期。
- 本地 `outputs/analysis/` 约 13GB，可按需清理（桌面 tar 包仍在）。

## 5. 关键文件索引

- 数据：`src/legal_landscape/data/{sanitize,dedup,build,cmdl,schema}.py`、`factors/extract.py`
- 反事实：`src/legal_landscape/counterfactual/{interventions,prompts,validators,generate,view}.py`、`scripts/generate_counterfactuals.py`
- 训练：`src/legal_landscape/training/train.py`（`run_real_training` 及上方的纯函数助手）、`models/losses.py`、`scripts/train_model.py`
- 评测：`src/legal_landscape/evaluation/{prepared,bootstrap}.py`、`scripts/evaluate_model.py`
- 编排：`run.sh`；配置：`configs/model/qwen35_9b_qlora.yaml`、`configs/cf/qwen38_27b.yaml`
- 测试：`tests/test_{data,dedup,factors_interventions,counterfactual,counterfactual_view,training,training_helpers,training_distributed,evaluation,evaluation_v2,run_script,cli,export_analysis_bundle}.py`
