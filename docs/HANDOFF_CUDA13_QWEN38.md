# CUDA 13 / Qwen3.8 项目交接说明

更新时间：2026-10-01（Asia/Shanghai）

## 当前代码位置

- 工作树：`/Users/gilbert/Desktop/Legal_Decision_Map/.worktrees/cuda13-qwen38`
- 分支：`codex/cuda13-qwen38`
- 基线：`5b71843897a5874e84db53cf513535bf4453da8d`
- 最后一个功能提交：`1b77051d38daf4f5fa2891f65a02bff51a1a1631`
- 主工作目录尚未合并本分支；不要在 `main` 上重复实现这些改动。

## 已完成

1. 依赖环境已经切换到 Ubuntu 24.04 / Python 3.12 / CUDA 13：
   - PyTorch `2.13.0`
   - vLLM `0.30.0`
   - Transformers `5.15.0`
   - Accelerate `1.15.0`
   - PEFT `0.21.1`
   - bitsandbytes `0.50.0`
   - 可选 FLA `0.5.2`，默认不安装、不启用。
2. Qwen3.8-27B BF16 已成为默认反事实生成器；Qwen3.6-27B 保留为显式对照；Qwen3.5-9B 仍是 NF4 QLoRA 训练骨干。
3. 默认服务器路径已经改为 `/data/cguo`：
   - `/data/cguo/Qwen3.8-27B`
   - `/data/cguo/Qwen3.6-27B`
   - `/data/cguo/Qwen3.5-9B`
   - `/data/cguo/datasets/CMDL`
   - `/data/cguo/datasets/CAIL2018`
4. `run.sh` 已支持：
   - 默认物理 GPU `4,5,6,7`；仍可通过 `GPU_IDS` 覆盖；
   - `GENERATOR_MODEL=qwen38|qwen36`；
   - CUDA Graph 默认开启，`VLLM_ENFORCE_EAGER=1` 可安全回退；
   - `P2P_POLICY=auto|enable|disable`，auto 只接受探测结果中的 JSON 布尔值 `true`，其余情况安全禁用 P2P 和 custom all-reduce；
   - prefix caching、仅本机监听、失败即停、生成/训练恢复、vLLM 进程组清理；
   - smoke/main/matrix 三种模式，matrix 仍生成 78 条训练命令。
5. 反事实、训练、检查点、预测和评估结果都按生成器隔离：
   - `outputs/counterfactuals/qwen38|qwen36/`
   - `outputs/runs/qwen38|qwen36/`
6. 已加入持久化生成器溯源与恢复保护：
   - 每个生成器目录保存不可覆盖的 `generator-provenance.json`；
   - 记录 selector、served model name、真实 checkpoint 路径、配置 revision 和有界 checkpoint 指纹；
   - 每条反事实 JSONL 同样保存完整生成器身份；
   - 权重路径/指纹不匹配，或旧产物缺少 manifest 时，会在启动 vLLM、生成或恢复训练前失败，避免静默混用；
   - 指纹读取配置/索引内容与权重文件名/大小，不读取数十 GB 权重内容。
7. P2P 探测会报告所有可见 GPU 的有向 pair，并可将与标准输出一致的 JSON 写入文件。
8. Qwen 请求继续强制 JSON 输出并关闭 thinking。
9. README、长期设计文档、配置默认值和安装说明均已更新；手动 vLLM 示例的 served-model alias 已修正。
10. 最终整分支代码审查已经通过，没有未处理的 Critical 或 Important 缺陷。

## 本地已验证

最后一个功能提交上的验证结果：

- `pytest -q`：127 passed，10 skipped；跳过项来自本机默认 Python 3.13 环境没有 Torch。
- Python 3.12 + 本地 CPU Torch 的训练/损失 unittest：21 passed。
- `ruff check .`：通过。
- `python -m compileall -q src scripts`：通过。
- `bash -n run.sh`：通过。
- smoke/main/matrix dry-run：通过；分别展开 2、4、78 条训练命令。
- Qwen3.6 smoke dry-run：通过。

完成本交接文件后必须重新运行上述验收，以最终输出为准。

## 还没有完成

以下工作只能在目标服务器或由用户决定，不能在本地 macOS 上伪造完成：

1. 尚未把 `codex/cuda13-qwen38` 合并到 `main`，也没有推送或创建 PR。
2. 尚未在目标服务器新建并安装 `legal-landscape-cu130` Conda 环境。
3. 尚未在驱动 `580.173.02`、CUDA `13.0`、RTX 4090（物理编号 4–7）上完成真实依赖加载和 GPU 探测。
4. 尚未加载真实 Qwen3.8/Qwen3.6/Qwen3.5 权重，未验证 TP4 vLLM、BF16、NF4、CUDA Graph、P2P、显存和吞吐。
5. 尚未执行真实端到端 smoke（数据构建、反事实生成、训练、预测、评估）。
6. 尚未执行约 100 个父案例的 Qwen3.8 与 Qwen3.6 质量对照并人工检查法理忠实性。
7. FLA 仍未获准在项目中启用。现有 `--probe-fla` 只验证导入和通用 Triton 加法内核；必须先在目标 GPU 上成功执行与模型相关的真实 FLA 运算。若没有明确收益，可以一直不安装 FLA。
8. 有界 checkpoint 指纹不会发现“原地替换且文件大小完全相同”的权重变化。模型目录应视为不可变；更换权重时使用新路径、revision 或新的 `OUTPUT_ROOT`。

## 下次继续的第一步

本地继续开发：

```bash
cd /Users/gilbert/Desktop/Legal_Decision_Map/.worktrees/cuda13-qwen38
git status --short --branch
git log --oneline --decorate -12
pytest -q
ruff check .
bash -n run.sh
```

服务器部署（代码同步到服务器后）：

```bash
conda create -n legal-landscape-cu130 python=3.12 -y
conda activate legal-landscape-cu130
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip check
MODE=smoke bash run.sh --dry-run
MODE=smoke bash run.sh
```

先保持 FLA 关闭。第一次真实运行前检查：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/probe_gpu_stack.py \
  --limit 4 --output /tmp/legal-landscape-gpu-probe.json
```

如需运行 Qwen3.6 对照：

```bash
GENERATOR_MODEL=qwen36 MODE=smoke bash run.sh
```

完整参数、迁移旧产物的方法及手动生成示例见仓库根目录 `README.md`。
