# 远程部署与真实模型验证

2026-11-26 已通过跳板机连接 GPU 主机，创建独立环境，下载全部资源并启动真实模型。实验结论、失败样例和验证边界见 [实验报告](real-model-results.md)。密码不保存到仓库、脚本或日志。

## 连接与目录

在本机 SSH 配置中设置自己的跳板别名（下例为 `my-jump`），并替换 GPU 主机地址：

```bash
ssh -J my-jump user@GPU_HOST
cd /path/to/ERAG
```

服务器为 Ubuntu 22.04，4 × A800 80GB，驱动 595.71.05。LLM、Embedding、Reranker 分别使用 GPU 0、1、2；扩展实验另启动同权重副本，用于互不竞争的问答任务。没有修改其他项目的环境或停止其他进程。

| 内容 | 项目内目录 |
|---|---|
| Python 3.11 conda 环境 `erag` | `.conda/envs/erag` |
| 固定 commit 的模型和公开数据 | `resources/models/`、`resources/datasets/` |
| 下载完成记录 | `resources/resources.lock.json` |
| 包、模型服务及 UI 缓存 | `.cache/` |
| 服务与安装日志 | `logs/` |
| 实验报告、逐题结果、独立知识库快照 | `data/experiments/` |
| 界面可选知识库 | `data/knowledge_bases/` |

`bash scripts/remote_runtime.sh <命令>` 会设置环境路径和缓存目录，不修改 shell 启动文件。也可以手动激活：

```bash
source /opt/conda/miniconda3/etc/profile.d/conda.sh
export CONDA_ENVS_PATH="$PWD/.conda/envs"
conda activate erag
```

## 从空目录复现安装

先同步仓库源码。以下环境创建只在目标环境不存在时执行，不覆盖已有环境：

```bash
mkdir -p logs .cache .conda/envs
CONDA_PKGS_DIRS="$PWD/.cache/conda" /opt/conda/miniconda3/bin/conda create \
  --prefix "$PWD/.conda/envs/erag" python=3.11 pip -y

bash scripts/remote_runtime.sh python -m pip install torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
bash scripts/remote_runtime.sh python -m pip install \
  -r config/remote-requirements.lock -e '.[all]'
```

`remote-requirements.lock` 是本次 Linux / Python 3.11 运行环境的完整包版本记录，不是跨平台通用锁文件。PyTorch CUDA 12.4 安装方式来自 [官方版本说明](https://docs.pytorch.org/get-started/previous-versions/)。常规核心依赖仍由 `pyproject.toml` 管理，精简部署约束见 `config/remote-constraints.txt`。

实际部署修复了 Xinference 1.13.0 的兼容性问题：FastAPI 0.165.0 虽能启动，但所有 API 因监控中间件读取不存在的路由 `path` 而返回 500。现固定 FastAPI 0.136.3，并在 `local` extra 中限制 `<0.137`。根因与 [Xinference 上游问题 #5063](https://github.com/xorbitsai/inference/issues/5063) 一致。不能仅根据端口监听判断模型服务可用。

并发实测还发现 Transformers 4.57.6 的 KV cache 接口与 Xinference 1.13 的 Qwen3 调度器不兼容，因此固定 Transformers 4.53.3。仅降版本仍不能避免并发解码串扰；当前 Xinference 请求按端点和模型 UID 使用跨进程锁排队，启动模型同时设置 `request_limits=1`。独立副本可并行。不要设置 `max_num_seqs=1`：该版本会进入不受支持的 Qwen3 非批处理路径。锁是此部署栈的兼容措施；需要并发吞吐时应单独验证 vLLM 等服务。

## 下载全部模型和数据

本次 GPU 主机直连 Hugging Face 超时，镜像可用；所有资源仍固定到清单中的 commit，不跟随 `main`：

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  bash scripts/remote_runtime.sh python -m erag.resources --root resources
```

默认下载 Qwen3-8B、Qwen3-Embedding-0.6B、Qwen3-Reranker-0.6B，以及 T2Retrieval 语料、qrels 和 MTRAG 对话，约 19 GB。中断后重复命令可继续下载；仅完整成功的资源才标记为 `complete`。可用 `--only llm embedding reranker` 只下载模型。详见 [资源清单说明](resources.md)。

初次配置：

```bash
test -e .env || cp config/remote.env.example .env
```

## 启动与检查服务

先用 `tmux list-sessions` 和 `curl -f http://127.0.0.1:9997/v1/models` 检查已有服务；以下为新启动步骤，不要重复启动已有 UID。

```bash
tmux new-session -d -s erag-models -c "$PWD" \
  'bash scripts/remote_runtime.sh python -m erag.services.start_xinference >>logs/xinference.log 2>&1'

# 等 /v1/models 返回 HTTP 200 后加载；冷启动可能需要约一分钟。
bash scripts/remote_runtime.sh python -m erag.services.launch_local_models \
  --resources-root resources --llm-gpu 0 --embedding-gpu 1 --reranker-gpu 2

# 真实调用三种模型，检查结构化回答、向量形状/相关性和重排结果。
bash scripts/remote_runtime.sh python -m erag.services.check_models

tmux new-session -d -s erag-webui -c "$PWD" \
  'bash scripts/remote_runtime.sh python -m erag.web.ui >>logs/webui.log 2>&1'
```

UI 和模型 API 均只监听回环地址。客户端建立隧道：

```bash
ssh -N -J my-jump -L 7860:127.0.0.1:7860 user@GPU_HOST
```

随后打开 `http://127.0.0.1:7860`。停止本项目服务时，可分别向 `erag-webui`、`erag-models` tmux 会话发送 `Ctrl-C`；不要停止其他任务。

## 可重复实验

每次使用新的输出目录，避免覆盖原始证据。

```bash
# 自编课程：真实增强建库、四种检索消融、批处理、历史/TCT 对照、资料不足问题。
bash scripts/remote_runtime.sh python -m benchmarks.validate_real_models \
  --env-file .env --output-dir data/experiments/course-new --repeats 3

# 比较回答逻辑时复用同一索引；验证语料内容和分块必须一致。
bash scripts/remote_runtime.sh python -m benchmarks.validate_real_models \
  --env-file .env --output-dir data/experiments/course-replay-new --repeats 3 \
  --reuse-index data/experiments/real-model-run-001/knowledge_bases/validation-course

# 公共相关性标签：固定抽样，不依据模型分数选择问题。
bash scripts/remote_runtime.sh python -m benchmarks.prepare_t2_fixture \
  --output-dir data/fixtures/t2-new --queries 32 --distractors 64 --seed 27
bash scripts/remote_runtime.sh python -m benchmarks.validate_real_models \
  --env-file .env --fixture-dir data/fixtures/t2-new --retrieval-only \
  --output-dir data/experiments/t2-new --repeats 3
```

检索路径在相同索引、模型、候选预算下预热后随机交错执行，每种方法重复三次。`legacy_and` 重现旧检索算法，不代表整个历史项目的完整运行。重复测量不增加独立问题数量；延迟只代表本次硬件和服务状态。服务失败使实验失败，不能偷偷降级后继续统计。

课程验证现在包含六轮问题；其中后三轮是在发现条件泛化问题后新增的回归样例，不是独立教学测试集。T2 子集仅选取全部相关文档都不超过 650 字的查询，保留全部正例并加入随机干扰文档，属于缩小语料的诊断实验，不能与完整 C-MTEB 排行榜成绩比较。

`report.json` 保存配置、代码/语料哈希、资源版本、快照、逐题指标、耗时、逻辑模型调用次数和完整回答。调用次数不包含 SDK 内部 HTTP 重试。`completed_needs_answer_review` 只表示收集完成，必须继续人工核对事实、指代、条件与引用；编号合法或 TCT 通过不等于语义正确。

## 已失败增强任务的恢复

当前归档的公开数据失败实验已使用下列命令补生成两个块，原实验不被覆盖：

```bash
bash scripts/remote_runtime.sh python -m benchmarks.repair_enrichment --env-file .env --failed-report data/experiments/t2-dev-seed27-run-001/report.json --indices 100 137 --output-dir data/experiments/repair-new
```

索引是失败日志中的零起始序号；恢复脚本校验降级摘要特征并要求覆盖全部可识别降级块。它重新调用真实模型，保留其他块，再创建新向量快照；不是删除告警或把降级结果标为成功。新建任务已自带一次 schema 修正重试，最终失败仍明确记录。

远程演示默认关闭 TCT（`ERAG_TCT_ENABLED=false`），因为本次对照没有发现答案质量收益，且出现额外的偏题补查。历史和重排仍开启；需要研究反馈检索时可显式开启 TCT。实验脚本始终显式对照开/关两种模式，不受演示默认值影响。
