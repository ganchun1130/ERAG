# ERAG · 知识小站

面向教学资料的 RAG 问答项目：上传教材、构建知识库、带来源回答，并保留可继续追问的历史对话。支持本地 Qwen3 和 OpenAI-compatible 模型服务。

界面采用薄荷绿与米白配色：左侧管理会话、知识库和检索设置，右侧聊天。支持新建、搜索历史、重命名、删除与撤销、Markdown 导出、复制、重新生成和停止生成。使用与备份方式见 [聊天工作台](docs/chat-workspace.md)。

## 快速开始

Python 3.11+。在仓库根目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,resources]'
cp .env.example .env
# 编辑 .env：填写 LLM 地址/模型/API Key，并配置 Embedding 与 Reranker 服务。
erag-ui
```

打开 <http://127.0.0.1:7860>。默认仅监听本机；远程 GPU 通过 SSH 隧道使用，见 [远程部署](docs/remote-validation.md)。

也可以使用命令行：

```bash
erag build --name 课程笔记 /path/to/教材.pdf
erag list
erag ask --kb 课程笔记 '两种评价方法有什么区别？'
erag ask --kb 课程笔记 --json '它们分别适合什么场景？'
```

命令行 `ask` 是单次请求；持续历史由网页保存，Python API 可显式传入 `history`。

## 下载与本地模型

全部模型和公共评测数据共用一份 [固定版本清单](erag/resource-manifest.json)：Qwen3-8B、Qwen3-Embedding-0.6B、Qwen3-Reranker-0.6B、T2Retrieval、qrels 和 MTRAG，合计约 19 GB。

```bash
erag-download --dry-run                 # 查看版本、体积与目录
erag-download                           # 下载全部六项资源
erag-download --only embedding reranker # 按需下载
```

默认保存到当前目录的 `resources/`；支持 `--root /path/to/resources`。固定 commit、校验 MTRAG SHA256、保存下载完成状态。完整目录与恢复说明见 [资源指南](docs/resources.md)。权重、数据和缓存不进入 Git。

GPU 环境按 [部署指南](docs/remote-validation.md) 安装匹配的 PyTorch 和固定依赖，然后分别在两个终端运行：

```bash
erag-server
# 服务启动后，在另一个终端加载模型：
erag-models --resources-root resources --llm-gpu 0 --embedding-gpu 1 --reranker-gpu 2
erag-check --env-file .env
```

本地配置模板为 `config/remote.env.example`。支持 Ark、Xinference、vLLM 和通用兼容服务；三类模型可以使用不同服务地址。更多参数见 [模型服务](docs/local-models.md)。

## 默认方案与实测结果

当前聊天默认使用**原文召回、50 个候选、重排后最多五段、历史指代消解，TCT 关闭**。原文检索不依赖摘要和标签，因此默认关闭建库时的 LLM 增强，避免为未使用的表示付费。仍可开启增强、双路摘要/标签检索、BM25/RRF 融合与 TCT 做实验。

2026-11-26 在四卡 A800 上使用真实模型完成增强建库及扩大消融，主要保留集结果如下：

| T2：160 查询、1,282 文档 | nDCG@5 | Recall@5 |
|---|---:|---:|
| 摘要/标签，15 候选 | 0.8790 | 0.8661 |
| 摘要/标签 + 重排 | 0.9151 | 0.8738 |
| 原文，50 候选，不重排 | **0.9201** | **0.8894** |
| 原文 + 重排 | 0.9161 | 0.8792 |

原文不重排相对摘要/标签不重排的 nDCG@5 提高 0.0411，配对 bootstrap 95% CI 为 [0.0112, 0.0725]；相对已有重排基线的差异只有 0.0050，区间跨零，不能宣称稳定胜出。重排对摘要/标签有收益，对原文没有稳定增益。

多轮保留集包含六个会话、36 轮，六个主方案共 216 份回答，另有 36 份关闭重排的对照。历史消解的 Recall@5 从 0.7598 到 0.7672，置信区间跨零；聚焦 TCT 为 0.7745，但调用从 79 次增加到 115 次，未证明稳定质量收益。关闭重排的多轮 Recall@5 为 0.7181，因此聊天保留重排，纯检索可关闭以减少一次服务调用。

这些是缩小语料的诊断实验，不是官方完整榜单。词重合 F1、引用编号有效性都不等于答案正确；人工历史回放也不等于自主对话。完整方法、负结果、18 份匿名审读与中文六轮回归见 [扩大评测报告](docs/expanded-evaluation.md)，汇总数据在 [reports/2026-11-26](reports/2026-11-26)。早期小样本结果保留于 [历史实验报告](docs/real-model-results.md)。

## 关键配置

`.env` 使用 `ERAG_` 前缀，环境变量优先。路径相对于启动目录，安装 wheel 后也不会写入包目录。

| 配置 | 默认 / 用途 |
|---|---|
| `RETRIEVAL_MODE` | `raw`；还支持 `dual`、`hybrid` |
| `PER_QUERY_TOP_K` / `FINAL_TOP_K` | `50` / `5` |
| `RERANK_ENABLED` | `true`；纯检索可设 `false` |
| `CONTEXTUAL_RANKING` | `true`，当前问题指代消解后参与重排 |
| `TCT_ENABLED` / `TCT_MODE` | `false` / `focused`，可选 `legacy` 对照 |
| `ENRICH_ENABLED` | `false`；设 `true` 才生成摘要和标签 |
| `HISTORY_TURNS` / `HISTORY_MAX_CHARS` | `5` / `12000` |
| `CONTEXT_MAX_CHARS` | `16000`，按完整知识块打包 |
| `KNOWLEDGE_BASE_ROOT` | `./data/knowledge_bases` |
| `CHUNK_MAX_CHARS` / `CHUNK_OVERLAP` | `900` / `120`，重叠不跨页或章节 |

若需要重现 ERAG 摘要/标签路线，同时设置 `ERAG_RETRIEVAL_MODE=dual`、`ERAG_ENRICH_ENABLED=true` 并重新建库。Embedding 模型、revision、指令或服务身份变化后也应重建。当前文本解析不含 OCR、复杂公式和表格结构恢复。

## 代码结构

```text
erag/
  engine.py, knowledge.py, providers.py  # 问答、索引与服务适配
  lexical.py, evaluation.py              # BM25/RRF 与检索指标
  schemas.py, settings.py, cli.py        # 数据契约、配置、CLI
  resources.py, resource-manifest.json   # 统一资源下载
  web/                                  # 清新聊天界面、SQLite 会话仓库
  services/                             # 模型启动、加载与真实健康检查
benchmarks/                             # 数据准备、真实实验、汇总与消融
config/                                 # 远程模板与实测依赖锁
examples/validation/                    # 自编中文课程回归集
reports/                                # 可提交的实验汇总和审读记录
scripts/remote_runtime.sh                # 项目内 conda 环境与缓存入口
docs/                                   # 架构、部署、使用和评测说明
tests/                                  # 不依赖 GPU/API Key 的行为测试
```

旧 `WebUI/`、`API/` 入口迁到 `erag.web`、`erag.services`；研究脚本迁到 `benchmarks/`。原命令需改成上面的入口或 `python -m <模块>`。Gradio 统一为已验证的 6.x，移除未覆盖测试的旧界面参数分支。

新索引为 v4，快照包含原文、摘要和标签向量。旧 v2/v3 索引可显式选择 `dual` 继续读取；使用原文/混合检索必须重建。失败构建不会替换现有 manifest；正在读取的请求保留旧快照。历史快照需停服备份后再离线回收。

## 开发与评测

```bash
pytest -q
ruff check .
ruff format --check .
python -m pip check
python -m benchmarks.validate_chat --env-file .env --output data/experiments/course-new
```

[评测指南](docs/evaluation.md) 与 [扩大评测复现](docs/expanded-evaluation.md) 提供完整命令。原始逐题结果和执行源码快照保存在本地/服务器 `data/experiments/`，不覆盖失败记录。公开仓库只提交汇总、代码哈希和可重新下载的数据版本。

[架构与 Git 历史审查](docs/architecture-review.md) 区分了原设计、回退和历史包袱。当前仍有短追问歧义、同名实体混淆、引用与论断不完全匹配等局限；不能仅靠增加反馈轮数解决。

MIT；模型权重、第三方数据和服务遵守各自许可证与条款。
