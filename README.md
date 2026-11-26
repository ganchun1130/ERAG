# ERAG · 知识小站

面向教学资料的 RAG 问答项目：上传资料、构建知识库、带来源回答，并保留多轮聊天历史。支持本地 Qwen3 和 OpenAI-compatible 模型服务。

界面采用薄荷绿与米白配色，左侧管理对话、知识库和检索设置，右侧聊天。支持新建、搜索历史、重命名、删除与撤销、Markdown 导出、复制、重新生成和停止。详见 [聊天工作台](docs/chat-workspace.md)。

## 快速开始

Python 3.11+，在仓库根目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,resources]'
cp .env.example .env
# 编辑 .env，配置 LLM、Embedding 和 Reranker 服务。
erag-ui
```

打开 <http://127.0.0.1:7860>。仓库不预置教材、知识库、对话或实验结果；通过左侧“添加知识”上传自己的资料。未选择知识库时可以普通聊天。启动界面会自动创建 `data/` 保存新会话，建库时再创建索引。

也可以使用命令行：

```bash
erag build --name 我的资料 /path/to/教材.pdf
erag list
erag ask --kb 我的资料 '这份资料的主要观点是什么？'
```

CLI 的 `ask` 是单次请求；网页负责保存持续历史，Python API 可显式传入 `history`。

## 本地模型

资源下载器默认只下载三个模型，使用 [固定版本清单](erag/resource-manifest.json)：

```bash
erag-download --dry-run
erag-download
erag-download --only embedding reranker  # 使用云端 LLM 时可只下载检索模型
```

模型默认位于 `resources/models/`。GPU 环境安装与服务启动见 [部署指南](docs/deployment.md)，资源目录和可选数据下载见 [资源指南](docs/resources.md)。

```bash
erag-server
# 服务启动后，在另一个终端加载模型：
erag-models --resources-root resources --llm-gpu 0 --embedding-gpu 1 --reranker-gpu 2
erag-check --env-file .env
```

## 核心配置

`.env` 使用 `ERAG_` 前缀，环境变量优先，路径相对于启动目录。

| 配置 | 默认 / 用途 |
|---|---|
| `RETRIEVAL_MODE` | `raw` 原文；还支持 `dual` 摘要/标签、`hybrid` 向量/BM25 |
| `PER_QUERY_TOP_K` / `FINAL_TOP_K` | `50` / `5` |
| `RERANK_ENABLED` | `true`，对候选证据重排 |
| `CONTEXTUAL_RANKING` | `true`，结合历史补全当前检索问题 |
| `TCT_ENABLED` / `TCT_MODE` | `false` / `focused`；按需开启补充检索 |
| `ENRICH_ENABLED` | `false`；开启后生成摘要与标签 |
| `HISTORY_TURNS` / `HISTORY_MAX_CHARS` | `5` / `12000` |
| `CONTEXT_MAX_CHARS` | `16000`，按完整知识块打包 |
| `KNOWLEDGE_BASE_ROOT` | `./data/knowledge_bases` |
| `CHUNK_MAX_CHARS` / `CHUNK_OVERLAP` | `900` / `120` |

使用摘要/标签路线时，设置 `ERAG_RETRIEVAL_MODE=dual`、`ERAG_ENRICH_ENABLED=true` 并重新建库。Embedding 模型、revision 或指令变化后应重建索引。当前解析不含 OCR、复杂公式和表格结构恢复；引用编号有效不代表每条论断都得到证据支持。

## 项目结构

```text
erag/                   # 问答引擎、知识库、模型适配、网页与服务入口
benchmarks/             # 数据准备、检索/多轮评测、消融与汇总代码
tests/                  # 无需模型服务的自动化测试
config/                 # 部署配置模板和依赖版本
scripts/                # 远程 conda 环境启动辅助脚本
docs/                   # 架构、部署、资源与聊天说明
.github/workflows/      # GitHub 自动检查
```

[架构说明](docs/architecture.md) 描述各模块职责。`data/`、`resources/`、缓存和安装生成物按需生成，不提交 Git。根目录的 `*.egg-info` 是安装/构建生成物，可以在安装完成后删除，下次构建可能再次生成。

## 开发与评测代码

```bash
pytest -q
ruff check .
ruff format --check .
python -m pip check
```

`benchmarks/` 保留评测能力，不附带历史结果和测试集。各入口使用 `python -m benchmarks.<模块> --help` 查看参数。课程验证必须用 `--fixture` 或 `--fixture-dir` 显式指定包含 `course.md` 与 `cases.json` 的自备数据；公共数据可按需下载：

```bash
erag-download --only retrieval-data retrieval-labels mtrag
python -m benchmarks.prepare_t2_fixture --help
python -m benchmarks.prepare_mtrag --help
python -m benchmarks.expanded_retrieval --help
python -m benchmarks.evaluate_dialogues --help
```

MIT；模型、第三方数据及服务遵守各自许可证与条款。
