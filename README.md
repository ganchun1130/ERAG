# ERAG 教学助手（Enhanced RAG）

ERAG 是一个面向中文教学问答的 Enhanced RAG 教学助手，支持 PDF/TXT/Markdown/CSV 知识库、复杂多轮问题、引用来源和本地部署。

## 这次升级解决了什么

- **one-model Agent**：Router、检索、回答、反思仍保留为可观测的逻辑角色，但默认共用一个 OpenAI-compatible 模型；通过 `ERAG_RETRIEVAL_MODEL` 可选用更快/更便宜的规划模型。
- **复杂查询处理**：语义完整分块；块摘要 + 关键问题标签的双向意图映射；情境保留与分治查询；章节路由；摘要召回 + 标签意图补充；可选重排序；TCT 证据审查与反馈检索（最多 3 轮）。
- **模型可替换**：默认示例使用火山引擎 Ark；同一套代码可切换到 Xinference、vLLM 或其他 OpenAI-compatible 网关。Embedding 默认 `Qwen3-Embedding-0.6B`，Reranker 默认 `Qwen3-Reranker-0.6B`。
- **可复现索引**：知识库保存 `chunks.json`、向量文件和 `manifest.json`。manifest 固化 embedding 模型、协议、revision 和指令；配置变化时拒绝静默复用旧向量。
- **Web UI 重写**：Gradio 5/6 兼容界面支持知识库选择、上传构建、进度结果、清空对话和参考来源。

## 选择模型后端

模型后端完全由使用者选择，不需要修改业务代码。设置 `ERAG_BACKEND=ark`、`vllm` 或 `xinference`，再按需覆盖各服务地址：

| 后端 | 适合场景 | 关键配置 |
|---|---|---|
| 火山引擎 Ark API | 快速试用、无需本地 GPU、希望降低运维成本 | `ERAG_LLM_BASE_URL`、`ERAG_LLM_API_KEY`、`ERAG_LLM_MODEL` |
| 本地 vLLM | Linux/GPU 服务器、高并发、需要 OpenAI-compatible 服务 | 将 `ERAG_LLM_BASE_URL` 指向 vLLM `/v1` |
| 本地 Xinference | 桌面环境、多模型统一管理、同时部署 Embedding/Reranker | 运行 `API/launch_local_models.py` |

Embedding 和 Reranker 也可以独立选择本地服务或兼容 API；因此可以组合成“Ark LLM + 本地 Embedding/Reranker”，不必三者全部部署在同一处。

## 快速开始（Ark API）

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
# 编辑 .env：填写 ERAG_LLM_API_KEY、ERAG_LLM_MODEL，以及 Embedding 服务地址
python -m erag.cli build --name 计算教育学 /path/to/教材.pdf
python -m WebUI.ui
```

命令行提问：

```bash
python -m erag.cli list
python -m erag.cli ask --kb 计算教育学 "比较形成性评价与总结性评价的适用场景"
```

## 本地 Xinference（LLM + Embedding + Reranker）

```bash
pip install -e '.[local]'
python API/start_xinference.py
python API/launch_local_models.py \
  --model-path /models/Qwen3-8B \
  --embedding-path /models/Qwen3-Embedding-0.6B \
  --reranker-path /models/Qwen3-Reranker-0.6B
```

然后将 `.env` 中的 `ERAG_LLM_BASE_URL`、`ERAG_EMBEDDING_BASE_URL`、`ERAG_RERANK_BASE_URL` 都设为 `http://127.0.0.1:9997/v1`，模型名分别设为 `qwen3-instruct`、`Qwen3-Embedding-0.6B`、`Qwen3-Reranker-0.6B`。显存不足时可以只本地部署 embedding/reranker，LLM 继续使用 Ark。更多参数见 [API/README.md](API/README.md)。

此时在 `.env` 设置 `ERAG_BACKEND=xinference`；如果显式填写了服务地址，则显式地址优先。

## 本地 vLLM

vLLM 负责提供 OpenAI-compatible LLM 接口，Embedding/Reranker 可以继续使用 Xinference 或其他兼容服务：

```bash
pip install vllm
vllm serve /models/Qwen3-8B --served-model-name qwen3-instruct --host 127.0.0.1 --port 8000
```

`.env`：

```dotenv
ERAG_LLM_BASE_URL=http://127.0.0.1:8000/v1
ERAG_LLM_API_KEY=local
ERAG_LLM_MODEL=qwen3-instruct
ERAG_BACKEND=vllm
```

## 配置要点

所有配置使用 `ERAG_` 前缀，可放在 `.env` 或环境变量中：

| 配置 | 作用 | 默认 |
|---|---|---|
| `ERAG_BACKEND` | 选择 Ark、vLLM 或 Xinference | `ark` |
| `ERAG_LLM_BASE_URL` / `ERAG_LLM_MODEL` | 生成、规划和反思模型 | Ark / `doubao-seed-1-6-flash-250615` |
| `ERAG_RETRIEVAL_MODEL` | 可选的检索规划模型 | 空（与 LLM 相同） |
| `ERAG_EMBEDDING_MODEL` | 向量模型 | `Qwen3-Embedding-0.6B` |
| `ERAG_RERANK_ENABLED` / `ERAG_RERANK_MODEL` | Cross-encoder 重排 | true / `Qwen3-Reranker-0.6B` |
| `ERAG_KNOWLEDGE_BASE_ROOT` | 索引目录 | `./data/knowledge_bases` |
| `ERAG_MAX_RETRIEVAL_ROUNDS` | TCT 最大检索轮数 | 3 |
| `ERAG_ENABLE_THINKING` | 是否开启服务商 thinking | false（结构化 JSON 更稳定） |

Embedding 模型或 instruction 变更后必须重新执行 `build`。不要把 API Key 提交到 Git；`.env` 已被忽略。

## 项目结构

```text
erag/
  engine.py       # one-model Router + contextual rewrite + TCT
  knowledge.py    # PDF/TXT 解析、语义分块、双向意图映射、向量索引
  providers.py    # OpenAI-compatible LLM/Embedding/Rerank 适配器
  schemas.py      # QueryPlan、Chunk、Hit、Answer 等契约
  settings.py     # .env/Pydantic 配置
  cli.py          # build/list/ask
WebUI/            # Gradio 5/6 界面
API/              # Xinference 启动与模型加载脚本
tests/            # 核心行为测试
```

## 从旧版本迁移

项目主流程统一调用 `ERAGEngine`，不再依赖旧版脚本、硬编码路径或固定模型 UID：

```python
from erag import ERAGEngine, Settings

engine = ERAGEngine(Settings())
engine.build_knowledge_base(["教材.pdf"], "计算教育学")
result = engine.ask("请比较两种教学评价方法", "计算教育学", history=[])
print(result.answer, result.sources, result.trace)
```

## 测试与扩展

运行核心测试：

```bash
pytest
```

未来可在 `KnowledgeBase` 增加网页/多模态解析，在 `ERAGEngine._answer` 增加引用校验和流式输出。

## 许可证

MIT。模型权重与第三方 API 请遵守各自许可证和服务条款。
