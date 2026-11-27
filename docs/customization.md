# 自定义 ERAG

ERAG 的默认流程是“资料 → 知识块 → 检索 → 带来源的回答”。模型部署、资料处理和知识库存储可以各自替换；普通使用不需要安装额外的 RAG 框架。

## 先选择需要替换的层

| 需求 | 入口 | 是否需要改 ERAG 源码 |
| --- | --- | --- |
| 换 vLLM、SGLang、llama.cpp、兼容 API 网关等服务 | `.env` / JSON 中的地址、模型名、密钥 | 否；服务需提供 Chat Completions / Embeddings 协议 |
| 用 Ollama 原生 API | `backend=ollama`、`embedding_protocol=ollama` | 否 |
| 让规划模型与回答模型使用不同服务 | `retrieval_base_url/model/api_key/backend` | 否 |
| 自定义请求头、模型参数、重排路径、查询前缀、提示词 | 配置字段 | 否 |
| 用自己的 OCR、解析器或分块方法 | `DocumentPipeline` 或 `IngestionPipeline` | 写独立扩展即可 |
| 已经用 LlamaIndex、Haystack 等生成知识块 | `build_chunks()`、`erag import`、JSONL 上传扩展 | 否；将自己的结果映射到 `Chunk` |
| 使用现有检索服务、向量库、GraphRAG | HTTP 检索桥或 `KnowledgeRepository` | 标准 HTTP 桥无需改代码；其他协议写适配器 |
| 对接原生非兼容模型 API / 本地 Python 模型 | `ChatBackend`、`EmbeddingBackend`、`RerankBackend` | 写独立扩展即可 |

“可替换”不等于任意 API 自动兼容。未实现的协议需要适配器；ERAG 不代替外部数据库的事务、备份、权限或容量管理。

## 1. 共用配置，避免 CLI 和网页各走一套

保留 `.env` 用法，也可以创建一个 JSON 配置文件，字段名使用小写的 `Settings` 字段名：

```json
{
  "backend": "openai",
  "llm_base_url": "https://your-provider.example/v1",
  "llm_model": "your-model-id",
  "rerank_enabled": false,
  "temperature": 0.2
}
```

密钥放在 `.env` 或进程环境变量 `ERAG_LLM_API_KEY` 中。知识库问答还需要正确的 Embedding 配置；普通聊天只需要 LLM。

```bash
erag --config my-config.json components
erag --config my-config.json build --name 我的资料 /path/to/notes.md
erag --config my-config.json ask --kb 我的资料 '这份资料讲了什么？'
erag-ui --config my-config.json
erag-check --config my-config.json
```

也可设置 `ERAG_CONFIG_FILE=my-config.json`。配置优先级为：**显式 Python/CLI 参数 → 进程环境变量 → JSON 配置 → .env → 默认值**。JSON 配置中的未知字段直接报错；字典字段整体替换，避免切换服务后混入另一家的请求头或私有参数。相对路径以启动进程的工作目录为准。

可从 [vLLM](../config/profiles/vllm.json)、[Ollama](../config/profiles/ollama.json) 配置开始修改。配置只连接服务，不负责启动模型；`erag-server` / `erag-models` 是 Xinference 的辅助命令，其他引擎按其部署方式启动。Ollama 模型名以本机 `ollama list` 为准，记录实际 digest 后再建立长期使用的索引。

`erag-check --chat-only` 仅检查结构化聊天能力；完整检查尊重重排开关与扩展的 `requires_embeddings` 声明，外部 HTTP 检索不强制安装本地 Embedding 服务。该命令检查模型协议与简单语义，不代表完整问答效果评测。

### 常用模型参数

| 字段 | 用途 |
| --- | --- |
| `llm_headers` / `retrieval_headers` / `embedding_headers` / `rerank_headers` | 各服务独立的额外请求头；值按密钥处理，不在 `components` 输出 |
| `llm_extra_body` / `embedding_extra_body` / `rerank_extra_body` | 服务特有参数；不能覆盖模型、输入等核心字段 |
| `temperature` / `structured_temperature` | 普通回答／结构化调用的温度；不支持温度的模型可填 `null` |
| `llm_token_parameter` | `max_tokens` 或 `max_completion_tokens` |
| `retrieval_base_url` / `retrieval_api_key` / `retrieval_backend` | 独立规划／指代修复服务；地址留空时沿用主服务 |
| `retrieval_extra_body` | 规划服务特有参数；独立地址默认不继承主服务参数或认证信息 |
| `rerank_path` | 相对 API 路径，默认 `/rerank` |
| `embedding_query_template` | 查询文本模板，必须恰好含一个 `{text}`；`{text}` 表示不增加前缀 |
| `embedding_document_prefix` | 仅加在文档端的前缀，例如某些模型的 `passage: ` |

原有 `backend=ark/vllm/xinference` 是对应服务的参数预设。其他兼容服务使用 `openai`，由 `llm_extra_body` 显式设置扩展参数。原生 Ollama 使用 `/api/chat` 和 `/api/embed`；默认禁止静默截断嵌入输入。

### 模型迁移与向量空间

默认保持严格校验：索引记录模型、版本、服务地址及查询预处理配置，配置改变会要求重建。已有 v2/v3/v4 索引继续按原规则读取。

如果会迁移服务地址或修改部署别名，可以在**首次建库前**指定 `embedding_space`，例如 `course-qwen3-embedding-97b0c6-v1`。它是你对相同模型权重与预处理的明确声明，并非根据维度猜测兼容性。设置后身份不再绑定地址与部署别名，但仍记录版本、实际查询模板、文档前缀和额外参数指纹。Qwen 模型使用自定义别名时，应显式设置查询模板。

不要让不同模型共用同一 space；不要手改 manifest 绕过校验。原先没有 space 的索引改用 space 时需要重建一次。自定义 Embedder 必须实现 `identity()`，返回不含密钥的稳定 JSON 对象；该组件自行负责查询与文档预处理。

## 2. 用自己的知识库构建方法

### 已有知识块：直接导入

```python
from erag import Chunk, ERAGEngine, Settings

chunks = [
    Chunk(
        id="manual-v1-section-3",
        content="这里放你自己解析和分块后的正文。",
        source="manual.pdf",
        chapter="第三章",
        page=12,
        metadata={"parser": "my-ocr-v2", "bbox": [10, 20, 300, 400]},
    )
]
with ERAGEngine(Settings()) as engine:
    engine.build_chunks(chunks, "我的资料")
```

也可以输出 JSONL，每行一个 `Chunk` 对象。`id`、`content`、`source` 必填，ID 必须唯一；其余字段可省略。所有输入先验证，再调用模型；失败构建不会替换原有本地索引。原样保留你的分块边界、ID、来源和 metadata。`enrich_enabled=false` 时不会重新生成摘要。

```bash
erag import --name 我的资料 /path/to/chunks.jsonl
erag-ui --config config/profiles/prebuilt-chunks.json
```

第二条命令为网页增加 JSONL 上传格式，仍支持原有 PDF/TXT/MD/CSV。其他模型配置继续来自 `.env`。

### 只替换解析器或分块器

把 `company_erag.py` 放在 Python 可以导入的位置，或安装为独立包：

```python
from erag import Extensions
from erag.ingestion import DocumentPipeline


def read_my_document(path):
    # 在这里调用你自己的 OCR / HTML / DOCX 解析工具。
    return path.read_text(encoding="utf-8")


def split_my_document(text, *, max_chars, overlap):
    # 返回 (content, chapter, page)；这里以空行为边界演示。
    return [(part, "", None) for part in text.split("\n\n") if part.strip()]


def create_extensions(settings):
    return Extensions(
        ingestion=DocumentPipeline(reader=read_my_document, splitter=split_my_document),
        upload_extensions=[".txt", ".md", ".html"],
    )
```

```bash
export ERAG_EXTENSION_FACTORY=company_erag:create_extensions
erag components
erag-ui
```

如果需要复杂多阶段处理，实现 `prepare(files, settings)` 并返回 `Iterable[Chunk]` 即可，无须采用上述 reader/splitter 组合。同一个工厂可组合多项扩展；参数放在 `extension_options`，密钥由扩展从服务端环境变量读取。

## 3. 使用已有外部知识库

[HTTP 桥配置](../config/profiles/external-knowledge.json) 将网页中的名称映射到外部知识库 ID。它直接查询远端，不要求复制向量或在 ERAG 中再次建库；网页会隐藏上传构建入口，资料由原服务维护。

ERAG 向配置的完整 `endpoint` 发起 POST：

```json
{"knowledge_id":"db-1","query":"用户问题","retrieval_setting":{"top_k":50,"score_threshold":0.0}}
```

服务返回按相关性评分的正文记录：

```json
{"records":[{"content":"支持回答的原文。","title":"第三章","score":0.91,"metadata":{"source":"manual.pdf","url":"https://docs.example/section-3"}}]}
```

该协议适用于自行实现的检索网关；**不是任意 Dify/LlamaIndex 服务地址都能直接接入**，先确认请求与返回结构一致。HTTP 错误、无效分数等会明确失败，不会伪造检索结果。ERAG 保留原文与 metadata，继续执行自己的多轮规划、可选重排/TCT 和引用回答。

若协议不同，或要直接连接 Qdrant/Milvus/Elasticsearch 等客户端，实现 [KnowledgeRepository / KnowledgeIndex](../erag/extensions.py)：

- Repository 提供 `open(name, providers, settings)` 和 `list(settings)`，管理共享连接；`open` 应返回轻量索引视图。
- Index 提供 `name/location/chapters`、`build/load/search_many`。自定义构建必须自行保证失败不覆盖旧版本。
- `cache_token()` 随数据版本变化；无法可靠获知版本时返回 `None`，禁用引擎缓存。
- 检索返回 `Hit`，包含正文 `Chunk`、有限分数、查询来源和检索路径。不要只返回最终答案，否则 ERAG 无法核对证据。
- 只读库返回 `Extensions(knowledge=repository, can_build=False)`。检索完全由远端完成时还可设置 `requires_embeddings=False`，避免健康检查要求本地嵌入服务。外部索引评测可运行，但若不暴露全部 chunks，报告会注明无法检查标注 ID 是否全部存在。

## 4. 非兼容模型与提示词

各模型能力可以单独实现，不必重写整个 Providers：

```python
from erag import ERAGEngine, Extensions, Settings


class MyChat:
    def chat(self, messages, *, structured=False, retrieval=False):
        # 换成原生 SDK 的调用；structured=True 时应返回 JSON 字符串。
        return my_client.generate(messages=messages, json_mode=structured)

    def close(self):
        my_client.close()


with ERAGEngine(Settings(), extensions=Extensions(chat=MyChat())) as engine:
    result = engine.ask("你好")
```

上例中的 `my_client` 由你的应用初始化。Embedding 扩展实现 `embed(texts, query=False)` 与 `identity()`；ERAG 校验条数、维度和有限值并归一化。重排扩展实现 `rerank(query, hits)`，必须返回全部候选、只改排序/分数，不得替换证据正文。组件可实现 `close()`，引擎关闭时按对象去重调用；网页并发使用的组件需支持并发请求。

提示词可通过配置的 `prompts` 字典覆盖：`plan`、`context_plan`、`review`、`focused_review`、`answer_rules`、`enrich`、`reference_repair`、`chat`。结构化输出字段和引用编号仍由 ERAG 校验。比如将 `chat` 改为自己的角色提示，把 `answer_rules` 改为自己领域的证据与写作要求。

## 维护边界

扩展 API 当前版本为 **1**。工厂必须返回 `Extensions`；版本或方法不匹配会在启动时失败，不会静默退回默认实现。扩展是服务端可信代码，由维护者安装；上传的文档和知识库 manifest 不会触发 Python 导入。

网页使用启动时的服务端配置，可加载同样的模型、入库和外部检索扩展；它目前不是一个为每位网页访客管理独立 API Key 的多租户平台。修改服务配置后重启网页。默认本地 NumPy 索引适合单机资料库，较大规模或多人写入由外部存储适配器承担。

设计参考：[Haystack 自定义组件](https://docs.haystack.deepset.ai/docs/custom-components) 的明确输入输出契约，以及 [LlamaIndex Ingestion Pipeline](https://developers.llamaindex.ai/python/framework/module_guides/loading/ingestion_pipeline/) 对入库转换和存储的分离。ERAG 采用这些边界设计，保持默认安装轻量，不强制依赖两套框架。

## 真实模型验证

2026-11-27，在 A800 主机上使用 vLLM 0.28.0 部署 Qwen3-8B、Qwen3-Embedding-0.6B 和 Qwen3-Reranker-0.6B，验证了三条路径：默认 Markdown 建库、自定义 JSONL 知识块、独立 HTTP 检索服务。每条路径包含首次提问、指代追问和话题切换，共 9 轮；全部通过预期事实、引用编号、证据来源和无服务降级检查。JSONL 的 ID 与 metadata 完整保留，相同向量空间更换服务地址后可继续读取索引；外部检索无需本地向量索引或 Embedding 服务。

复现时先启动模型服务，并将配置中的地址与模型名改为实际值，然后在仓库根目录运行：

```bash
python -m benchmarks.validate_extensions --config my-config.json --output /tmp/erag-extension-check.json
```

脚本使用临时合成资料和独立 HTTP BM25 服务，结束后清理测试知识库。这是扩展链路的集成验收，不是检索效果提升的统计证据。Ollama 原生 API、额外认证头和自定义组件的离线测试使用协议模拟；尚未逐一实测所有兼容服务和外部数据库产品。
