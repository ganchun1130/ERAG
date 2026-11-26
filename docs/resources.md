# 模型与数据资源

资源版本只有一个清单：[erag/resource-manifest.json](../erag/resource-manifest.json)。模型与 T2 数据使用 Hugging Face 官方 `snapshot_download`；MTRAG 使用固定 Git commit 的 HTTPS 文件并验证 SHA256。所有资源固定完整 commit SHA，支持已有文件复用，不执行资源仓库内的 Python 代码；不同 revision 使用独立目录。参见 [Hugging Face 下载说明](https://huggingface.co/docs/huggingface_hub/guides/download)。

## 下载方式

在项目根目录、Python 3.11+ 环境中运行：

```bash
pip install -e '.[resources]'

# 只查看清单、目录和体积，不访问网络
python -m erag.resources --dry-run

# 下载所有模型和公开数据
python -m erag.resources

# 若 LLM 使用云 API，只下载本地 embedding/reranker
python -m erag.resources --only embedding reranker

# 只下载公开检索数据与标签
python -m erag.resources --only retrieval-data retrieval-labels

# 验证网络与版本，只取说明/许可证/配置
python -m erag.resources --metadata-only
```

`--root /绝对路径/ERAG-resources` 可更改位置；`--max-workers 4` 控制下载并发。网络中断后重新运行同一命令，下载库会复用已完成内容。需要认证时使用标准 `HF_TOKEN` 环境变量；可通过 `HF_ENDPOINT` 自行配置可信镜像，不在代码中硬编码第三方站点。

## 内容、容量和用途

2026-11-26 按固定 revision 查询到的仓库文件总大小如下，单位为十进制 GB；磁盘应为缓存及索引预留额外空间。这些是下载体积，不是运行显存要求。

| 资源名 | 官方仓库 | 约大小 | 用途 |
|---|---|---|---|
| llm | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 16.397 GB | 本地规划、回答、证据审查 |
| embedding | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 1.207 GB | 文档与查询编码 |
| reranker | [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | 1.207 GB | 候选证据重排 |
| retrieval-data | [C-MTEB/T2Retrieval](https://huggingface.co/datasets/C-MTEB/T2Retrieval) | 0.158 GB | 通用中文检索语料和查询，原始 Parquet |
| mtrag | [IBM/mt-rag-benchmark](https://github.com/IBM/mt-rag-benchmark) | 0.010 GB | 人工编写、复核的多轮问答及相关段落 |
| retrieval-labels | [C-MTEB/T2Retrieval-qrels](https://huggingface.co/datasets/C-MTEB/T2Retrieval-qrels) | 0.001 GB | 查询与 passage 的相关性标注 |

全部约 **18.981 GB**；仅 embedding/reranker 约 **2.415 GB**。用户教材、原毕业项目实验数据不在 Git 中，不能从历史路径恢复其内容；应自行放到例如 `data/documents/`，再构建知识库。公开 benchmark 的接入边界见 [评测指南](evaluation.md)。

默认目录结构：

```text
resources/
  resources.lock.json
  models/
    Qwen3-8B/<commit SHA>/
    Qwen3-Embedding-0.6B/<commit SHA>/
    Qwen3-Reranker-0.6B/<commit SHA>/
  datasets/
    T2Retrieval/<commit SHA>/
    T2Retrieval-qrels/<commit SHA>/
    mt-rag-benchmark/<commit SHA>/conversations.json
```

`resources.lock.json` 在每项资源成功后原子更新，记录实际路径、版本和 `complete`。只下载元数据时标记为不完整，不能用于启动模型；失败不会将当前资源标记为完成。完整下载后再次验证元数据不会把完成状态改回不完整。该目录已被 Git 忽略。

2026-11-26 已在 GPU 主机项目的 `resources/` 完整下载全部六项资源，并成功加载三种模型。本机工作区只保留小文件验证资源；完整权重位于远程共享盘。`--dry-run` 不意味着权重已经存在。实测部署方式见 [远程验证](remote-validation.md)。

## 加载本地模型

PyTorch 根据目标机器的 CPU/CUDA 环境单独安装；本项目没有为所有硬件选择同一种 CUDA 发行版。

```bash
pip install -e '.[local]'

# 终端一：保持服务运行
python -m erag.services.start_xinference

# 终端二：先查看参数，再加载完整下载的模型
python -m erag.services.launch_local_models --resources-root resources --dry-run
python -m erag.services.launch_local_models --resources-root resources

# 云 LLM + 本地检索模型
python -m erag.services.launch_local_models --resources-root resources --skip-llm
```

也可继续使用 `--model-path`、`--embedding-path`、`--reranker-path` 传已有目录。远程 Xinference 的 `model_path` 指服务所在机器的路径；不要把当前机器的下载目录误传给另一台服务器。

注册模型名与调用 UID 是两个概念：Qwen3-8B 使用注册名 `qwen3`、大小 `8`、格式 `pytorch`，对外 UID 为 `qwen3-instruct`。默认通过 Transformers 引擎加载；可用 `--engine`、`--llm-size`、`--llm-name`、`--llm-uid` 调整。此处固定 UID 是为了匹配 `.env`，不是假定服务自动生成的 UID 等于模型名称。[Xinference 客户端契约](https://github.com/xorbitsai/inference/blob/v1.7.0/xinference/client/restful/restful_client.py)

本地 `.env` 要点：

```dotenv
ERAG_BACKEND=xinference
ERAG_LLM_BASE_URL=http://127.0.0.1:9997/v1
ERAG_LLM_API_KEY=local
ERAG_LLM_MODEL=qwen3-instruct
ERAG_EMBEDDING_BASE_URL=http://127.0.0.1:9997/v1
ERAG_EMBEDDING_MODEL=Qwen3-Embedding-0.6B
ERAG_EMBEDDING_REVISION=97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
ERAG_RERANK_BASE_URL=http://127.0.0.1:9997/v1
ERAG_RERANK_MODEL=Qwen3-Reranker-0.6B
```

若下载清单升级了 embedding revision，应同步更新 `.env` 并重建知识库。权重、推理服务和 API 协议三者都需要匹配；manifest 只能验证配置声明，无法检测远端服务偷偷替换权重。

Ark 使用 `thinking.type`，vLLM 的 Qwen3 使用 `chat_template_kwargs.enable_thinking`，Xinference 使用对应的 thinking 开关；用户在 `ERAG_LLM_EXTRA_BODY` 中填写的值优先。[Ark 接口](https://docs.volcengine.com/docs/ark/chat-api?lang=en)、[Qwen vLLM 部署说明](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)。换成其他模型时，应核对该模型是否支持对应字段。

对于普通 OpenAI-compatible 服务，可设置 `ERAG_BACKEND=openai` 并显式填写地址和模型名，此模式仅发送用户提供的 `llm_extra_body`，不会自动注入上述供应商参数。
