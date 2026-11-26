# 模型与可选数据

[资源清单](../erag/resource-manifest.json) 固定模型和可选评测数据的版本。模型与 T2 数据通过 Hugging Face `snapshot_download` 下载，MTRAG 使用固定 Git commit 的 HTTPS 文件并验证 SHA256；不会执行资源仓库内的代码。

## 下载

```bash
pip install -e '.[resources]'
erag-download --dry-run                 # 查看模型下载计划，不访问网络
erag-download                           # 默认只下载三个模型
erag-download --only embedding reranker # LLM 使用云服务时
erag-download --only retrieval-data retrieval-labels mtrag # 按需取评测数据
erag-download --all                      # 显式下载全部模型和可选数据
```

`--root /path/to/resources` 可指定位置，`--max-workers 4` 控制并发。中断后重复命令可继续下载；`--metadata-only` 只取说明、许可证和配置，不能用于加载模型。认证使用标准 `HF_TOKEN`，镜像可通过 `HF_ENDPOINT` 设置。

| 资源名 | 仓库 | 用途 |
|---|---|---|
| llm | Qwen/Qwen3-8B | 规划、回答与证据审查 |
| embedding | Qwen/Qwen3-Embedding-0.6B | 文档与查询编码 |
| reranker | Qwen/Qwen3-Reranker-0.6B | 候选证据重排 |
| retrieval-data | C-MTEB/T2Retrieval | 可选中文检索数据 |
| retrieval-labels | C-MTEB/T2Retrieval-qrels | 可选相关性标签 |
| mtrag | IBM/mt-rag-benchmark | 可选多轮问答数据 |

三个模型约 18.8 GB，仅 Embedding/Reranker 约 2.4 GB，实际体积以 `--dry-run` 的固定清单为准。下载目录按 revision 隔离：

```text
resources/
  resources.lock.json   # 本机实际下载状态
  models/<模型>/<commit>/
  datasets/<数据集>/<commit>/  # 显式下载评测数据后才出现
```

`resources.lock.json` 在每项下载成功后原子更新。删除数据目录后，应移除对应的本机下载状态；清单 `erag/resource-manifest.json` 保留，使评测代码能够按需重新获取数据。`downloaded_path` 会校验状态与实际目录，不将已删除目录当成可用资源。

## 模型服务

先按 [部署指南](deployment.md) 安装适配硬件的 PyTorch 与服务依赖，再启动 Xinference：

```bash
erag-server
erag-models --resources-root resources --dry-run
erag-models --resources-root resources
erag-check --env-file .env
```

也可使用 `--model-path`、`--embedding-path`、`--reranker-path` 指定已有权重。路径指模型服务所在主机的文件系统。Qwen3-8B 的注册名为 `qwen3`，对外 UID 默认 `qwen3-instruct`；使用云端 LLM 时加 `--skip-llm`。

若使用 vLLM 提供 LLM：

```bash
vllm serve /models/Qwen3-8B --served-model-name qwen3-instruct --host 127.0.0.1 --port 8000
```

对应设置 `ERAG_BACKEND=vllm`、`ERAG_LLM_BASE_URL=http://127.0.0.1:8000/v1`。Embedding/Reranker 可以继续由 Xinference 提供。Ark、vLLM、Xinference 分别映射 thinking 参数；`ERAG_LLM_EXTRA_BODY` 显式设置优先。普通兼容服务使用 `ERAG_BACKEND=openai`。

Embedding 的权重 revision、查询指令或服务身份变化后应重建知识库。不要把另一台机器上的路径当成本机已下载资源。
