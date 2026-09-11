# 本地模型（Xinference）

ERAG 同时支持火山引擎 Ark 和本地 OpenAI-compatible 服务。推荐先启动 Xinference，再按需加载 Qwen3 指令模型、Qwen3 Embedding 与 Qwen3 Reranker；模型 UID 要与 `.env` 保持一致。

```bash
pip install -e '.[local]'
python API/start_xinference.py --host 127.0.0.1 --port 9997
python API/launch_local_models.py \
  --model-path /models/Qwen3-8B \
  --embedding-path /models/Qwen3-Embedding-0.6B \
  --reranker-path /models/Qwen3-Reranker-0.6B
```

`.env` 本地配置示例：

```dotenv
ERAG_LLM_BASE_URL=http://127.0.0.1:9997/v1
ERAG_LLM_API_KEY=local
ERAG_LLM_MODEL=qwen3-instruct
ERAG_EMBEDDING_BASE_URL=http://127.0.0.1:9997/v1
ERAG_EMBEDDING_API_KEY=local
ERAG_EMBEDDING_MODEL=Qwen3-Embedding-0.6B
ERAG_RERANK_BASE_URL=http://127.0.0.1:9997/v1
ERAG_RERANK_API_KEY=local
ERAG_RERANK_MODEL=Qwen3-Reranker-0.6B
ERAG_RERANK_ENABLED=true
```

显存有限时可仅启动 embedding/reranker，将 LLM 切换回 Ark。生产环境请绑定 `127.0.0.1` 或内网地址，并为 API 网关配置认证与 HTTPS。

## vLLM 作为本地 LLM 后端

如果更关注 GPU 吞吐和并发，可以用 vLLM 单独提供 LLM：

```bash
vllm serve /models/Qwen3-8B --served-model-name qwen3-instruct --host 127.0.0.1 --port 8000
```

将 `ERAG_BACKEND=vllm`、`ERAG_LLM_BASE_URL=http://127.0.0.1:8000/v1`，Embedding/Reranker 仍可使用本页 Xinference 服务。三类服务可以部署在不同机器上，只要网络可达即可。
