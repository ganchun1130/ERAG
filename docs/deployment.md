# 远程部署

在 GPU 主机的项目目录中安装和运行。SSH 跳板机、主机地址和登录信息使用自己的配置，不写入仓库。

## 环境

`scripts/remote_runtime.sh` 按 `ERAG_PYTHON_PREFIX`、已激活的 `CONDA_PREFIX`、项目内 `.conda/envs/erag` 的顺序选择环境，并将缓存集中到 `.cache/`。运行环境不随源码分发，已有环境时无需重复创建。也可使用自己管理的 Python 环境，直接运行 `erag-*` 命令。以下安装步骤用于 Xinference 部署。

```bash
mkdir -p logs .cache .conda/envs
CONDA_PKGS_DIRS="$PWD/.cache/conda" /opt/conda/miniconda3/bin/conda create \
  --prefix "$PWD/.conda/envs/erag" python=3.11 pip -y
bash scripts/remote_runtime.sh python -m pip install torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
bash scripts/remote_runtime.sh python -m pip install \
  -r config/remote-requirements.lock -e '.[all]'
```

上述锁文件面向 Linux、Python 3.11、CUDA 12.4；其他硬件应按目标环境选择 PyTorch。主依赖由 `pyproject.toml` 管理，部署约束见 `config/remote-constraints.txt`。

Xinference 1.13 使用 FastAPI 0.136.3、Transformers 4.53.3；更新前需验证模型请求。当前配置使用 `request_limits=1` 和模型级串行门控。不要设置 `max_num_seqs=1`，该版本会进入不受支持的 Qwen3 非批处理路径。

## Xinference：下载和启动

```bash
bash scripts/remote_runtime.sh erag-download
# 创建配置后填写模型服务参数：
test -e .env || cp config/remote.env.example .env

tmux new-session -d -s erag-models -c "$PWD" \
  'bash scripts/remote_runtime.sh erag-server >>logs/xinference.log 2>&1'
# 等待服务就绪后加载模型：
bash scripts/remote_runtime.sh erag-models --resources-root resources \
  --llm-gpu 0 --embedding-gpu 1 --reranker-gpu 2
bash scripts/remote_runtime.sh erag-check

tmux new-session -d -s erag-webui -c "$PWD" \
  'bash scripts/remote_runtime.sh erag-ui >>logs/webui.log 2>&1'
```

启动前通过 `tmux list-sessions` 和 `curl -f http://127.0.0.1:9997/v1/models` 检查已有服务，避免重复启动。LLM、Embedding、Reranker 的 UID 应与 `.env` 一致。下载器默认只取模型；网络需镜像时通过 `HF_ENDPOINT` 设置，具体说明见 [资源指南](resources.md)。

服务默认仅监听本机。客户端通过 SSH 隧道访问网页：

```bash
ssh -N -J my-jump -L 7860:127.0.0.1:7860 user@GPU_HOST
```

打开 <http://127.0.0.1:7860>，上传自己的资料。空目录启动不需要示例教材或评测数据。

## 使用其他部署引擎

ERAG 客户端只需 `pip install -e .`，可以与模型服务使用不同的 Python 环境。vLLM、Ollama 或云端 API 的配置见 [自定义指南](customization.md)；无需安装 Xinference。`erag-server` / `erag-models` 仅用于上述 Xinference 流程。

例如，先用 `erag-download` 下载模型，再在 vLLM 环境中分别启动以下服务。将三个变量设为实际模型目录；可通过 `erag-download --dry-run` 查看资源清单。不同终端启动，每个服务占用一张空闲 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve "$CHAT_MODEL_DIR" --host 127.0.0.1 --port 8000 \
  --served-model-name erag-chat --max-model-len 8192
CUDA_VISIBLE_DEVICES=1 vllm serve "$EMBEDDING_MODEL_DIR" --host 127.0.0.1 --port 8001 \
  --served-model-name erag-embedding --runner pooling --max-model-len 4096
CUDA_VISIBLE_DEVICES=2 vllm serve "$RERANKER_MODEL_DIR" --host 127.0.0.1 --port 8002 \
  --served-model-name erag-reranker --runner pooling --max-model-len 4096 \
  --hf-overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'
```

上面的第三条命令用于原版 Qwen3-Reranker；其他重排模型按其模型文档配置。将 `config/profiles/vllm.json` 中的 `rerank_enabled` 改为 `true` 后，运行 `erag-check --config config/profiles/vllm.json`，再用同一配置启动网页。只有聊天需求时只启动第一项并使用 `erag-check --chat-only`。

该路径已用 vLLM 0.28.0 验证；不同版本和硬件的安装及参数以 [vLLM 在线服务](https://docs.vllm.ai/en/stable/serving/online_serving/) 和 [重排模型说明](https://docs.vllm.ai/en/stable/models/pooling_models/scoring/) 为准。实际链路验收记录见 [真实模型验证](customization.md#真实模型验证)。

## 清理边界

- `.conda/envs/erag` 是可重新创建的运行环境。先停止网页和模型服务，再删除 `.conda/`；新机器按上面的命令重装即可。
- `.cache/xinference` 保存模型注册及服务状态，运行时保留；所有服务停止后可随 `.cache/` 一起删除。
- 安装完成且无安装任务时，可清理下载缓存；不要清理环境所依赖的符号链接目标。
- 停止网页后可删除 `data/`，这会清空知识库、历史会话和浏览器密钥。再次启动会生成新的空状态。
- `resources/datasets/` 只用于评测，可删除并按需重新下载。`resources/models/` 是本地模型权重。
- 根目录 `*.egg-info` 是构建生成物；环境内 `site-packages/*.dist-info` 是已安装包元数据，需要保留。

`.github/` 用于自动检查，`.env` 是本机配置，`.gitignore` 用于排除运行数据；它们都有用途。
