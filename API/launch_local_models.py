"""Launch Qwen3 LLM, embedding and reranker in a running Xinference server."""
from __future__ import annotations
import argparse
from xinference.client import RESTfulClient


def launch(client, *, name, path, model_type=None, engine="transformers"):
    kwargs = {"model_name": name, "model_path": path, "model_engine": engine}
    if model_type:
        kwargs["model_type"] = model_type
    uid = client.launch_model(**kwargs)
    print(f"{model_type or 'LLM'}: {uid}")
    return uid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default="http://127.0.0.1:9997")
    p.add_argument("--model-path", help="Qwen3-Instruct 本地路径")
    p.add_argument("--embedding-path", help="Qwen3-Embedding-0.6B 本地路径")
    p.add_argument("--reranker-path", help="Qwen3-Reranker-0.6B 本地路径")
    p.add_argument("--skip-llm", action="store_true")
    args = p.parse_args()
    client = RESTfulClient(args.endpoint)
    if not args.skip_llm:
        if not args.model_path:
            p.error("未指定 --model-path（或使用 --skip-llm）")
        launch(client, name="qwen3-instruct", path=args.model_path)
    if args.embedding_path:
        launch(client, name="Qwen3-Embedding-0.6B", path=args.embedding_path, model_type="embedding")
    if args.reranker_path:
        launch(client, name="Qwen3-Reranker-0.6B", path=args.reranker_path, model_type="rerank")


if __name__ == "__main__":
    main()
