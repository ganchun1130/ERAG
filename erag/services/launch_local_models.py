"""Launch pinned Qwen3 models with stable UIDs in an existing Xinference server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from erag.resources import downloaded_path


def launch_spec(
    *,
    name: str,
    uid: str,
    path: str,
    model_type: str = "LLM",
    engine: str = "Transformers",
    size: str = "8",
    gpu: int | None = None,
) -> dict:
    spec = {"model_name": name, "model_uid": uid, "model_path": path, "model_type": model_type}
    if model_type == "LLM":
        spec.update(
            model_engine=engine,
            model_size_in_billions=size,
            model_format="pytorch",
            quantization="none",
            # This backend's Qwen3 batch decoder corrupts concurrent requests. Providers
            # queue calls per model; reject uncoordinated external concurrent requests.
            **({"request_limits": 1} if engine.lower() == "transformers" else {}),
        )
    if gpu is not None:
        if gpu < 0:
            raise ValueError("GPU 编号不能为负数")
        spec.update(n_gpu=1, gpu_idx=[gpu])
    return spec


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:9997")
    parser.add_argument("--resources-root", type=Path, help="下载脚本的资源目录，自动读取完整下载记录")
    parser.add_argument("--model-path", help="Qwen3 本地模型目录")
    parser.add_argument("--embedding-path", help="Qwen3-Embedding-0.6B 本地目录")
    parser.add_argument("--reranker-path", help="Qwen3-Reranker-0.6B 本地目录")
    parser.add_argument("--llm-name", default="qwen3", help="Xinference 注册模型名，与服务 UID 区分")
    parser.add_argument("--llm-uid", default="qwen3-instruct")
    parser.add_argument("--llm-size", default="8")
    parser.add_argument("--engine", default="Transformers")
    parser.add_argument("--llm-gpu", type=int, help="LLM 使用的单张 GPU 编号；不指定时由服务决定")
    parser.add_argument("--embedding-gpu", type=int, help="Embedding 使用的单张 GPU 编号")
    parser.add_argument("--reranker-gpu", type=int, help="Reranker 使用的单张 GPU 编号")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只输出启动参数，不连接服务")
    args = parser.parse_args(argv)
    if any(gpu is not None and gpu < 0 for gpu in [args.llm_gpu, args.embedding_gpu, args.reranker_gpu]):
        parser.error("GPU 编号不能为负数")
    try:
        if args.resources_root:
            if not args.skip_llm and not args.model_path:
                args.model_path = downloaded_path(args.resources_root, "llm")
            args.embedding_path = args.embedding_path or downloaded_path(args.resources_root, "embedding")
            args.reranker_path = args.reranker_path or downloaded_path(args.resources_root, "reranker")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    if not args.skip_llm and not args.model_path:
        parser.error("未指定 --model-path 或 --resources-root（也可使用 --skip-llm）")
    specs = []
    if not args.skip_llm:
        specs.append(
            launch_spec(
                name=args.llm_name,
                uid=args.llm_uid,
                path=args.model_path,
                engine=args.engine,
                size=args.llm_size,
                gpu=args.llm_gpu,
            )
        )
    for path, name, model_type, gpu in [
        (args.embedding_path, "Qwen3-Embedding-0.6B", "embedding", args.embedding_gpu),
        (args.reranker_path, "Qwen3-Reranker-0.6B", "rerank", args.reranker_gpu),
    ]:
        if path:
            specs.append(launch_spec(name=name, uid=name, path=path, model_type=model_type, gpu=gpu))
    if not specs:
        parser.error("没有指定要启动的模型")
    if args.dry_run:
        print(json.dumps(specs, ensure_ascii=False, indent=2))
        return

    from xinference.client import RESTfulClient

    client = RESTfulClient(args.endpoint)
    for spec in specs:
        uid = client.launch_model(**spec)
        print(f"{spec['model_type']}: {uid}")


if __name__ == "__main__":
    main()
