from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .engine import ERAGEngine
from .evaluation import evaluate_retrieval
from .settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="erag", description="ERAG 教学助手：知识库构建与 one-model RAG")
    parser.add_argument("--env-file", help="可选 .env 路径")
    parser.add_argument("--config", type=Path, help="JSON 配置文件（环境变量可覆盖）")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="从 PDF/TXT/MD/CSV 构建知识库")
    build.add_argument("--name", required=True)
    build.add_argument("files", nargs="+", type=Path)
    imported = sub.add_parser("import", help="导入自行解析分块的 Chunk JSONL，保留 ID、来源和 metadata")
    imported.add_argument("--name", required=True)
    imported.add_argument("file", type=Path)
    ask = sub.add_parser("ask", help="提问")
    ask.add_argument("--kb")
    ask.add_argument("query")
    ask.add_argument("--json", action="store_true", help="输出包含来源、规划、轨迹和警告的 JSON")
    evaluate = sub.add_parser("evaluate", help="以人工标注 JSONL 计算检索指标，不调用回答模型")
    evaluate.add_argument("--kb", required=True)
    evaluate.add_argument("--cases", required=True, type=Path)
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument("--summary-only", action="store_true")
    evaluate.add_argument("--rerank", action="store_true")
    evaluate.add_argument("--output", type=Path)
    sub.add_parser("list", help="列出知识库")
    sub.add_parser("components", help="查看已启用的扩展与上传格式，不显示密钥")
    args = parser.parse_args()
    options = {"_env_file": args.env_file} if args.env_file else {}
    if args.config:
        options["config_file"] = args.config
    settings = Settings(**options)
    with ERAGEngine(settings) as engine:
        if args.command == "build":
            print(
                json.dumps(engine.build_knowledge_base(args.files, args.name), ensure_ascii=False, indent=2)
            )
        elif args.command == "import":
            chunks = [
                json.loads(line)
                for line in args.file.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
            print(json.dumps(engine.build_chunks(chunks, args.name), ensure_ascii=False, indent=2))
        elif args.command == "components":
            extensions = engine.extensions
            print(
                json.dumps(
                    {
                        "extension_api": extensions.api_version,
                        "factory": settings.extension_factory or None,
                        "chat": type(extensions.chat).__name__
                        if extensions.chat is not None
                        else settings.backend,
                        "embedding": type(extensions.embedding).__name__
                        if extensions.embedding is not None
                        else settings.embedding_protocol,
                        "ingestion": type(engine.ingestion).__name__,
                        "knowledge": type(extensions.knowledge).__name__
                        if extensions.knowledge is not None
                        else "local-numpy",
                        "upload_extensions": engine.upload_extensions,
                        "prompt_overrides": sorted(settings.prompts),
                        "can_build": extensions.can_build,
                        "requires_embeddings": extensions.requires_embeddings,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.command == "list":
            print("\n".join(engine.list_knowledge_bases()) or "（暂无知识库）")
        elif args.command == "evaluate":
            kb = engine.open_knowledge_base(args.kb)
            result = evaluate_retrieval(
                kb,
                args.cases,
                k=args.top_k,
                use_tags=not args.summary_only,
                rerank=args.rerank,
                settings=settings,
                providers=engine.providers,
            )
            serialized = json.dumps(result, ensure_ascii=False, indent=2)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(serialized, encoding="utf-8")
            print(serialized)
        else:
            result = engine.ask(args.query, args.kb)
            if args.json:
                print(result.model_dump_json(indent=2))
                return
            print(result.answer)
            if result.sources:
                print("\n参考来源:")
                for i, hit in enumerate(result.sources, 1):
                    page = f" | 第{hit.chunk.page}页" if hit.chunk.page is not None else ""
                    print(f"[{i}] {hit.chunk.source} | {hit.chunk.chapter}{page} | score={hit.score:.3f}")
            for warning in result.warnings:
                print(f"提示: {warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
