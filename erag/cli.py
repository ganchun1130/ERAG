from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import ERAGEngine
from .settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="erag", description="ERAG 教学助手：知识库构建与 one-model RAG")
    parser.add_argument("--env-file", help="可选 .env 路径")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="从 PDF/TXT/MD/CSV 构建知识库")
    build.add_argument("--name", required=True)
    build.add_argument("files", nargs="+", type=Path)
    ask = sub.add_parser("ask", help="提问")
    ask.add_argument("--kb")
    ask.add_argument("query")
    sub.add_parser("list", help="列出知识库")
    args = parser.parse_args()
    settings = Settings(_env_file=args.env_file) if args.env_file else Settings()
    engine = ERAGEngine(settings)
    if args.command == "build":
        print(json.dumps(engine.build_knowledge_base(args.files, args.name), ensure_ascii=False, indent=2))
    elif args.command == "list":
        print("\n".join(engine.list_knowledge_bases()) or "（暂无知识库）")
    else:
        result = engine.ask(args.query, args.kb)
        print(result.answer)
        if result.sources:
            print("\n参考来源:")
            for i, hit in enumerate(result.sources, 1):
                print(f"[{i}] {hit.chunk.source} | {hit.chunk.chapter} | score={hit.score:.3f}")


if __name__ == "__main__":
    main()
