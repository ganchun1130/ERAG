"""Thin WebUI adapter; business logic lives in :mod:`erag.engine`."""
from __future__ import annotations

from html import escape
from pathlib import Path

from erag import ERAGEngine, Settings

_engine: ERAGEngine | None = None


def get_engine() -> ERAGEngine:
    global _engine
    if _engine is None:
        _engine = ERAGEngine(Settings())
    return _engine


def list_knowledge_bases() -> list[str]:
    return get_engine().list_knowledge_bases()


def build_knowledge_base(files, name: str | None = None) -> str:
    paths = [getattr(f, "name", f) for f in (files or [])]
    if not paths:
        return "请选择文件"
    result = get_engine().build_knowledge_base(paths, name or Path(paths[0]).stem)
    return f"已完成：{result['name']}，写入 {result['chunks']} 个知识块"


def process_query(query: str, kb_name: str):
    result = get_engine().ask(query, kb_name)
    sources = [{"source_title": f"{h.chunk.source} · {h.chunk.chapter or '未分类'}",
                "text": h.chunk.content, "score": h.score} for h in result.sources]
    return result.answer, sources


def format_response_with_sources(response: str, sources: list[dict]) -> str:
    if not sources:
        return response
    items = "".join(f"<li><strong>{escape(str(x['source_title']))}</strong>"
                    f"（{x.get('score', 0):.3f}）<br>{escape(str(x['text']))}</li>" for x in sources)
    return f"{response}\n\n<details><summary>参考来源（{len(sources)}）</summary><ol>{items}</ol></details>"


def create_history_item(title, date):
    return f"<div class='history-item'><strong>{escape(title)}</strong><small>{escape(date)}</small></div>"
