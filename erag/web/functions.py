"""Thin UI adapter; keep display markup out of model conversation history."""

from __future__ import annotations

import atexit
from html import escape
from pathlib import Path
from threading import Lock

from erag import ERAGEngine, Settings
from erag.schemas import Hit

_engine: ERAGEngine | None = None
_engine_lock = Lock()


def get_engine(settings: Settings | None = None) -> ERAGEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = ERAGEngine(settings or Settings())
            atexit.register(_engine.close)
    return _engine


def list_knowledge_bases() -> list[str]:
    return get_engine().list_knowledge_bases()


def build_knowledge_base(files, name: str | None = None) -> dict:
    paths = [f if isinstance(f, (str, Path)) else f.name for f in (files or [])]
    if not paths:
        raise ValueError("请选择文件")
    return get_engine().build_knowledge_base(paths, name.strip() if name and name.strip() else None)


def format_response_with_sources(response: str, sources: list[Hit]) -> str:
    if not sources:
        return response
    items = []
    for hit in sources:
        chunk = hit.chunk
        title = f"{Path(chunk.source).name} · {chunk.chapter or '未分类'}"
        if chunk.page is not None:
            title += f" · 第{chunk.page}页"
        items.append(f"<li><strong>{escape(title)}</strong><br>{escape(chunk.content)}</li>")
    return f"{response}\n\n<details><summary>参考来源（{len(sources)}）</summary><ol>{''.join(items)}</ol></details>"
