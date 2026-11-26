"""Thin UI adapter; keep display markup out of model conversation history."""

from __future__ import annotations

import atexit
from html import escape
from pathlib import Path
from threading import Lock

from erag import ERAGEngine, Settings
from erag.schemas import Answer, Hit

_engine: ERAGEngine | None = None
_engine_lock = Lock()
_configured_engines: dict[tuple[bool, bool], ERAGEngine] = {}


def get_engine() -> ERAGEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = ERAGEngine(Settings())
            atexit.register(_engine.close)
    return _engine


def list_knowledge_bases() -> list[str]:
    return get_engine().list_knowledge_bases()


def build_knowledge_base(files, name: str | None = None) -> dict:
    paths = [f if isinstance(f, (str, Path)) else f.name for f in (files or [])]
    if not paths:
        raise ValueError("请选择文件")
    return get_engine().build_knowledge_base(paths, name.strip() if name and name.strip() else None)


def process_query(
    query: str,
    kb_name: str | None,
    history: list[dict] | None = None,
    *,
    rerank_enabled: bool | None = None,
    tct_enabled: bool | None = None,
) -> Answer:
    engine = get_engine()
    if rerank_enabled is not None or tct_enabled is not None:
        key = (
            engine.settings.rerank_enabled if rerank_enabled is None else rerank_enabled,
            engine.settings.tct_enabled if tct_enabled is None else tct_enabled,
        )
        with _engine_lock:
            if key not in _configured_engines:
                configured = engine.settings.model_copy(
                    update={"rerank_enabled": key[0], "tct_enabled": key[1]}
                )
                _configured_engines[key] = ERAGEngine(configured)
                atexit.register(_configured_engines[key].close)
            engine = _configured_engines[key]
    return engine.ask(query, kb_name, history=history)


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
