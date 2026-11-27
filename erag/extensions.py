"""Public, synchronous extension contracts shared by Python, CLI and WebUI.

Factories are trusted application code installed by the server operator. Uploaded
documents and index manifests never select or execute Python modules.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .schemas import Chunk, Hit, SubQuery

if TYPE_CHECKING:
    from .providers import Providers
    from .settings import Settings

EXTENSION_API_VERSION = 1


class ChatBackend(Protocol):
    def chat(self, messages: list[dict], *, structured: bool = False, retrieval: bool = False) -> str: ...


class EmbeddingBackend(Protocol):
    def embed(self, texts: list[str], *, query: bool = False) -> np.ndarray: ...

    def identity(self) -> dict:
        """Stable, JSON-safe vector-space identity; include weights and preprocessing, never secrets."""
        ...


class RerankBackend(Protocol):
    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]: ...


class IngestionPipeline(Protocol):
    def prepare(self, files: list[Path], settings: Settings) -> Iterable[Chunk]: ...


class KnowledgeIndex(Protocol):
    name: str

    @property
    def location(self) -> str: ...

    @property
    def chapters(self) -> list[str]: ...

    def cache_token(self) -> Hashable | None:
        """Change when indexed content changes; None disables engine caching."""
        ...

    def build(self, chunks: Iterable[Chunk]) -> int:
        """Publish a complete replacement; failed builds must preserve the old version."""
        ...

    def load(self) -> KnowledgeIndex: ...

    def search_many(
        self,
        queries: list[SubQuery],
        *,
        key_query: str | None = None,
        top_k: int = 8,
        embedding_cache: dict[str, np.ndarray] | None = None,
    ) -> list[Hit]: ...


class KnowledgeRepository(Protocol):
    def open(self, name: str, providers: Providers, settings: Settings) -> KnowledgeIndex: ...

    def list(self, settings: Settings) -> list[str]: ...


@dataclass
class Extensions:
    """Override only the capabilities you need. Missing fields use ERAG defaults.

    Instances belong to one engine; optional close() methods run once per object
    on engine shutdown. Components used by WebUI must support concurrent calls.
    """

    api_version: int = EXTENSION_API_VERSION
    chat: ChatBackend | None = None
    embedding: EmbeddingBackend | None = None
    reranker: RerankBackend | None = None
    ingestion: IngestionPipeline | None = None
    knowledge: KnowledgeRepository | None = None
    upload_extensions: list[str] | None = None
    can_build: bool = True
    requires_embeddings: bool = True
    _closed: bool = field(default=False, init=False, repr=False)

    def validate(self) -> Extensions:
        if self.api_version != EXTENSION_API_VERSION:
            raise ValueError(f"不支持的扩展 API 版本: {self.api_version}")
        for component, methods in (
            (self.chat, ("chat",)),
            (self.embedding, ("embed", "identity")),
            (self.reranker, ("rerank",)),
            (self.ingestion, ("prepare",)),
            (self.knowledge, ("open", "list")),
        ):
            if component is not None:
                for method in methods:
                    if not callable(getattr(component, method, None)):
                        raise TypeError(f"扩展 {type(component).__name__} 缺少方法 {method}()")
        if self.upload_extensions is not None:
            import re

            if not self.upload_extensions or any(
                not re.fullmatch(r"\.[A-Za-z0-9]+", suffix) for suffix in self.upload_extensions
            ):
                raise ValueError("upload_extensions 必须是非空扩展名列表，例如 ['.md', '.docx']")
        if self.embedding is not None:
            validate_embedding_identity(self.embedding.identity())
        return self

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        seen, errors = set(), []
        for component in (self.chat, self.embedding, self.reranker, self.ingestion, self.knowledge):
            if component is None or id(component) in seen:
                continue
            seen.add(id(component))
            close = getattr(component, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise ExceptionGroup("扩展资源释放失败", errors)


def validate_embedding_identity(value: dict) -> dict:
    if not isinstance(value, dict) or not value:
        raise ValueError("自定义 embedding 必须提供非空向量空间身份")
    # Round-trip gives a detached JSON value, rejecting NaN and unserializable values.
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def load_extensions(settings: Settings) -> Extensions:
    if not settings.extension_factory:
        return Extensions()
    module, separator, name = settings.extension_factory.partition(":")
    if not separator or not module or not name or ":" in name:
        raise ValueError("ERAG_EXTENSION_FACTORY 格式应为已安装模块:工厂函数")
    factory: Callable = getattr(importlib.import_module(module), name)
    if not callable(factory):
        raise TypeError("扩展工厂必须可调用")
    extensions = factory(settings)
    if not isinstance(extensions, Extensions):
        raise TypeError("扩展工厂必须返回 erag.extensions.Extensions")
    try:
        return extensions.validate()
    except Exception:
        extensions.close()
        raise
