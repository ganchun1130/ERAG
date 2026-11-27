"""Read-only HTTP retrieval bridge; the remote service owns ingestion/storage.

POST endpoint: {knowledge_id, query, retrieval_setting: {top_k, score_threshold}}
Response: {records: [{content, title, score, metadata: {...}}]}.
"""

from __future__ import annotations

import hashlib
import json
import os

import httpx
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, JsonValue

from erag.extensions import Extensions
from erag.knowledge import validate_name
from erag.schemas import Chunk, Hit


class ExternalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    endpoint: HttpUrl
    knowledge_bases: dict[str, str] = Field(min_length=1)
    api_key_env: str = ""
    score_threshold: float = Field(0, ge=0, le=1)


class Record(BaseModel):
    content: str = Field(min_length=1)
    title: str = ""
    score: float = Field(allow_inf_nan=False)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ExternalRepository:
    def __init__(self, config: ExternalConfig, timeout: float, *, api_key: str | None = None):
        for name, identifier in config.knowledge_bases.items():
            validate_name(name)
            if not identifier.strip():
                raise ValueError("外部知识库 ID 不能为空")
        key = (
            api_key
            if api_key is not None
            else os.environ.get(config.api_key_env, "")
            if config.api_key_env
            else ""
        )
        if api_key is None and config.api_key_env and not key:
            raise ValueError(f"请设置外部知识库密钥环境变量 {config.api_key_env}")
        self.config = config
        self.client = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {key}"} if key else {})

    def list(self, settings):
        return sorted(self.config.knowledge_bases)

    def open(self, name, providers, settings):
        if name not in self.config.knowledge_bases:
            raise FileNotFoundError(f"外部知识库未配置: {name}")
        return ExternalIndex(name, self)

    def close(self):
        self.client.close()


class ExternalIndex:
    def __init__(self, name, repository):
        self.name, self.repository = name, repository
        self.identifier = repository.config.knowledge_bases[name]

    @property
    def location(self):
        return f"external://{self.identifier}"

    @property
    def chapters(self):
        return []

    def cache_token(self):
        return None  # Fetch current remote content on every request.

    def load(self):
        return self

    def build(self, chunks):
        raise ValueError("请在外部知识库服务中构建或更新资料")

    def search_many(self, queries, *, key_query=None, top_k=8, embedding_cache=None):
        if top_k < 1:
            raise ValueError("top_k 必须大于 0")
        config, hits = self.repository.config, []
        for sub in queries:
            response = self.repository.client.post(
                str(config.endpoint),
                json={
                    "knowledge_id": self.identifier,
                    "query": sub.query,
                    "retrieval_setting": {"top_k": top_k, "score_threshold": config.score_threshold},
                },
            )
            response.raise_for_status()
            rows = response.json()["records"]
            if not isinstance(rows, list):
                raise ValueError("外部知识库 records 必须是列表")
            records = [Record.model_validate(row) for row in rows]
            for record in sorted(records, key=lambda r: r.score, reverse=True)[:top_k]:
                if record.score < config.score_threshold:
                    continue
                source = (
                    record.metadata.get("source")
                    or record.metadata.get("url")
                    or record.title
                    or self.location
                )
                if not isinstance(source, str):
                    raise ValueError("外部知识库 source/url 必须是字符串")
                identity = json.dumps([self.identifier, source, record.content], ensure_ascii=False)
                chunk = Chunk(
                    id=hashlib.sha256(identity.encode()).hexdigest(),
                    content=record.content,
                    summary=record.content,
                    source=source,
                    chapter=record.title,
                    metadata=record.metadata,
                )
                hits.append(Hit(chunk=chunk, score=record.score, sub_query=sub.query, route="external"))
        return hits


def create_extensions(settings):
    config = ExternalConfig.model_validate(settings.extension_options)
    return Extensions(
        knowledge=ExternalRepository(config, settings.timeout), can_build=False, requires_embeddings=False
    )
