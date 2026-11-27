"""Browser-workspace settings, isolated model clients and bounded engine lifetimes."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Literal
from urllib.parse import urlsplit

import httpx
from openai import APIStatusError
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from erag import ERAGEngine, Settings
from erag.extensions import load_extensions
from erag.ingestion import DocumentPipeline
from erag.integrations.chunks import JsonlPipeline
from erag.integrations.external_knowledge import ExternalConfig, ExternalRepository
from erag.knowledge import validate_name
from erag.schemas import Chunk, Hit, SubQuery

from .history import identifier


class ConnectionCheckError(ValueError):
    """A credential-free explanation produced by our semantic checks."""


def api_url(value: str) -> str:
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("API 地址须以 http:// 或 https:// 开头，并包含主机名")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("API 地址中不要包含密钥、查询参数或片段，请在密钥栏填写认证信息")
    return value


class Connection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol: str = "inherit"
    url: str = ""
    model: str = ""
    key: SecretStr = SecretStr("")
    revision: str = ""
    query_template: str = ""


class WebProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chat: Connection = Field(default_factory=Connection)
    embedding: Connection = Field(default_factory=Connection)
    rerank: Connection = Field(default_factory=Connection)
    method: Literal["inherit", "documents", "jsonl", "external"] = "inherit"
    chunk_size: int = Field(900, ge=100, le=20000)
    chunk_overlap: int = Field(120, ge=0)
    enrich: bool = False
    external_url: str = ""
    external_name: str = "我的外部知识库"
    external_id: str = ""
    external_key: SecretStr = SecretStr("")

    @model_validator(mode="after")
    def valid(self):
        roles = {
            "chat": {"inherit", "openai", "vllm", "xinference", "ollama", "ark"},
            "embedding": {"inherit", "openai", "ollama", "ark_multimodal"},
            "rerank": {"inherit", "openai", "disabled"},
        }
        for role, allowed in roles.items():
            connection = getattr(self, role)
            if connection.protocol not in allowed:
                raise ValueError("请选择有效的模型服务类型")
            if connection.protocol not in {"inherit", "disabled"}:
                connection.url = api_url(connection.url)
                connection.model = connection.model.strip()
                if not connection.model:
                    raise ValueError("自定义模型服务需要填写模型名称")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("重叠长度必须小于分块长度")
        if self.embedding.query_template and self.embedding.query_template.count("{text}") != 1:
            raise ValueError("查询模板须包含且只包含一个 {text}")
        if self.method == "external":
            self.external_url = api_url(self.external_url)
            self.external_name = validate_name(self.external_name.strip())
            self.external_id = self.external_id.strip()
            if not self.external_id:
                raise ValueError("请填写远程知识库 ID")
        return self


def private_json(profile: WebProfile) -> str:
    def encode(value):
        if isinstance(value, SecretStr):
            return value.get_secret_value()
        raise TypeError(type(value).__name__)

    return json.dumps(profile.model_dump(), default=encode, ensure_ascii=False)


class ProfileStore:
    """Private server-side storage. Never send stored credentials back to the browser."""

    def __init__(self, path: Path, defaults: WebProfile):
        self.path, self.defaults = Path(path), defaults
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.touch(mode=0o600, exist_ok=True)
        self.path.chmod(0o600)
        with self.connection() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS profiles "
                "(owner TEXT PRIMARY KEY, revision INTEGER NOT NULL, value TEXT NOT NULL)"
            )

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def load(self, owner: str) -> tuple[int, WebProfile]:
        with self.connection() as db:
            row = db.execute(
                "SELECT revision,value FROM profiles WHERE owner=?", (identifier(owner),)
            ).fetchone()
        return (
            (row[0], WebProfile.model_validate_json(row[1]))
            if row
            else (0, self.defaults.model_copy(deep=True))
        )

    def save(self, owner: str, profile: WebProfile):
        with self.connection() as db:
            db.execute(
                "INSERT INTO profiles VALUES (?,1,?) ON CONFLICT(owner) DO UPDATE SET "
                "revision=revision+1,value=excluded.value",
                (identifier(owner), private_json(profile)),
            )

    def prepare(self, owner, values, clear=()) -> WebProfile:
        _, previous = self.load(owner)
        draft = WebProfile.model_validate(values)
        for role in ("chat", "embedding", "rerank"):
            old, new = getattr(previous, role), getattr(draft, role)
            if (
                role not in clear
                and not new.key.get_secret_value()
                and (new.protocol, new.url) == (old.protocol, old.url)
            ):
                new.key = old.key
        if (
            "external" not in clear
            and not draft.external_key.get_secret_value()
            and draft.external_url == previous.external_url
        ):
            draft.external_key = previous.external_key
        # Explicit clear wins over a filled password field, including password-manager autofill.
        for role in clear:
            if role == "external":
                draft.external_key = SecretStr("")
            elif role in {"chat", "embedding", "rerank"}:
                getattr(draft, role).key = SecretStr("")
        return draft


@dataclass
class Workspace:
    engine: ERAGEngine
    scope: str
    users: int = 0


class WorkspaceRuntime:
    def __init__(self, base: Settings, directory: Path, *, cache_size=8):
        self.base, self.directory = base, Path(directory)
        defaults = WebProfile(
            chunk_size=base.chunk_max_chars, chunk_overlap=base.chunk_overlap, enrich=base.enrich_enabled
        )
        self.profiles = ProfileStore(self.directory / "web_settings" / "profiles.sqlite3", defaults)
        self._lock, self._engines, self.cache_size = RLock(), OrderedDict(), cache_size
        self._operations = {}

    def create(self, owner, profile, *, rerank=None, tct=None) -> Workspace:
        owner = identifier(owner)
        values = self.base.model_dump()
        values["config_file"] = None
        for role, prefix in (("chat", "llm"), ("embedding", "embedding"), ("rerank", "rerank")):
            connection = getattr(profile, role)
            if connection.protocol in {"inherit", "disabled"}:
                continue
            values.update(
                {
                    f"{prefix}_base_url": connection.url,
                    f"{prefix}_model": connection.model,
                    f"{prefix}_api_key": connection.key,
                    f"{prefix}_headers": {},
                    f"{prefix}_extra_body": {},
                }
            )
            if role == "chat":
                values.update(
                    backend=connection.protocol,
                    retrieval_model="",
                    retrieval_base_url="",
                    retrieval_api_key=SecretStr(""),
                    retrieval_headers={},
                    retrieval_backend=None,
                    retrieval_extra_body=None,
                )
            elif role == "embedding":
                values.update(
                    embedding_protocol=connection.protocol,
                    embedding_space="",
                    embedding_revision=connection.revision or "web-custom-v1",
                    embedding_query_template=connection.query_template or None,
                    embedding_document_prefix="",
                )
            else:
                values.update(rerank_enabled=True, rerank_path="/rerank")
        if rerank is not None:
            values["rerank_enabled"] = bool(rerank)
        if profile.rerank.protocol == "disabled":
            values["rerank_enabled"] = False
        if tct is not None:
            values["tct_enabled"] = bool(tct)
        if profile.method in {"documents", "jsonl"}:
            values.update(
                chunk_max_chars=profile.chunk_size,
                chunk_overlap=profile.chunk_overlap,
                enrich_enabled=profile.enrich,
            )
        settings = Settings(_env_file=None, **values)
        extensions = load_extensions(settings)
        original = [
            extensions.chat,
            extensions.embedding,
            extensions.reranker,
            extensions.ingestion,
            extensions.knowledge,
        ]
        try:
            for role, attribute in (("chat", "chat"), ("embedding", "embedding"), ("rerank", "reranker")):
                if getattr(profile, role).protocol != "inherit":
                    setattr(extensions, attribute, None)
            scope = ""
            if profile.method in {"documents", "jsonl"}:
                extensions.knowledge = None
                extensions.ingestion = JsonlPipeline() if profile.method == "jsonl" else DocumentPipeline()
                extensions.upload_extensions = [".jsonl"] if profile.method == "jsonl" else None
                extensions.can_build = extensions.requires_embeddings = True
            elif profile.method == "external":
                config = ExternalConfig(
                    endpoint=profile.external_url,
                    knowledge_bases={profile.external_name: profile.external_id},
                )
                extensions.knowledge = ExternalRepository(
                    config, settings.timeout, api_key=profile.external_key.get_secret_value()
                )
                extensions.can_build = extensions.requires_embeddings = False
                scope = (
                    "external:"
                    + hashlib.sha256(f"{profile.external_url}|{profile.external_id}".encode()).hexdigest()
                )
            if extensions.knowledge is None and (
                profile.method != "inherit" or profile.embedding.protocol != "inherit"
            ):
                identity = (
                    extensions.embedding.identity() if extensions.embedding else settings.embedding_identity()
                )
                space = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
                settings.knowledge_base_root = (
                    self.directory / "workspaces" / owner / "knowledge_bases" / space
                )
                scope = "local:" + space
            remaining = [
                extensions.chat,
                extensions.embedding,
                extensions.reranker,
                extensions.ingestion,
                extensions.knowledge,
            ]
            discarded = {
                id(obj): obj
                for obj in original
                if obj is not None and not any(obj is other for other in remaining)
            }
            for obj in discarded.values():
                if callable(getattr(obj, "close", None)):
                    obj.close()
            return Workspace(ERAGEngine(settings, extensions=extensions), scope)
        except Exception:
            extensions.close()
            raise

    @contextmanager
    def use(self, owner, *, rerank=None, tct=None, operation=False):
        owner = identifier(owner)
        with self._lock:
            # Read the revision and acquire its lease atomically with configuration publication.
            revision, profile = self.profiles.load(owner)
            key = (owner, revision, rerank, tct)
            if key not in self._engines:
                self._engines[key] = self.create(owner, profile, rerank=rerank, tct=tct)
            workspace = self._engines[key]
            workspace.users += 1
            if operation:
                self._operations[owner] = self._operations.get(owner, 0) + 1
            self._engines.move_to_end(key)
        try:
            yield workspace
        finally:
            with self._lock:
                workspace.users -= 1
                if operation:
                    self._operations[owner] -= 1
                    if not self._operations[owner]:
                        del self._operations[owner]
                current_revision, _ = self.profiles.load(owner)
                for old_key, old in list(self._engines.items()):
                    stale = old_key[0] == owner and old_key[1] != current_revision
                    if not old.users and (stale or len(self._engines) > self.cache_size):
                        del self._engines[old_key]
                        old.engine.close()

    def apply(self, owner, profile):
        # Fail before publishing invalid factory/configuration; do not require network availability to save.
        owner = identifier(owner)
        with self._lock:
            if self._operations.get(owner, 0):
                raise ValueError("当前工作空间仍在处理请求，请完成后再保存设置")
            workspace = self.create(owner, profile)
            workspace.engine.close()
            self.profiles.save(owner, profile)

    def test(self, owner, profile, role):
        workspace = self.create(owner, profile)
        with workspace.engine as engine:
            if role == "chat":
                if engine.providers.json("计算加法，输出 sum 字段。", {"a": 17, "b": 25}).get("sum") != 42:
                    raise ConnectionCheckError("聊天服务未通过结构化输出检查，请检查模型能力")
                return "聊天连接正常，结构化输出检查通过。"
            if role == "knowledge":
                if not engine.extensions.requires_embeddings:
                    names = engine.list_knowledge_bases()
                    if not names:
                        raise ConnectionCheckError("未配置外部知识库")
                    engine.open_knowledge_base(names[0]).search_many([SubQuery(query="连接测试")], top_k=1)
                    return "外部知识库连接正常。"
                vectors = engine.providers.embed(["知识库连接测试"])
                return f"嵌入连接正常，向量维度为 {vectors.shape[1]}。"
            if role != "rerank" or not engine.settings.rerank_enabled:
                raise ConnectionCheckError("请先启用并配置重排服务")
            hits = [
                Hit(
                    chunk=Chunk(id=str(i), content=text, source="connection-test"),
                    score=0,
                    sub_query="中国的首都",
                    route="check",
                )
                for i, text in enumerate(["北京是中国的首都。", "鲸鱼生活在海洋中。"])
            ]
            if engine.providers.rerank("中国的首都是哪里？", hits)[0].chunk.id != "0":
                raise ConnectionCheckError("重排服务返回了结果，但语义检查未通过，请核对模型模板")
            return "重排连接正常，语义检查通过。"

    def close(self):
        with self._lock:
            for workspace in self._engines.values():
                workspace.engine.close()
            self._engines.clear()


def service_error(exc) -> str:
    """Do not store or echo provider error bodies that may contain credentials."""
    if isinstance(exc, ConnectionCheckError):
        return str(exc)
    if isinstance(exc, (APIStatusError, httpx.HTTPStatusError)):
        code = exc.status_code if isinstance(exc, APIStatusError) else exc.response.status_code
        if code in {401, 403}:
            return "服务认证失败，请检查 API 密钥和访问权限。"
        if code == 404:
            return "未找到服务或模型，请核对 API 地址和模型名称。"
        return f"服务返回 HTTP {code}，请检查服务状态后重试。"
    return "操作未完成，请检查模型连接、资料格式及知识库配置后重试。"
