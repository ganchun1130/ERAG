"""Provider adapters with pooled connections and validated model responses."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import tempfile
from contextlib import nullcontext
from pathlib import Path
from threading import RLock

import httpx
import numpy as np
from filelock import FileLock
from openai import OpenAI

from .extensions import Extensions, validate_embedding_identity
from .schemas import Hit
from .settings import Settings


def parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("模型必须返回 JSON 对象")
    return value


class Providers:
    def __init__(self, settings: Settings, *, extensions: Extensions | None = None):
        self.s = settings
        self.extensions = extensions or Extensions()
        self._chat: OpenAI | None = None
        self._retrieval: OpenAI | None = None
        self._embedding: OpenAI | None = None
        self._http: httpx.Client | None = None
        self._lock = RLock()

    @staticmethod
    def _headers(values) -> dict[str, str]:
        return {key: value.get_secret_value() for key, value in values.items()}

    def _chat_endpoint(self, retrieval: bool = False):
        independent = retrieval and bool(self.s.retrieval_base_url)
        return (
            self.s.retrieval_base_url if independent else self.s.llm_base_url,
            (self.s.retrieval_api_key if independent else self.s.llm_api_key).get_secret_value(),
            (self.s.retrieval_backend or "openai") if independent else self.s.backend,
        )

    def _client(self, *, embedding: bool = False, retrieval: bool = False) -> OpenAI:
        with self._lock:
            attribute = (
                "_embedding"
                if embedding
                else "_retrieval"
                if retrieval and self.s.retrieval_base_url
                else "_chat"
            )
            client = getattr(self, attribute)
            if client is None:
                url, key, _ = self._chat_endpoint(retrieval)
                client = OpenAI(
                    base_url=self.s.embedding_base_url if embedding else url,
                    api_key=(self.s.embedding_api_key.get_secret_value() if embedding else key) or "local",
                    # Do not forward the main service's custom auth headers to a separate endpoint.
                    default_headers=self._headers(
                        self.s.embedding_headers
                        if embedding
                        else self.s.retrieval_headers
                        if retrieval and self.s.retrieval_base_url
                        else self.s.llm_headers
                    ),
                    timeout=self.s.timeout,
                    max_retries=2,
                )
                setattr(self, attribute, client)
            return client

    def _http_client(self) -> httpx.Client:
        with self._lock:
            if self._http is None:
                self._http = httpx.Client(timeout=self.s.timeout)
            return self._http

    def _extra_body(self, retrieval: bool = False) -> dict:
        independent = retrieval and bool(self.s.retrieval_base_url)
        override = self.s.retrieval_extra_body if retrieval else None
        body = copy.deepcopy(
            override if override is not None else {} if independent else self.s.llm_extra_body
        )
        _, _, backend = self._chat_endpoint(retrieval)
        # Explicit provider-specific overrides always win.
        if backend == "ark":
            body.setdefault("thinking", {"type": "enabled" if self.s.enable_thinking else "disabled"})
        elif backend == "vllm":
            body.setdefault("chat_template_kwargs", {}).setdefault("enable_thinking", self.s.enable_thinking)
        elif backend == "xinference":
            body.setdefault("enable_thinking", self.s.enable_thinking)
        return body

    def chat(self, messages: list[dict], *, structured: bool = False, retrieval: bool = False) -> str:
        if self.extensions.chat is not None:
            content = self.extensions.chat.chat(
                copy.deepcopy(messages), structured=structured, retrieval=retrieval
            )
            return self._chat_content(content)
        model = (self.s.retrieval_model or self.s.llm_model) if retrieval else self.s.llm_model
        if not model:
            raise ValueError("请配置 ERAG_LLM_MODEL（独立检索模型可配置 ERAG_RETRIEVAL_MODEL）")
        url, key, backend = self._chat_endpoint(retrieval)
        temperature = self.s.structured_temperature if structured else self.s.temperature
        if backend == "ollama":
            body = self._extra_body(retrieval)
            options = body.setdefault("options", {})
            options.setdefault("num_predict", self.s.max_tokens)
            if temperature is not None:
                options.setdefault("temperature", temperature)
            body.update(model=model, messages=messages, stream=False)
            if structured:
                body.setdefault("format", "json")
            response = self._http_client().post(
                url.rstrip("/") + "/api/chat",
                json=body,
                headers=({"Authorization": "Bearer " + key} if key else {})
                | self._headers(
                    self.s.retrieval_headers
                    if retrieval and self.s.retrieval_base_url
                    else self.s.llm_headers
                ),
            )
            response.raise_for_status()
            result = response.json()
            if result.get("done_reason") == "length":
                raise ValueError("模型输出被截断，请提高 ERAG_MAX_TOKENS")
            return self._chat_content(result["message"]["content"])
        # Xinference 1.13's Qwen3 Transformers scheduler corrupts overlapping decodes.
        # A cross-process, per-model gate also coordinates CLI, builds and WebUI.
        # Separate replicas remain concurrent; hosted/vLLM backends are unaffected.
        gate = nullcontext()
        if backend == "xinference":
            key = hashlib.sha256(f"{url.rstrip('/')}|{model}".encode()).hexdigest()
            directory = Path(tempfile.gettempdir()) / "erag-model-locks"
            directory.mkdir(mode=0o700, exist_ok=True)
            gate = FileLock(directory / f"{key}.lock", timeout=self.s.timeout).acquire(poll_interval=0.01)
        with gate:
            options = {self.s.llm_token_parameter: self.s.max_tokens}
            if temperature is not None:
                options["temperature"] = temperature
            result = self._client(retrieval=retrieval).chat.completions.create(
                model=model,
                messages=messages,
                extra_body=self._extra_body(retrieval),
                **options,
            )
        if not result.choices:
            raise ValueError("模型未返回候选回答")
        if result.choices[0].finish_reason == "length":
            raise ValueError("模型输出被截断，请提高 ERAG_MAX_TOKENS 或关闭模型思考模式")
        return self._chat_content(result.choices[0].message.content)

    @staticmethod
    def _chat_content(content) -> str:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("模型返回空回答，请检查模型的思考模式和输出预算")
        return content

    def embedding_identity(self) -> dict:
        return (
            validate_embedding_identity(self.extensions.embedding.identity())
            if self.extensions.embedding is not None
            else self.s.embedding_identity()
        )

    def json(self, system: str, payload: dict, *, retrieval: bool = False) -> dict:
        return parse_json(
            self.chat(
                [
                    {"role": "system", "content": system + "\n只输出合法 JSON 对象，不输出解释。"},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                structured=True,
                retrieval=retrieval,
            )
        )

    def embed(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        if not texts or any(not t.strip() for t in texts):
            raise ValueError("嵌入输入不能为空")
        if self.extensions.embedding is not None:
            return self._normalize(self.extensions.embedding.embed(list(texts), query=query), len(texts))
        texts = [
            self.s.query_template().replace("{text}", t) if query else self.s.embedding_document_prefix + t
            for t in texts
        ]
        vectors = []
        for start in range(0, len(texts), self.s.embedding_batch_size):
            batch = texts[start : start + self.s.embedding_batch_size]
            if self.s.embedding_protocol == "ollama":
                response = self._http_client().post(
                    self.s.embedding_base_url.rstrip("/") + "/api/embed",
                    headers=(
                        {"Authorization": "Bearer " + self.s.embedding_api_key.get_secret_value()}
                        if self.s.embedding_api_key.get_secret_value()
                        else {}
                    )
                    | self._headers(self.s.embedding_headers),
                    json={
                        "truncate": False,
                        **self.s.embedding_extra_body,
                        "model": self.s.embedding_model,
                        "input": batch,
                    },
                )
                response.raise_for_status()
                rows = response.json()["embeddings"]
                if len(rows) != len(batch):
                    raise ValueError("嵌入服务返回条数错误")
                vectors.extend(rows)
            elif self.s.embedding_protocol == "ark_multimodal":
                # An array of multimodal inputs describes ONE item, not a document batch.
                for text in batch:
                    response = self._http_client().post(
                        self.s.embedding_base_url.rstrip("/") + "/embeddings/multimodal",
                        headers=(
                            {"Authorization": "Bearer " + self.s.embedding_api_key.get_secret_value()}
                            if self.s.embedding_api_key.get_secret_value()
                            else {}
                        )
                        | self._headers(self.s.embedding_headers),
                        json={
                            **self.s.embedding_extra_body,
                            "model": self.s.embedding_model,
                            "input": [{"type": "text", "text": text}],
                        },
                    )
                    response.raise_for_status()
                    data = response.json()["data"]
                    if isinstance(data, list):
                        if len(data) != 1:
                            raise ValueError("多模态嵌入服务返回条数错误")
                        data = data[0]
                    vectors.append(data["embedding"])
            else:
                response = self._client(embedding=True).embeddings.create(
                    model=self.s.embedding_model,
                    input=batch,
                    extra_body=copy.deepcopy(self.s.embedding_extra_body),
                )
                rows = sorted(response.data, key=lambda row: row.index)
                if [row.index for row in rows] != list(range(len(batch))):
                    raise ValueError("嵌入服务返回条数或索引错误")
                vectors.extend(row.embedding for row in rows)
        return self._normalize(vectors, len(texts))

    @staticmethod
    def _normalize(vectors, count: int) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] != count or array.shape[1] == 0 or not np.isfinite(array).all():
            raise ValueError("嵌入服务返回无效向量")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms < 1e-12):
            raise ValueError("嵌入服务返回零向量或数值溢出，已停止操作")
        return np.ascontiguousarray(array / norms, dtype=np.float32)

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return []
        if self.extensions.reranker is not None:
            ranked = [
                Hit.model_validate(h.model_dump())
                for h in self.extensions.reranker.rerank(query, [h.model_copy(deep=True) for h in hits])
            ]
            originals = {h.chunk.id: h.chunk for h in hits}
            if sorted(h.chunk.id for h in ranked) != sorted(originals) or any(
                h.chunk != originals[h.chunk.id] for h in ranked
            ):
                raise ValueError("自定义重排必须返回全部候选且不能修改来源内容")
            return sorted(ranked, key=lambda h: h.score, reverse=True)
        response = self._http_client().post(
            self.s.rerank_base_url.rstrip("/") + self.s.rerank_path,
            headers=(
                {"Authorization": "Bearer " + self.s.rerank_api_key.get_secret_value()}
                if self.s.rerank_api_key.get_secret_value()
                else {}
            )
            | self._headers(self.s.rerank_headers),
            json={
                **self.s.rerank_extra_body,
                "model": self.s.rerank_model,
                "query": query,
                "documents": [h.chunk.content for h in hits],
                "top_n": len(hits),
            },
        )
        response.raise_for_status()
        rows = response.json()["results"]
        indices = [r["index"] for r in rows]
        if any(type(i) is not int for i in indices) or sorted(indices) != list(range(len(hits))):
            raise ValueError("重排服务返回索引不完整")
        ranked = []
        for row in rows:
            score = float(row.get("relevance_score", row.get("score", float("nan"))))
            if not math.isfinite(score):
                raise ValueError("重排服务返回无效分数")
            ranked.append(hits[row["index"]].model_copy(update={"score": score}))
        return sorted(ranked, key=lambda hit: hit.score, reverse=True)

    def close(self) -> None:
        with self._lock:
            for attribute in ("_chat", "_retrieval", "_embedding", "_http"):
                client = getattr(self, attribute)
                if client is not None:
                    client.close()
                    setattr(self, attribute, None)
