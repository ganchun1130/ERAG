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
    def __init__(self, settings: Settings):
        self.s = settings
        self._chat: OpenAI | None = None
        self._embedding: OpenAI | None = None
        self._http: httpx.Client | None = None
        self._lock = RLock()

    def _client(self, *, embedding: bool = False) -> OpenAI:
        with self._lock:
            attribute = "_embedding" if embedding else "_chat"
            client = getattr(self, attribute)
            if client is None:
                client = OpenAI(
                    base_url=self.s.embedding_base_url if embedding else self.s.llm_base_url,
                    api_key=(
                        self.s.embedding_api_key if embedding else self.s.llm_api_key
                    ).get_secret_value(),
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

    def _extra_body(self) -> dict:
        body = copy.deepcopy(self.s.llm_extra_body)
        # Explicit provider-specific overrides always win.
        if self.s.backend == "ark":
            body.setdefault("thinking", {"type": "enabled" if self.s.enable_thinking else "disabled"})
        elif self.s.backend == "vllm":
            body.setdefault("chat_template_kwargs", {}).setdefault("enable_thinking", self.s.enable_thinking)
        elif self.s.backend == "xinference":
            body.setdefault("enable_thinking", self.s.enable_thinking)
        return body

    def chat(self, messages: list[dict], *, structured: bool = False, retrieval: bool = False) -> str:
        if not self.s.llm_model or not self.s.llm_api_key.get_secret_value():
            raise ValueError("请在 .env 配置 ERAG_LLM_MODEL 和 ERAG_LLM_API_KEY（本地可填 local）")
        model = (self.s.retrieval_model or self.s.llm_model) if retrieval else self.s.llm_model
        # Xinference 1.13's Qwen3 Transformers scheduler corrupts overlapping decodes.
        # A cross-process, per-model gate also coordinates CLI, builds and WebUI.
        # Separate replicas remain concurrent; hosted/vLLM backends are unaffected.
        gate = nullcontext()
        if self.s.backend == "xinference":
            key = hashlib.sha256(f"{self.s.llm_base_url.rstrip('/')}|{model}".encode()).hexdigest()
            directory = Path(tempfile.gettempdir()) / "erag-model-locks"
            directory.mkdir(mode=0o700, exist_ok=True)
            gate = FileLock(directory / f"{key}.lock", timeout=self.s.timeout).acquire(poll_interval=0.01)
        with gate:
            result = self._client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=0 if structured else 0.3,
                max_tokens=self.s.max_tokens,
                extra_body=self._extra_body(),
            )
        if not result.choices:
            raise ValueError("模型未返回候选回答")
        if result.choices[0].finish_reason == "length":
            raise ValueError("模型输出被截断，请提高 ERAG_MAX_TOKENS 或关闭模型思考模式")
        content = result.choices[0].message.content
        if not content or not content.strip():
            raise ValueError("模型返回空回答，请检查模型的思考模式和输出预算")
        return content

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
        if query and self.s.embedding_query_instruction and "qwen3" in self.s.embedding_model.lower():
            texts = [f"Instruct: {self.s.embedding_query_instruction}\nQuery: {t}" for t in texts]
        vectors = []
        for start in range(0, len(texts), self.s.embedding_batch_size):
            batch = texts[start : start + self.s.embedding_batch_size]
            if self.s.embedding_protocol == "ark_multimodal":
                # An array of multimodal inputs describes ONE item, not a document batch.
                for text in batch:
                    response = self._http_client().post(
                        self.s.embedding_base_url.rstrip("/") + "/embeddings/multimodal",
                        headers={"Authorization": "Bearer " + self.s.embedding_api_key.get_secret_value()},
                        json={"model": self.s.embedding_model, "input": [{"type": "text", "text": text}]},
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
                    model=self.s.embedding_model, input=batch
                )
                rows = sorted(response.data, key=lambda row: row.index)
                if [row.index for row in rows] != list(range(len(batch))):
                    raise ValueError("嵌入服务返回条数或索引错误")
                vectors.extend(row.embedding for row in rows)
        array = np.asarray(vectors, dtype=np.float64)
        if (
            array.ndim != 2
            or array.shape[0] != len(texts)
            or array.shape[1] == 0
            or not np.isfinite(array).all()
        ):
            raise ValueError("嵌入服务返回无效向量")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms < 1e-12):
            raise ValueError("嵌入服务返回零向量或数值溢出，已停止操作")
        return np.ascontiguousarray(array / norms, dtype=np.float32)

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return []
        response = self._http_client().post(
            self.s.rerank_base_url.rstrip("/") + "/rerank",
            headers={"Authorization": "Bearer " + self.s.rerank_api_key.get_secret_value()},
            json={
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
            for attribute in ("_chat", "_embedding", "_http"):
                client = getattr(self, attribute)
                if client is not None:
                    client.close()
                    setattr(self, attribute, None)
