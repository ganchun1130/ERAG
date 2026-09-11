"""Explicit provider adapters; inference failures never become fake embeddings."""
import json
import re
try:  # Optional at import time so ingestion/tests can run without API extras.
    import httpx
except ImportError:  # pragma: no cover
    httpx = None
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None
from .settings import Settings


def parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("模型必须返回 JSON 对象")
    return value


class Providers:
    def __init__(self, settings: Settings):
        self.s = settings
        self._chat = None
        self._embedding = None

    def chat(self, messages, *, structured=False, retrieval=False):
        if OpenAI is None:
            raise RuntimeError("缺少 openai 依赖，请运行 pip install -e '.[all]'")
        if not self.s.llm_model or not self.s.llm_api_key.get_secret_value():
            raise ValueError("请在 .env 配置 ERAG_LLM_MODEL 和 ERAG_LLM_API_KEY（本地可填 local）")
        if self._chat is None:
            self._chat = OpenAI(base_url=self.s.llm_base_url,
                                api_key=self.s.llm_api_key.get_secret_value(),
                                timeout=self.s.timeout, max_retries=2)
        result = self._chat.chat.completions.create(
            model=(self.s.retrieval_model or self.s.llm_model) if retrieval else self.s.llm_model,
            messages=messages, temperature=0 if structured else 0.3,
            max_tokens=self.s.max_tokens,
            extra_body={**self.s.llm_extra_body, "enable_thinking": self.s.enable_thinking} or None)
        if result.choices[0].finish_reason == "length":
            raise ValueError("模型输出被截断，请提高 ERAG_MAX_TOKENS 或关闭模型思考模式")
        content = result.choices[0].message.content
        if not content or not content.strip():
            raise ValueError("模型返回空回答，请检查模型的思考模式和输出预算")
        return content

    def json(self, system, payload, *, retrieval=False):
        text = self.chat([{"role": "system", "content": system + "\n只输出合法 JSON 对象，不输出解释。"},
                          {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                         structured=True, retrieval=retrieval)
        return parse_json(text)

    def embed(self, texts: list[str], *, query=False):
        if OpenAI is None or np is None:
            raise RuntimeError("缺少 openai/numpy 依赖，请运行 pip install -e '.[all]'")
        if not texts or any(not t.strip() for t in texts):
            raise ValueError("嵌入输入不能为空")
        if query and self.s.embedding_query_instruction and "qwen3" in self.s.embedding_model.lower():
            texts = [f"Instruct: {self.s.embedding_query_instruction}\nQuery: {t}" for t in texts]
        if self._embedding is None:
            self._embedding = OpenAI(base_url=self.s.embedding_base_url,
                                     api_key=self.s.embedding_api_key.get_secret_value(),
                                     timeout=self.s.timeout, max_retries=2)
        vectors = []
        for start in range(0, len(texts), self.s.embedding_batch_size):
            batch = texts[start:start + self.s.embedding_batch_size]
            if self.s.embedding_protocol == "ark_multimodal":
                # A multimodal input array describes ONE item, not a batch of documents.
                if httpx is None:
                    raise RuntimeError("ark_multimodal 协议需要 httpx")
                for text in batch:
                    response = httpx.post(self.s.embedding_base_url.rstrip("/") + "/embeddings/multimodal",
                        headers={"Authorization": "Bearer " + self.s.embedding_api_key.get_secret_value()},
                        json={"model": self.s.embedding_model, "input": [{"type": "text", "text": text}]}, timeout=self.s.timeout)
                    response.raise_for_status()
                    payload = response.json(); data = payload["data"]
                    row = data[0] if isinstance(data, list) else data
                    vectors.append(row["embedding"])
            else:
                response = self._embedding.embeddings.create(model=self.s.embedding_model, input=batch)
                rows = sorted(response.data, key=lambda row: row.index)
                if [row.index for row in rows] != list(range(len(batch))):
                    raise ValueError("嵌入服务返回条数或索引错误")
                vectors.extend(row.embedding for row in rows)
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != len(texts) or not np.isfinite(array).all():
            raise ValueError("嵌入服务返回无效向量")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if np.any(norms < 1e-12):
            raise ValueError("嵌入服务返回零向量，已停止操作")
        return np.ascontiguousarray(array / norms)

    def rerank(self, query, hits):
        if httpx is None:
            raise RuntimeError("缺少 httpx 依赖，请运行 pip install -e '.[all]'")
        with httpx.Client(timeout=self.s.timeout) as client:
            response = client.post(self.s.rerank_base_url.rstrip("/") + "/rerank",
                headers={"Authorization": "Bearer " + self.s.rerank_api_key.get_secret_value()},
                json={"model": self.s.rerank_model, "query": query,
                      "documents": [h.chunk.content for h in hits], "top_n": len(hits)})
            response.raise_for_status()
        rows = response.json()["results"]
        if sorted(r["index"] for r in rows) != list(range(len(hits))):
            raise ValueError("重排服务返回索引不完整")
        def score(row): return float(row.get("relevance_score", row.get("score", 0.0)))
        return [hits[r["index"]].model_copy(update={"score": score(r)})
                for r in sorted(rows, key=score, reverse=True)]

    def close(self):
        for client in (self._chat, self._embedding):
            if client:
                client.close()
