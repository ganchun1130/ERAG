from collections import deque

import numpy as np
import pytest

from erag.schemas import Chunk
from erag.settings import Settings


class FakeProviders:
    """Deterministic, offline doubles; no API keys or local model server required."""

    def __init__(self):
        self.vectors = {}
        self.embed_calls = []
        self.json_calls = []
        self.chat_calls = []
        self.responses = deque()
        self.closed = False

    def embed(self, texts, *, query=False):
        self.embed_calls.append((list(texts), query))
        vectors = np.asarray([self.vectors.get(text, [1, 0, 0]) for text in texts], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def json(self, system, payload, **kwargs):
        self.json_calls.append((system, payload))
        if self.responses:
            response = self.responses.popleft()
            if isinstance(response, Exception):
                raise response
            return response
        if "schema" in payload:
            return {"key_query": payload["query"], "sub_queries": [{"query": payload["query"]}]}
        if "sub_queries" in payload:
            return {"complete": True}
        return {"answer": "资料支持此结论。[1]"}

    def chat(self, messages, **kwargs):
        self.chat_calls.append(messages)
        return "这是回答。"

    def rerank(self, query, hits):
        return hits

    def close(self):
        self.closed = True


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        knowledge_base_root=tmp_path / "kbs",
        backend="xinference",
        retrieval_mode="dual",
        rerank_enabled=False,
        enrich_enabled=False,
        tct_enabled=False,
    )


@pytest.fixture
def providers():
    return FakeProviders()


@pytest.fixture
def chunks():
    return [
        Chunk(
            id="a",
            content="形成性评价用于反馈。",
            summary="summary-a",
            tags=["tag-a"],
            source="教材.md",
            chapter="第一章",
            page=1,
        ),
        Chunk(
            id="b",
            content="总结性评价用于总结。",
            summary="summary-b",
            tags=["tag-b"],
            source="教材.md",
            chapter="第二章",
            page=2,
        ),
    ]
