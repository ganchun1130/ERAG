"""Backward-compatible adapter around the new :class:`ERAGEngine` retriever."""
from __future__ import annotations

from pathlib import Path
from erag import ERAGEngine, Settings


class RetrievalAgent:
    def __init__(self, embedding_model: str | None = None, top_k: int = 4, engine: ERAGEngine | None = None):
        self.engine = engine or ERAGEngine(Settings(**({"embedding_model": embedding_model} if embedding_model else {})))
        self.top_k = top_k
        self.kb_name: str | None = None

    def load_documents(self, documents, metadatas=None):
        # Legacy callers passed strings; write no temporary files and build through a
        # small in-memory adapter is intentionally not supported by the persistent API.
        raise NotImplementedError("请改用 engine.build_knowledge_base([文件路径], name)")

    def load_from_files(self, file_paths):
        self.kb_name = Path(file_paths[0]).stem
        return self.engine.build_knowledge_base(file_paths, self.kb_name)

    def retrieve(self, query):
        if not self.kb_name: raise ValueError("请先加载知识库")
        return [h.chunk for h in self.engine.ask(query, self.kb_name).sources[:self.top_k]]

    def get_context(self, query):
        return "\n\n".join(x.content for x in self.retrieve(query))
