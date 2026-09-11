"""Backward-compatible generation adapter; one model now owns all ERAG roles."""
from __future__ import annotations

from erag import ERAGEngine, Settings


class GenerationAgent:
    def __init__(self, model_name: str | None = None, endpoint: str | None = None,
                 system_prompt: str | None = None, retrieval_agent=None):
        overrides = {}
        if model_name: overrides["llm_model"] = model_name
        if endpoint: overrides["llm_base_url"] = endpoint[:-3] + "/v1" if endpoint.endswith("/v1") else endpoint
        self.engine = ERAGEngine(Settings(**overrides))
        self.retrieval_agent = retrieval_agent
        self.model_name = model_name or self.engine.settings.llm_model

    def set_retrieval_agent(self, retrieval_agent): self.retrieval_agent = retrieval_agent

    def generate(self, query: str, use_retrieval: bool = True, temperature: float = 0.3, max_tokens: int = 2048) -> str:
        kb = getattr(self.retrieval_agent, "kb_name", None) if self.retrieval_agent else None
        return self.engine.ask(query, kb if use_retrieval else None).answer

    def chat(self, messages, use_retrieval: bool = True, temperature: float = 0.3, max_tokens: int = 2048):
        last = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        return {"choices": [{"message": {"role": "assistant", "content": self.generate(last, use_retrieval)}}]}
