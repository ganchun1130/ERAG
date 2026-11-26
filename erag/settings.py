from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ERAG_", env_file=".env", extra="ignore")
    # Any OpenAI-compatible endpoint works: Volcano Ark, Xinference, vLLM, etc.
    llm_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    backend: Literal["ark", "vllm", "xinference", "openai"] = "ark"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = "doubao-seed-1-6-flash-250615"
    retrieval_model: str = ""
    llm_extra_body: dict = Field(default_factory=dict)
    embedding_base_url: str = "http://127.0.0.1:9997/v1"
    embedding_api_key: SecretStr = SecretStr("local")
    embedding_model: str = "Qwen3-Embedding-0.6B"
    embedding_protocol: Literal["openai", "ark_multimodal"] = "openai"
    embedding_query_instruction: str = (
        "Given a question, retrieve relevant teaching materials that answer the question"
    )
    embedding_revision: str = "2025-06"
    embedding_batch_size: int = Field(16, ge=1, le=128)
    rerank_enabled: bool = True
    rerank_base_url: str = "http://127.0.0.1:9997/v1"
    rerank_api_key: SecretStr = SecretStr("local")
    rerank_model: str = "Qwen3-Reranker-0.6B"
    knowledge_base_root: Path = Path("data/knowledge_bases")
    retrieval_mode: Literal["dual", "raw", "hybrid"] = "raw"
    contextual_ranking: bool = True
    per_query_top_k: int = Field(50, ge=1, le=100)
    final_top_k: int = Field(5, ge=1, le=100)
    summary_min_score: float = Field(0.2, ge=-1, le=1)
    tag_min_score: float = Field(0.2, ge=-1, le=1)
    tct_enabled: bool = False
    tct_mode: Literal["legacy", "focused"] = "focused"
    max_retrieval_rounds: int = Field(3, ge=1, le=3)
    context_max_chars: int = Field(16000, ge=1000, le=200000)
    history_turns: int = Field(5, ge=0, le=30)
    history_max_chars: int = Field(12000, ge=1000, le=200000)
    chunk_max_chars: int = Field(900, ge=100, le=20000)
    chunk_overlap: int = Field(120, ge=0)
    enrich_enabled: bool = False
    kb_cache_size: int = Field(4, ge=0, le=100)
    timeout: float = Field(90, gt=0, le=600)
    max_tokens: int = Field(4096, ge=256, le=32768)
    ui_host: str = "127.0.0.1"
    ui_port: int = Field(7860, ge=1, le=65535)
    enable_thinking: bool = False

    def model_post_init(self, __context):
        """Make backend selection useful while preserving explicit URL overrides."""
        if self.chunk_overlap >= self.chunk_max_chars:
            raise ValueError("chunk_overlap 必须小于 chunk_max_chars")
        if self.backend in {"vllm", "xinference"} and "llm_base_url" not in self.model_fields_set:
            self.llm_base_url = (
                "http://127.0.0.1:8000/v1" if self.backend == "vllm" else "http://127.0.0.1:9997/v1"
            )
        if self.backend in {"vllm", "xinference"} and "llm_model" not in self.model_fields_set:
            self.llm_model = "qwen3-instruct"
        if self.backend in {"vllm", "xinference"} and not self.llm_api_key.get_secret_value():
            self.llm_api_key = SecretStr("local")

    def embedding_identity(self):
        return {
            "model": self.embedding_model,
            "base_url": self.embedding_base_url.rstrip("/"),
            "protocol": self.embedding_protocol,
            "revision": self.embedding_revision,
            "query_instruction": self.embedding_query_instruction,
        }
