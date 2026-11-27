import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ERAG_", env_file=".env", extra="ignore")
    config_file: Path | None = None
    extension_factory: str = ""
    extension_options: dict = Field(default_factory=dict)
    prompts: dict[str, str] = Field(default_factory=dict)
    # Any OpenAI-compatible endpoint works: Volcano Ark, Xinference, vLLM, etc.
    llm_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    backend: Literal["ark", "vllm", "xinference", "openai", "ollama"] = "ark"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = "doubao-seed-1-6-flash-250615"
    retrieval_model: str = ""
    retrieval_base_url: str = ""
    retrieval_api_key: SecretStr = SecretStr("")
    retrieval_headers: dict[str, SecretStr] = Field(default_factory=dict)
    retrieval_backend: Literal["ark", "vllm", "xinference", "openai", "ollama"] | None = None
    retrieval_extra_body: dict | None = None
    llm_extra_body: dict = Field(default_factory=dict)
    llm_headers: dict[str, SecretStr] = Field(default_factory=dict)
    temperature: float | None = Field(0.3, ge=0, le=2)
    structured_temperature: float | None = Field(0, ge=0, le=2)
    llm_token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    embedding_base_url: str = "http://127.0.0.1:9997/v1"
    embedding_api_key: SecretStr = SecretStr("local")
    embedding_model: str = "Qwen3-Embedding-0.6B"
    embedding_protocol: Literal["openai", "ark_multimodal", "ollama"] = "openai"
    embedding_headers: dict[str, SecretStr] = Field(default_factory=dict)
    embedding_extra_body: dict = Field(default_factory=dict)
    embedding_space: str = ""
    embedding_query_template: str | None = None
    embedding_document_prefix: str = ""
    embedding_query_instruction: str = (
        "Given a question, retrieve relevant teaching materials that answer the question"
    )
    embedding_revision: str = "2025-06"
    embedding_batch_size: int = Field(16, ge=1, le=128)
    rerank_enabled: bool = True
    rerank_base_url: str = "http://127.0.0.1:9997/v1"
    rerank_api_key: SecretStr = SecretStr("local")
    rerank_model: str = "Qwen3-Reranker-0.6B"
    rerank_headers: dict[str, SecretStr] = Field(default_factory=dict)
    rerank_path: str = "/rerank"
    rerank_extra_body: dict = Field(default_factory=dict)
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

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        def profile():
            selected = next(
                (
                    source().get("config_file")
                    for source in (init_settings, env_settings, dotenv_settings)
                    if source().get("config_file")
                ),
                None,
            )
            if not selected:
                return {}
            data = json.loads(Path(selected).read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict):
                raise ValueError("配置文件必须是 JSON 对象")
            unknown = set(data) - cls.model_fields.keys()
            if unknown or "config_file" in data:
                raise ValueError(
                    f"配置文件含未知或递归字段: {sorted(unknown | ({'config_file'} & data.keys()))}"
                )
            return data

        def merged():
            # A profile must override a copied example .env. Replace complete maps
            # rather than deep-merging stale vendor headers/body into another API.
            values = {}
            for source in (file_secret_settings, dotenv_settings, profile, env_settings, init_settings):
                values.update(source())
            return values

        # Explicit Python/CLI options > shell > JSON profile > dotenv > defaults.
        return (merged,)

    def model_post_init(self, __context):
        """Make backend selection useful while preserving explicit URL overrides."""
        if self.chunk_overlap >= self.chunk_max_chars:
            raise ValueError("chunk_overlap 必须小于 chunk_max_chars")
        allowed_prompts = {
            "plan",
            "context_plan",
            "review",
            "focused_review",
            "answer_rules",
            "enrich",
            "reference_repair",
            "chat",
        }
        if set(self.prompts) - allowed_prompts or any(not text.strip() for text in self.prompts.values()):
            raise ValueError(f"prompts 只支持非空提示词: {sorted(allowed_prompts)}")
        if self.embedding_query_template is not None and self.embedding_query_template.count("{text}") != 1:
            raise ValueError("embedding_query_template 必须恰好包含一个 {text}")
        if not self.rerank_path.startswith("/") or self.rerank_path.startswith("//"):
            raise ValueError("rerank_path 必须是以 / 开头的 API 路径")
        for label, body, reserved in (
            ("llm_extra_body", self.llm_extra_body, {"model", "messages", "stream"}),
            ("retrieval_extra_body", self.retrieval_extra_body or {}, {"model", "messages", "stream"}),
            ("embedding_extra_body", self.embedding_extra_body, {"model", "input"}),
            ("rerank_extra_body", self.rerank_extra_body, {"model", "query", "documents", "top_n"}),
        ):
            if reserved & body.keys():
                raise ValueError(f"{label} 不可覆盖请求核心字段: {sorted(reserved & body.keys())}")
        if self.backend in {"vllm", "xinference"} and "llm_base_url" not in self.model_fields_set:
            self.llm_base_url = (
                "http://127.0.0.1:8000/v1" if self.backend == "vllm" else "http://127.0.0.1:9997/v1"
            )
        if self.backend in {"vllm", "xinference"} and "llm_model" not in self.model_fields_set:
            self.llm_model = "qwen3-instruct"
        if self.backend in {"vllm", "xinference"} and not self.llm_api_key.get_secret_value():
            self.llm_api_key = SecretStr("local")
        if self.backend in {"openai", "ollama"}:
            if "llm_base_url" not in self.model_fields_set:
                self.llm_base_url = (
                    "https://api.openai.com/v1" if self.backend == "openai" else "http://127.0.0.1:11434"
                )
            if "llm_model" not in self.model_fields_set:
                self.llm_model = ""  # Explicit model IDs avoid a vendor-specific silent fallback.

    def query_template(self) -> str:
        if self.embedding_query_template is not None:
            return self.embedding_query_template
        if self.embedding_query_instruction and "qwen3" in self.embedding_model.lower():
            return f"Instruct: {self.embedding_query_instruction}\nQuery: {{text}}"
        return "{text}"

    def embedding_identity(self):
        identity = {
            "model": self.embedding_model,
            "base_url": self.embedding_base_url.rstrip("/"),
            "protocol": self.embedding_protocol,
            "revision": self.embedding_revision,
            "query_instruction": self.embedding_query_instruction,
        }
        if self.embedding_space:
            identity = {
                "space": self.embedding_space,
                "revision": self.embedding_revision,
                "query_template": self.query_template(),
            }
        if self.embedding_query_template is not None:
            identity["query_template"] = self.embedding_query_template
        if self.embedding_document_prefix:
            identity["document_prefix"] = self.embedding_document_prefix
        if self.embedding_extra_body:
            identity["parameters_sha256"] = hashlib.sha256(
                json.dumps(self.embedding_extra_body, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
        return identity
