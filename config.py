"""Legacy constants kept for old experiment scripts.

New code should use ``erag.Settings``; values are loaded from ``.env`` and never
contain credentials in source control.
"""
from erag.settings import Settings

_s = Settings()
DOUBAO_API_URL = _s.llm_base_url
DOUBAO_API_KEY = _s.llm_api_key.get_secret_value()
DOUBAO_MODEL = _s.llm_model
LLM_API_URL = _s.llm_base_url
LLM_MODEL_UID = _s.llm_model
EMBEDDING_API_URL = _s.embedding_base_url
EMBEDDING_MODEL_UID = _s.embedding_model
RERANKER_API_URL = _s.rerank_base_url
RERANKER_MODEL_UID = _s.rerank_model
KNOWLEDGE_BASE_ROOT = str(_s.knowledge_base_root)
SUB_QUERY_TOP_K = _s.sub_query_top_k
SUB_QUERY_TOP_P = 0.85
KEY_QUERY_TOP_K = _s.per_query_top_k
KEY_QUERY_TOP_P = 0.8
FINAL_DOCS_TOP_K = _s.final_top_k
