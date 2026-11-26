import pytest

from erag.settings import Settings


def test_backend_defaults_and_explicit_override():
    assert Settings(_env_file=None, backend="vllm").llm_base_url == "http://127.0.0.1:8000/v1"
    explicit = "https://ark.cn-beijing.volces.com/api/v3"
    assert Settings(_env_file=None, backend="vllm", llm_base_url=explicit).llm_base_url == explicit
    assert Settings(_env_file=None, backend="xinference").llm_model == "qwen3-instruct"


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError, match="chunk_overlap"):
        Settings(_env_file=None, chunk_max_chars=100, chunk_overlap=100)
