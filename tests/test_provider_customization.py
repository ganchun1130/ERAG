import json

import httpx
import numpy as np
import pytest
from pydantic import SecretStr

from erag import Settings
from erag.providers import Providers


def test_generic_backend_has_no_vendor_defaults():
    settings = Settings(_env_file=None, backend="openai")
    assert settings.llm_base_url == "https://api.openai.com/v1"
    assert not settings.llm_model
    assert Providers(settings)._extra_body() == {}


def test_retrieval_endpoint_never_inherits_main_secrets_or_vendor_body(settings):
    settings.llm_api_key = SecretStr("main-secret")
    settings.llm_headers = {"X-API-Key": SecretStr("main-header-secret")}
    settings.llm_extra_body = {"vendor_parameter": True}
    settings.retrieval_base_url = "https://other.example/v1"
    settings.retrieval_api_key = SecretStr("retrieval-secret")
    provider = Providers(settings)
    client = provider._client(retrieval=True)
    assert client.api_key == "retrieval-secret"
    assert "X-API-Key" not in client.default_headers
    assert provider._extra_body(retrieval=True) == {}
    settings.retrieval_backend = "vllm"
    assert provider._extra_body(retrieval=True) == {"chat_template_kwargs": {"enable_thinking": False}}
    provider.close()


@pytest.mark.parametrize("backend", ["openai", "ollama"])
def test_secondary_endpoint_accepts_its_own_headers(settings, backend):
    settings.llm_headers = {"X-Main-Key": SecretStr("main-secret")}
    settings.retrieval_headers = {"X-Retrieval-Key": SecretStr("retrieval-secret")}
    settings.retrieval_base_url = "https://retrieval.invalid"
    settings.retrieval_backend = backend
    provider = Providers(settings)
    if backend == "openai":
        headers = provider._client(retrieval=True).default_headers
    else:
        seen = []

        def handler(request):
            seen.append(request.headers)
            return httpx.Response(200, json={"message": {"content": "ready"}})

        provider._http = httpx.Client(transport=httpx.MockTransport(handler))
        assert provider.chat([{"role": "user", "content": "hello"}], retrieval=True) == "ready"
        headers = seen[0]
    assert headers["X-Retrieval-Key"] == "retrieval-secret"
    assert "X-Main-Key" not in headers
    provider.close()


def test_ollama_chat_and_embeddings_use_native_protocol(settings):
    settings.backend = "ollama"
    settings.llm_base_url = "http://ollama.invalid"
    settings.embedding_protocol = "ollama"
    settings.embedding_base_url = "http://ollama.invalid"
    settings.embedding_query_template = "query: {text}"
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"content": '{"answer":"done"}'}, "done": True})
        return httpx.Response(200, json={"embeddings": [[3, 4] for _ in body["input"]]})

    provider = Providers(settings)
    provider._http = httpx.Client(transport=httpx.MockTransport(handler))
    assert provider.json("Answer", {}) == {"answer": "done"}
    np.testing.assert_allclose(provider.embed(["a", "b"], query=True), [[0.6, 0.8], [0.6, 0.8]])
    assert requests[0][1]["stream"] is False and requests[0][1]["format"] == "json"
    assert requests[1][1]["input"] == ["query: a", "query: b"]
    assert requests[1][1]["truncate"] is False
    provider.close()


def test_ollama_truncation_is_reported(settings):
    settings.backend = "ollama"
    provider = Providers(settings)
    provider._http = httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"message": {"content": "partial"}, "done_reason": "length"})
        )
    )
    with pytest.raises(ValueError, match="截断"):
        provider.chat([{"role": "user", "content": "question"}])
    provider.close()


def test_custom_rerank_path_headers_and_parameters(settings, chunks):
    settings.rerank_path = "/custom/rank"
    settings.rerank_headers = {"X-Company": SecretStr("company-token")}
    settings.rerank_extra_body = {"instruction": "Rank precisely"}
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"results": [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.2}]})

    provider = Providers(settings)
    provider._http = httpx.Client(transport=httpx.MockTransport(handler))
    from erag.schemas import Hit

    hits = [Hit(chunk=c, score=0, sub_query="q", route="raw") for c in chunks]
    assert provider.rerank("q", hits)[0].chunk.id == chunks[1].id
    assert requests[0].url.path.endswith("/custom/rank")
    assert requests[0].headers["X-Company"] == "company-token"
    assert json.loads(requests[0].content)["instruction"] == "Rank precisely"
    provider.close()


@pytest.mark.parametrize(
    "overrides",
    [
        {"llm_extra_body": {"messages": []}},
        {"embedding_extra_body": {"input": []}},
        {"rerank_extra_body": {"documents": []}},
        {"prompts": {"unknown": "text"}},
        {"embedding_query_template": "no placeholder"},
    ],
)
def test_invalid_customization_fails_before_network_calls(overrides):
    with pytest.raises(ValueError):
        Settings(_env_file=None, **overrides)
