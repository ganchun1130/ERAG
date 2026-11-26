import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from time import sleep

import httpx
import numpy as np
import pytest
from openai import OpenAI

from erag.providers import Providers, parse_json
from erag.schemas import Hit


@pytest.mark.parametrize("text", ['{"a": 1}', '<think>reasoning</think>```json\n{"a": 1}\n```'])
def test_json_parser_handles_supported_wrappers(text):
    assert parse_json(text) == {"a": 1}


@pytest.mark.parametrize("text", ["[]", '"text"', "invalid"])
def test_json_parser_rejects_non_objects(text):
    with pytest.raises(ValueError):
        parse_json(text)


@pytest.mark.parametrize(
    "backend,expected",
    [
        ("ark", {"thinking": {"type": "disabled"}}),
        ("vllm", {"chat_template_kwargs": {"enable_thinking": False}}),
        ("xinference", {"enable_thinking": False}),
        ("openai", {}),
    ],
)
def test_thinking_parameters_follow_backend(settings, backend, expected):
    settings.backend = backend
    assert Providers(settings)._extra_body() == expected


def test_explicit_thinking_override_preserved(settings):
    settings.backend = "vllm"
    settings.llm_extra_body = {"chat_template_kwargs": {"enable_thinking": True}}
    provider = Providers(settings)
    assert provider._extra_body()["chat_template_kwargs"]["enable_thinking"]
    assert settings.llm_extra_body == {"chat_template_kwargs": {"enable_thinking": True}}


def embedding_provider(settings, handler):
    provider = Providers(settings)
    provider._embedding = OpenAI(
        api_key="test",
        base_url="https://test.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return provider


def test_embedding_reorders_batches_and_only_instructs_queries(settings):
    settings.embedding_batch_size = 2
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        rows = [
            {"object": "embedding", "index": i, "embedding": [i + 1, 1]}
            for i in reversed(range(len(payload["input"])))
        ]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": rows,
                "model": "test",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    provider = embedding_provider(settings, handler)
    vectors = provider.embed(["一", "二", "三"], query=True)
    assert vectors.shape == (3, 2)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1)
    np.testing.assert_allclose(vectors[0], [2**-0.5, 2**-0.5])
    assert len(requests) == 2
    assert requests[0]["input"][0].endswith("\nQuery: 一")
    provider.embed(["原文"])
    assert requests[-1]["input"] == ["原文"]
    provider.close()


@pytest.mark.parametrize(
    "rows",
    [
        [{"index": 0, "embedding": [0, 0]}],
        [{"index": 0, "embedding": []}],
        [{"index": 1, "embedding": [1, 0]}],
        [],
    ],
)
def test_embedding_rejects_invalid_responses(settings, rows):
    provider = embedding_provider(settings, lambda request: httpx.Response(200, json={"data": rows}))
    with pytest.raises(ValueError):
        provider.embed(["问题"])
    provider.close()


def test_multimodal_uses_one_item_per_request_and_pooled_client(settings):
    settings.embedding_protocol = "ark_multimodal"
    provider, requests = Providers(settings), []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"data": {"embedding": [3, 4]}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider._http = client
    assert provider.embed(["甲", "乙"]).shape == (2, 2)
    assert [r["input"] for r in requests] == [
        [{"type": "text", "text": "甲"}],
        [{"type": "text", "text": "乙"}],
    ]
    assert provider._embedding is None
    provider.close()
    assert client.is_closed


@pytest.mark.parametrize(
    "rows",
    [
        [{"index": 0, "score": 1}, {"index": 0, "score": 2}],
        [{"index": 0, "score": 1}, {"index": 1, "score": "NaN"}],
        [{"index": 0, "score": 1}, {"index": 1}],
    ],
)
def test_rerank_rejects_corrupted_results(settings, chunks, rows):
    provider = Providers(settings)
    provider._http = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"results": rows}))
    )
    hits = [Hit(chunk=c, score=0.8, sub_query="问题", route="global") for c in chunks]
    with pytest.raises(ValueError):
        provider.rerank("问题", hits)
    provider.close()


def test_xinference_chat_is_serialized_across_provider_instances(settings, tmp_path):
    settings.backend = "xinference"
    settings.llm_base_url = f"https://{tmp_path.name}.invalid/v1"
    start, counter = Barrier(4), Lock()
    active = peak = 0

    def handler(request):
        nonlocal active, peak
        with counter:
            active += 1
            peak = max(peak, active)
        sleep(0.02)
        with counter:
            active -= 1
        return httpx.Response(
            200,
            json={
                "id": "chat",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.loads(request.content)["messages"][0]["content"],
                        },
                    }
                ],
            },
        )

    def call(index):
        provider = Providers(settings)
        provider._chat = OpenAI(
            api_key="test",
            base_url=settings.llm_base_url,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        try:
            start.wait(timeout=5)
            return provider.chat([{"role": "user", "content": str(index)}])
        finally:
            provider.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(call, range(4))) == ["0", "1", "2", "3"]
    assert peak == 1
