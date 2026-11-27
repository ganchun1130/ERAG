import json
import sys
from types import ModuleType

import httpx
import numpy as np
import pytest

from erag import ERAGEngine, Extensions, Settings
from erag.extensions import load_extensions
from erag.ingestion import DocumentPipeline
from erag.integrations.external_knowledge import create_extensions
from erag.knowledge import KnowledgeBase
from erag.providers import Providers
from erag.schemas import Hit


def test_json_profile_precedence_and_unknown_fields(tmp_path, monkeypatch):
    config = tmp_path / "profile.json"
    config.write_text(json.dumps({"backend": "vllm", "final_top_k": 8, "llm_model": "own-alias"}))
    dotenv = tmp_path / ".env"
    dotenv.write_text("ERAG_FINAL_TOP_K=7\n")
    assert Settings(config_file=config, _env_file=None).final_top_k == 8
    assert Settings(config_file=config, _env_file=dotenv).final_top_k == 8
    monkeypatch.setenv("ERAG_FINAL_TOP_K", "6")
    assert Settings(config_file=config, _env_file=dotenv).final_top_k == 6
    assert Settings(config_file=config, _env_file=dotenv, final_top_k=5).final_top_k == 5
    monkeypatch.setenv("ERAG_CONFIG_FILE", str(config))
    assert Settings(_env_file=None).llm_model == "own-alias"
    config.write_text('{"final_top_kk": 8}')
    with pytest.raises(ValueError, match="未知"):
        Settings(_env_file=None)


def test_profile_can_clear_vendor_parameters_from_dotenv(tmp_path):
    config, dotenv = tmp_path / "profile.json", tmp_path / ".env"
    config.write_text('{"backend":"vllm","llm_extra_body":{}}')
    dotenv.write_text('ERAG_BACKEND=ark\nERAG_LLM_EXTRA_BODY={"thinking":{"type":"enabled"}}\n')
    settings = Settings(config_file=config, _env_file=dotenv)
    assert settings.backend == "vllm"
    assert settings.llm_extra_body == {}


def test_factory_loaded_once_and_optional_capabilities_closed_once(settings, monkeypatch):
    module = ModuleType("custom_erag_plugin")
    calls = []

    class Model:
        def chat(self, messages, **kwargs):
            return "custom reply"

        def embed(self, texts, **kwargs):
            return [[3, 4] for _ in texts]

        def identity(self):
            return {"model": "company-v1", "revision": "exact-weights"}

        def close(self):
            calls.append("closed")

    def factory(settings):
        calls.append("loaded")
        model = Model()
        return Extensions(chat=model, embedding=model)

    module.factory = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    settings.extension_factory = module.__name__ + ":factory"
    engine = ERAGEngine(settings)
    assert engine.ask("hello").answer == "custom reply"
    np.testing.assert_allclose(engine.providers.embed(["hello"]), [[0.6, 0.8]])
    assert engine.providers.embedding_identity()["model"] == "company-v1"
    engine.close()
    engine.close()
    assert calls == ["loaded", "closed"]


@pytest.mark.parametrize(
    "extensions",
    [Extensions(api_version=2), Extensions(chat=object()), Extensions(upload_extensions=["bad"])],
)
def test_invalid_extensions_fail_at_startup(extensions):
    with pytest.raises((TypeError, ValueError)):
        extensions.validate()


def test_broken_factory_is_not_silently_ignored(settings):
    settings.extension_factory = "package_that_is_not_installed:create"
    with pytest.raises(ModuleNotFoundError):
        load_extensions(settings)


def test_custom_ingestion_and_prebuilt_metadata_survive(settings, providers, chunks, tmp_path):
    class Pipeline:
        def prepare(self, files, settings):
            yield chunks[0].model_copy(update={"metadata": {"parser": "custom-ocr", "box": [1, 2, 3, 4]}})

    with ERAGEngine(
        settings, providers, extensions=Extensions(ingestion=Pipeline(), upload_extensions=[".docx"])
    ) as engine:
        result = engine.build_knowledge_base([tmp_path / "not-a-native-format.docx"], "custom")
        assert result["chunks"] == 1
        assert engine.upload_extensions == [".docx"]
        loaded = engine.open_knowledge_base("custom")
        assert loaded.chunks[0].id == chunks[0].id
        assert loaded.chunks[0].metadata["box"] == [1, 2, 3, 4]
        assert engine.ask("问题", "custom").sources[0].chunk.metadata["parser"] == "custom-ocr"


def test_reader_and_splitter_can_be_replaced_independently(settings, tmp_path):
    pipeline = DocumentPipeline(
        reader=lambda path: "Custom document text",
        splitter=lambda text, **kwargs: [(text, "custom chapter", 7)],
    )
    chunks = list(pipeline.prepare([tmp_path / "document.html"], settings))
    assert chunks[0].page == 7 and chunks[0].chapter == "custom chapter"
    assert chunks[0].content == "Custom document text"


def test_invalid_import_never_calls_model_or_replaces_index(settings, providers, chunks):
    engine = ERAGEngine(settings, providers)
    engine.build_chunks(chunks, "existing")
    before = engine.open_knowledge_base("existing").manifest_path.read_bytes()
    providers.embed_calls.clear()
    for invalid in [[chunks[0], chunks[0]], [chunks[0].model_copy(update={"content": " "})]]:
        with pytest.raises(ValueError):
            engine.build_chunks(invalid, "existing")
    assert not providers.embed_calls
    assert engine.open_knowledge_base("existing").manifest_path.read_bytes() == before


def test_jsonl_factory_supports_web_upload_and_cli_ingestion(settings, providers, chunks, tmp_path):
    settings.extension_factory = "erag.integrations.chunks:create_extensions"
    file = tmp_path / "own-chunks.jsonl"
    file.write_text("\n".join(c.model_dump_json() for c in chunks))
    with ERAGEngine(settings, providers) as engine:
        assert ".jsonl" in engine.upload_extensions
        assert engine.build_knowledge_base([file], "prepared")["chunks"] == 2
        assert engine.open_knowledge_base("prepared").chunks == chunks


def test_custom_embedding_identity_protects_index(settings, chunks):
    class Embedder:
        revision = "one"

        def identity(self):
            return {"revision": self.revision}

        def embed(self, texts, **kwargs):
            return [[1, 1] for _ in texts]

    embedder = Embedder()
    providers = Providers(settings, extensions=Extensions(embedding=embedder))
    kb = KnowledgeBase(settings.knowledge_base_root, "custom", providers, settings)
    kb.build(chunks)
    embedder.revision = "two"
    with pytest.raises(ValueError, match="embedding"):
        kb.load()


def test_explicit_space_survives_endpoint_and_alias_migration(settings, providers, chunks):
    settings.embedding_space = "my-weights-v1"
    settings.embedding_query_template = "query: {text}"
    kb = KnowledgeBase(settings.knowledge_base_root, "migrate", providers, settings)
    kb.build(chunks)
    settings.embedding_base_url = "http://new-host/v1"
    settings.embedding_model = "deployment-alias"
    assert kb.load().chunks == chunks
    settings.embedding_query_template = "different: {text}"
    with pytest.raises(ValueError, match="embedding"):
        kb.load()


def test_custom_reranker_cannot_replace_sources(settings, chunks):
    class Reranker:
        def rerank(self, query, hits):
            hits[0].chunk.content = "invented evidence"
            return hits

    providers = Providers(settings, extensions=Extensions(reranker=Reranker()))
    hits = [Hit(chunk=c, score=0.5, sub_query="q", route="custom") for c in chunks]
    with pytest.raises(ValueError, match="来源"):
        providers.rerank("q", hits)
    assert chunks[0].content == "形成性评价用于反馈。"


def test_external_repository_uses_no_local_index_or_embedding(settings, providers, monkeypatch):
    settings.extension_options = {
        "endpoint": "https://knowledge.invalid/retrieval",
        "knowledge_bases": {"外部库": "db-1"},
    }
    extension = create_extensions(settings)
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "records": [
                        {
                            "content": "形成性评价用于反馈。",
                            "title": "评价",
                            "score": 0.9,
                            "metadata": {"source": "manual://assessment", "custom": 1},
                        },
                    ]
                },
            )
        )
    )
    extension.knowledge.client.close()
    extension.knowledge.client = client
    with ERAGEngine(settings, providers, extensions=extension) as engine:
        assert engine.list_knowledge_bases() == ["外部库"]
        assert engine.ask("形成性评价有什么作用？", "外部库").sources[0].chunk.metadata["custom"] == 1
        assert not providers.embed_calls and not settings.knowledge_base_root.exists()
        with pytest.raises(ValueError, match="外部服务"):
            engine.build_chunks([], "外部库")
    assert client.is_closed


def test_knowledge_repository_cache_contract(settings, providers, chunks):
    class Index:
        name, location, chapters = "custom", "memory://custom", ["第一章"]
        loads = 0
        version = 1

        def cache_token(self):
            return self.version

        def load(self):
            self.loads += 1
            return self

    class Repository:
        index = Index()

        def open(self, name, providers, settings):
            return self.index

        def list(self, settings):
            return ["custom"]

    repository = Repository()
    engine = ERAGEngine(settings, providers, extensions=Extensions(knowledge=repository))
    engine.open_knowledge_base("custom")
    engine.open_knowledge_base("custom")
    assert repository.index.loads == 1
    repository.index.version = 2
    engine.open_knowledge_base("custom")
    assert repository.index.loads == 2
    repository.index.version = None
    engine.open_knowledge_base("custom")
    engine.open_knowledge_base("custom")
    assert repository.index.loads == 4


def test_custom_prompts_reach_model(settings, providers, chunks):
    settings.prompts = {
        "answer_rules": "Answer using these sources in English.",
        "chat": "Custom conversational role.",
    }
    engine = ERAGEngine(settings, providers)
    engine.ask("hello")
    assert providers.chat_calls[-1][0]["content"] == settings.prompts["chat"]
    engine.build_chunks(chunks, "notes")
    engine.ask("question", "notes")
    assert any(system.startswith(settings.prompts["answer_rules"]) for system, _ in providers.json_calls)
