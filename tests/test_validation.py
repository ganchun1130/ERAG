import pytest

from benchmarks.validate_real_models import (
    compare_batching,
    compare_retrieval,
    legacy_filter,
    verify_reused_fixture,
)
from erag.knowledge import KnowledgeBase


def measured_fake(providers, monkeypatch):
    providers.events = []
    original_embed, original_rerank = providers.embed, providers.rerank

    def embed(*args, **kwargs):
        providers.events.append({"operation": "embed"})
        return original_embed(*args, **kwargs)

    def rerank(*args, **kwargs):
        providers.events.append({"operation": "rerank"})
        return original_rerank(*args, **kwargs)

    monkeypatch.setattr(providers, "embed", embed)
    monkeypatch.setattr(providers, "rerank", rerank)


@pytest.mark.parametrize("kind", ["chat", "full"])
def test_validation_requires_external_fixture_before_creating_output(kind, tmp_path, capsys):
    from benchmarks.validate_chat import main as chat_main
    from benchmarks.validate_real_models import main as full_main

    output = tmp_path / "output"
    main = chat_main if kind == "chat" else full_main
    output_flag = "--output" if kind == "chat" else "--output-dir"
    fixture_flag = "--fixture" if kind == "chat" else "--fixture-dir"
    arguments = [output_flag, str(output), "--env-file", str(tmp_path / ".env")]
    with pytest.raises(SystemExit) as exc:
        main(arguments)
    assert exc.value.code == 2 and fixture_flag in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        main([*arguments, fixture_flag, str(tmp_path / "missing")])
    assert exc.value.code == 2 and "测试集缺少文件" in capsys.readouterr().err
    assert not output.exists()


def test_legacy_ablation_reproduces_and_filter(settings, providers, chunks):
    providers.vectors["tag-a"] = [0, 1, 0]
    kb = KnowledgeBase(settings.knowledge_base_root, "教材", providers, settings)
    kb.build(chunks)
    assert [hit.chunk.id for hit in legacy_filter(kb, "问题", 2)] == ["b"]
    assert {hit.chunk.id for hit in kb.search("问题", key_query="问题", top_k=2)} == {"a", "b"}


def test_retrieval_comparison_keeps_labels_and_repeats_separate(settings, providers, chunks, monkeypatch):
    measured_fake(providers, monkeypatch)
    kb = KnowledgeBase(settings.knowledge_base_root, "教材", providers, settings)
    kb.build(chunks)
    result = compare_retrieval(kb, [{"query": "问题", "relevant_chunk_ids": ["a", "b"]}], k=2, repeats=2)
    assert result["unique_queries"] == 1 and result["repeats"] == 2
    assert set(result["methods"]) == {"legacy_and", "summary", "dual", "dual_rerank"}
    for method in result["methods"].values():
        assert method["aggregate"]["metrics"]["recall"] == pytest.approx(1)
        assert method["aggregate"]["measurements"] == 2


def test_batch_comparison_measures_calls_and_equivalent_evidence(settings, providers, chunks, monkeypatch):
    measured_fake(providers, monkeypatch)
    kb = KnowledgeBase(settings.knowledge_base_root, "教材", providers, settings)
    kb.build(chunks)
    result = compare_batching(kb, [f"问题{i}" for i in range(5)], repeats=2)
    assert result["same_evidence"]
    assert all(row["embedding_calls"] == 5 for row in result["sequential"])
    assert all(row["embedding_calls"] == 1 for row in result["batched"])


def test_reused_fixture_checks_content_not_only_chapter_labels(settings, providers, chunks, tmp_path):
    kb = KnowledgeBase(settings.knowledge_base_root, "教材", providers, settings)
    path = tmp_path / "course.md"
    path.write_text("# 第一章\n\n原始规则", encoding="utf-8")
    kb.chunks = [chunks[0].model_copy(update={"content": "# 第一章\n\n原始规则", "page": None})]
    verify_reused_fixture(kb, path)
    path.write_text("# 第一章\n\n改变后的规则", encoding="utf-8")
    with pytest.raises(ValueError, match="不能混用评测标签"):
        verify_reused_fixture(kb, path)
