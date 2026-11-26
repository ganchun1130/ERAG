import json

import numpy as np
import pytest

from erag.knowledge import KnowledgeBase, list_knowledge_bases, read_document, split_semantic
from erag.schemas import SubQuery


def test_split_semantic_keeps_page_and_heading_metadata():
    text = "# 第一章 基础\n\n[第2页]\n定义与原理。\n\n应用场景与限制。"
    chunks = split_semantic(text, max_chars=1000)
    assert len(chunks) == 1
    content, chapter, page = chunks[0]
    assert "定义与原理" in content
    assert chapter.startswith("第一章")
    assert page == 2


def test_split_semantic_splits_large_paragraphs_with_overlap():
    text = "\n\n".join([f"段落{i}：" + "内容" * 40 for i in range(4)])
    chunks = split_semantic(text, max_chars=100, overlap=10)
    assert len(chunks) > 1
    assert all(item[0] for item in chunks)
    assert all(len(item[0]) <= 100 for item in chunks)


def test_single_long_paragraph_is_bounded_without_losing_text():
    text = "abcdefghijklmnopqrstuvwxyz" * 100
    chunks = split_semantic(text, max_chars=100, overlap=10)
    assert max(len(c[0]) for c in chunks) <= 100
    assert chunks[0][0] + "".join(c[0][10:] for c in chunks[1:]) == text


def test_page_and_chapter_boundaries_keep_original_metadata():
    text = "[第1页]\n# 第一章\n甲知识。\n[第2页]\n乙知识。\n# 第二章\n丙知识。"
    chunks = split_semantic(text)
    assert [(c[1], c[2]) for c in chunks] == [("第一章", 1), ("第一章", 2), ("第二章", 2)]
    assert "乙" not in chunks[0][0]
    assert "甲" not in chunks[1][0]
    assert "乙" not in chunks[2][0]


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_split_parameters(size, overlap):
    with pytest.raises(ValueError):
        split_semantic("文本", max_chars=size, overlap=overlap)


def test_csv_retains_headers_and_quoted_fields(tmp_path):
    path = tmp_path / "教材.csv"
    path.write_text('\ufeff概念,含义\n评价,"反馈,改进"\n', encoding="utf-8")
    assert read_document(path) == "概念：评价；含义：反馈,改进"


def test_text_decoding_does_not_silently_drop_bytes(tmp_path):
    path = tmp_path / "invalid.txt"
    path.write_bytes(b"text\xff")
    with pytest.raises(UnicodeDecodeError):
        read_document(path)


@pytest.mark.parametrize("name", ["", ".", "..", "../escape", "/absolute", "a/b", "a\\b", " name", "a\n"])
def test_names_cannot_escape_storage(settings, providers, name):
    with pytest.raises(ValueError):
        KnowledgeBase(settings.knowledge_base_root, name, providers, settings)


def test_symlink_is_rejected(settings, providers, tmp_path):
    settings.knowledge_base_root.mkdir()
    (settings.knowledge_base_root / "linked").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        KnowledgeBase(settings.knowledge_base_root, "linked", providers, settings)


def test_roundtrip_and_legacy_v2(settings, providers, chunks):
    kb = KnowledgeBase(settings.knowledge_base_root, "教材", providers, settings)
    assert kb.build(chunks) == 2
    assert kb.load().chunks == chunks
    manifest = json.loads(kb.manifest_path.read_text())
    snapshot = kb.root / "snapshots" / manifest["snapshot"]
    for name in ["chunks.json", "summary.npy", "tags.npy"]:
        (kb.root / name).write_bytes((snapshot / name).read_bytes())
    manifest["version"] = 2
    kb.manifest_path.write_text(json.dumps(manifest))
    assert kb.load().chunks == chunks
    assert list_knowledge_bases(settings.knowledge_base_root) == ["教材"]


def test_failed_rebuild_preserves_previous_snapshot(settings, providers, chunks, monkeypatch):
    kb = KnowledgeBase(settings.knowledge_base_root, "教材", providers, settings)
    kb.build(chunks)
    manifest = kb.manifest_path.read_bytes()

    def disk_full(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(np, "save", disk_full)
    with pytest.raises(OSError, match="disk full"):
        kb.build(chunks[:1])
    assert kb.manifest_path.read_bytes() == manifest
    assert kb.load().chunks == chunks
    assert len(list((kb.root / "snapshots").iterdir())) == 1


def test_changed_embedding_identity_rejected(settings, providers, chunks):
    kb = KnowledgeBase(settings.knowledge_base_root, "教材", providers, settings)
    kb.build(chunks)
    settings.embedding_revision = "different"
    with pytest.raises(ValueError, match="embedding"):
        kb.load()


@pytest.mark.parametrize("vectors", [np.zeros((2, 3)), np.ones((1, 3)), np.full((2, 3), np.nan)])
def test_corrupted_vectors_rejected(settings, providers, chunks, vectors):
    kb = KnowledgeBase(settings.knowledge_base_root, "教材", providers, settings)
    kb.build(chunks)
    snapshot = json.loads(kb.manifest_path.read_text())["snapshot"]
    np.save(kb.root / "snapshots" / snapshot / "summary.npy", vectors)
    with pytest.raises(ValueError, match="向量"):
        kb.load()


def test_tags_are_independent_recall_not_a_summary_filter(settings, providers, chunks):
    providers.vectors.update(
        {"summary-a": [1, 0, 0], "summary-b": [0, 1, 0], "tag-a": [0, 1, 0], "tag-b": [1, 0, 0]}
    )
    kb = KnowledgeBase(settings.knowledge_base_root, "教材", providers, settings)
    kb.build(chunks)
    hits = kb.search("问题", key_query="关键问题", top_k=2)
    assert {h.chunk.id for h in hits} == {"a", "b"}
    assert hits[0].score == pytest.approx(0.7)


def test_chapter_filter_precedes_top_k(settings, providers, chunks):
    many = [chunks[0].model_copy(update={"id": str(i)}) for i in range(20)] + [chunks[1]]
    providers.vectors["summary-b"] = [0.8, 0.6, 0]
    kb = KnowledgeBase(settings.knowledge_base_root, "教材", providers, settings)
    kb.build(many)
    assert [h.chunk.id for h in kb.search("问题", chapter="第二章", top_k=1)] == ["b"]
    assert kb.search("问题", chapter="不存在的章节", top_k=1)[0].route == "global"


def test_queries_are_batched_and_key_embedding_reused(settings, providers, chunks):
    kb = KnowledgeBase(settings.knowledge_base_root, "教材", providers, settings)
    kb.build(chunks)
    providers.embed_calls.clear()
    cache = {}
    kb.search_many(
        [SubQuery(query="问题1"), SubQuery(query="问题2")], key_query="关键问题", embedding_cache=cache
    )
    kb.search_many([SubQuery(query="问题3")], key_query="关键问题", embedding_cache=cache)
    assert providers.embed_calls == [(["问题1", "问题2", "关键问题"], True), (["问题3"], True)]


def test_query_dimension_mismatch_has_clear_error(settings, providers, chunks):
    kb = KnowledgeBase(settings.knowledge_base_root, "教材", providers, settings)
    kb.build(chunks)
    providers.vectors["问题"] = [1, 0]
    with pytest.raises(ValueError, match="维度"):
        kb.search("问题")
