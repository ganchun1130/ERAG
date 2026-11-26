import numpy as np
import pytest

from erag.knowledge import KnowledgeBase
from erag.lexical import BM25, reciprocal_rank_fusion
from erag.schemas import SubQuery


def test_raw_recall_recovers_fact_omitted_by_summary(settings, providers, chunks):
    settings.retrieval_mode = "raw"
    providers.vectors.update({chunks[0].content: [0, 1, 0], chunks[1].content: [1, 0, 0], "细节": [0, 1, 0]})
    kb = KnowledgeBase(settings.knowledge_base_root, "教材", providers, settings)
    kb.build(chunks)
    loaded = KnowledgeBase(settings.knowledge_base_root, "教材", providers, settings).load()
    assert loaded.search("细节", top_k=1)[0].chunk.id == "a"
    assert loaded.manifest["version"] == 4
    settings.retrieval_mode = "dual"
    assert not loaded.search("细节", top_k=1)


def test_hybrid_exact_terms_can_recover_dense_miss(settings, providers, chunks):
    settings.retrieval_mode = "hybrid"
    enriched = chunks[1].model_copy(update={"content": "错误 ZX9087 表示授权令牌过期"})
    providers.vectors[enriched.content] = [0, 1, 0]
    kb = KnowledgeBase(settings.knowledge_base_root, "教材", providers, settings)
    kb.build([chunks[0], enriched])
    assert {h.chunk.id for h in kb.search_many([SubQuery(query="ZX9087")], top_k=2)} == {"a", "b"}


def test_legacy_index_requires_explicit_rebuild_for_raw(settings, providers, chunks):
    kb = KnowledgeBase(settings.knowledge_base_root, "教材", providers, settings)
    kb.build(chunks)
    kb.content_vectors = None
    settings.retrieval_mode = "raw"
    with pytest.raises(ValueError, match="v4"):
        kb.search("问题")


def test_bm25_and_rank_fusion_are_deterministic():
    index = BM25(["AB12 课程实验安全", "AB12 软件配置", "完全无关"])
    scores = index.scores("ab12 课程实验")
    assert np.argsort(-scores).tolist() == [0, 1, 2]
    assert index.scores("不存在词语").sum() == 0
    assert reciprocal_rank_fusion([[2, 2, 1], [1, 2]]) == [(1, 1 / 61 + 1 / 62), (2, 1 / 61 + 1 / 62)]


def test_hybrid_does_not_return_zero_score_lexical_matches(settings, providers, chunks):
    settings.retrieval_mode = "hybrid"
    providers.vectors["unmatched-term"] = [0, 1, 0]
    kb = KnowledgeBase(settings.knowledge_base_root, "教材", providers, settings)
    kb.build(chunks)
    assert kb.search("unmatched-term") == []
