import json

import pytest

from erag.evaluation import evaluate_retrieval, ranking_metrics
from erag.knowledge import KnowledgeBase


def test_ranking_metrics_do_not_double_count_evidence():
    metrics = ranking_metrics(["wrong", "a", "a", "b"], {"a", "b", "c"}, 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == 0.5
    assert 0 < metrics["ndcg"] < 1
    assert metrics["hit_rate"] == 1
    assert ranking_metrics([], {"a"}, 3)["recall"] == 0


def test_evaluation_is_reproducible_and_never_calls_llm(settings, providers, chunks, tmp_path):
    kb = KnowledgeBase(settings.knowledge_base_root, "教材", providers, settings)
    kb.build(chunks)
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"query": "问题", "relevant_chunk_ids": ["a", "b"]}), encoding="utf-8")
    report = evaluate_retrieval(kb, cases, k=2)
    assert report["metrics"]["recall"] == 1
    assert report["metrics"]["ndcg"] == 1
    assert len(report["cases_sha256"]) == 64
    assert not providers.json_calls and not providers.chat_calls


def test_stale_labels_fail_before_api_calls(settings, providers, chunks, tmp_path):
    kb = KnowledgeBase(settings.knowledge_base_root, "教材", providers, settings)
    kb.build(chunks)
    providers.embed_calls.clear()
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"query": "问题", "relevant_chunk_ids": ["missing"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="不存在"):
        evaluate_retrieval(kb, cases)
    assert not providers.embed_calls
