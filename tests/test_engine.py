import pytest

from erag.engine import ERAGEngine, merge_hits
from erag.knowledge import KnowledgeBase
from erag.schemas import Hit


def make_engine(settings, providers, chunks):
    KnowledgeBase(settings.knowledge_base_root, "教材", providers, settings).build(chunks)
    return ERAGEngine(settings, providers)


def hit(chunk, score, query="问题"):
    return Hit(chunk=chunk, score=score, sub_query=query, route="global")


def test_dedup_preserves_best_score_and_query(chunks):
    merged = merge_hits([hit(chunks[0], 0.9, "最佳问题"), hit(chunks[1], 0.8), hit(chunks[0], 0.1)])
    assert [(h.chunk.id, h.score) for h in merged] == [("a", 0.9), ("b", 0.8)]
    assert merged[0].sub_query == "最佳问题"


def test_history_zero_disables_history(settings, providers):
    settings.history_turns = 0
    engine = ERAGEngine(settings, providers)
    engine.ask("你好", history=[{"role": "user", "content": "旧问题"}])
    assert len(providers.chat_calls[0]) == 2
    assert not providers.json_calls  # No planner needed without a knowledge base.


def test_history_counts_turns_and_bounds_text(settings, providers):
    settings.history_turns = 1
    engine = ERAGEngine(settings, providers)
    history = [{"role": role, "content": str(i)} for i in range(3) for role in ["user", "assistant"]]
    engine.ask("继续", history=history)
    assert providers.chat_calls[0][1:-1] == history[-2:]
    settings.history_max_chars = 1000
    assert sum(len(m["content"]) for m in engine._history([{"role": "user", "content": "文" * 2000}])) <= 1000


def test_planner_and_generator_receive_history(settings, providers, chunks):
    engine = make_engine(settings, providers, chunks)
    history = [{"role": "user", "content": "解释形成性评价"}, {"role": "assistant", "content": "用于反馈"}]
    engine.ask("它有什么作用？", "教材", history)
    assert providers.json_calls[0][1]["history"] == history
    assert providers.json_calls[-1][1]["history"] == history
    assert providers.json_calls[0][1]["chapters"] == ["第一章", "第二章"]


def test_invalid_plan_falls_back_with_warning(settings, providers, chunks):
    engine = make_engine(settings, providers, chunks)
    providers.responses.append({"needs_retrieval": "false", "sub_queries": None})
    result = engine.ask("问题", "教材")
    assert result.plan.needs_retrieval
    assert any("规划失败" in w for w in result.warnings)


def test_no_evidence_does_not_generate_unsupported_answer(settings, providers, chunks):
    engine = make_engine(settings, providers, chunks)
    providers.vectors["无关问题"] = [0, 1, 0]
    result = engine.ask("无关问题", "教材")
    assert not result.sources
    assert "没有找到" in result.answer
    assert len(providers.json_calls) == 1


def test_context_budget_keeps_whole_chunks_and_consistent_sources(settings, providers, chunks):
    settings.context_max_chars = 1000
    engine = ERAGEngine(settings, providers)
    candidates = [
        hit(chunks[0].model_copy(update={"content": "甲" * 900}), 1),
        hit(chunks[1].model_copy(update={"content": "乙" * 200}), 0.9),
        hit(chunks[1].model_copy(update={"id": "c", "content": "丙"}), 0.8),
    ]
    context, selected = engine._evidence(candidates)
    assert len(context) <= 1000
    assert [h.chunk.id for h in selected] == ["a", "c"]
    assert "[1]" in context and "[2]" in context and "[3]" not in context
    assert "乙" not in context and "甲" * 900 in context


def test_review_failure_is_not_reported_as_complete(settings, providers, chunks):
    settings.tct_enabled = True
    engine = make_engine(settings, providers, chunks)
    providers.responses.extend(
        [{"key_query": "问题", "sub_queries": ["问题"]}, RuntimeError("review offline")]
    )
    result = engine.ask("问题", "教材")
    reviewer = next(t for t in result.trace if t["role"] == "reviewer")
    assert reviewer["complete"] is None
    assert any("审查失败" in w for w in result.warnings)


def test_feedback_never_merges_reranker_and_vector_scores(settings, providers, chunks, monkeypatch):
    settings.tct_enabled = settings.rerank_enabled = True
    engine = make_engine(settings, providers, chunks)
    kb = engine._kb("教材")
    batches = iter([[hit(chunks[0], 0.7)], [hit(chunks[1], 0.9), hit(chunks[0], 0.8)]])
    monkeypatch.setattr(kb, "search_many", lambda *args, **kwargs: next(batches))
    inputs = []

    def rerank(query, hits):
        inputs.append([(h.chunk.id, h.score) for h in hits])
        return [h.model_copy(update={"score": 100 - i}) for i, h in enumerate(hits)]

    monkeypatch.setattr(providers, "rerank", rerank)
    providers.responses.extend(
        [
            {"key_query": "问题", "sub_queries": ["问题"]},
            {"complete": False, "missing_queries": ["补充"]},
            {"complete": True},
        ]
    )
    result = engine.ask("问题", "教材")
    assert inputs[0] == [("a", 0.7)]
    assert inputs[1:] and all(batch == [("b", 0.9), ("a", 0.8)] for batch in inputs[1:])
    assert [h.chunk.id for h in result.sources] == ["b", "a"]


@pytest.mark.parametrize("missing,reason", [("问题", "repeated_queries"), ("补充", "no_new_evidence")])
def test_feedback_stops_without_progress(settings, providers, chunks, missing, reason):
    settings.tct_enabled = True
    engine = make_engine(settings, providers, chunks)
    providers.responses.extend(
        [{"key_query": "问题", "sub_queries": ["问题"]}, {"complete": False, "missing_queries": [missing]}]
    )
    result = engine.ask("问题", "教材")
    assert any(t.get("stop") == reason for t in result.trace)
    assert result.warnings


def test_rerank_receives_all_candidates_before_final_cut(settings, providers, chunks, monkeypatch):
    settings.rerank_enabled, settings.final_top_k = True, 1
    engine = make_engine(settings, providers, chunks)
    seen = []

    def rerank(query, hits):
        seen.extend(hits)
        return list(reversed(hits))

    monkeypatch.setattr(providers, "rerank", rerank)
    assert len(engine.ask("问题", "教材").sources) == 1
    assert len(seen) == 2


def test_cache_reused_and_invalidated_after_rebuild(settings, providers, chunks):
    engine = make_engine(settings, providers, chunks)
    first = engine._kb("教材")
    assert engine._kb("教材") is first
    KnowledgeBase(settings.knowledge_base_root, "教材", providers, settings).build(chunks[:1])
    assert engine._kb("教材") is not first
    assert len(engine._kb("教材").chunks) == 1
    assert len(first.chunks) == 2  # In-flight readers retain their original snapshot.


def test_enrichment_validation_and_deterministic_ids(settings, providers, tmp_path):
    settings.enrich_enabled = True
    document = tmp_path / "资料.txt"
    document.write_text("有效内容。", encoding="utf-8")
    engine = ERAGEngine(settings, providers)
    providers.responses.append({"summary": "摘要", "tags": "不是数组"})
    built = engine.build_knowledge_base([document], "教材")
    assert built["warnings"]
    first = engine._kb("教材").chunks[0]
    assert first.tags == ["未分类"]
    settings.enrich_enabled = False
    engine.build_knowledge_base([document, document], "教材")
    assert engine._kb("教材").chunks[0].id == first.id
    assert len(engine._kb("教材").chunks) == 1


def test_enrichment_repairs_schema_once_without_truncating_tags(settings, providers, tmp_path):
    settings.enrich_enabled = True
    document = tmp_path / "资料.txt"
    document.write_text("有效内容。", encoding="utf-8")
    providers.responses.extend(
        [{"summary": "摘要", "tags": ["甲", "乙", "丙", "丁"]}, {"summary": "摘要", "tags": ["综合问题"]}]
    )
    engine = ERAGEngine(settings, providers)
    built = engine.build_knowledge_base([document], "教材")
    assert not built["warnings"] and built["enrichment_repairs"] == 1
    assert engine._kb("教材").chunks[0].tags == ["综合问题"]
    assert len(providers.json_calls) == 2
    assert "validation_error" in providers.json_calls[1][1]


def test_enrichment_transport_error_does_not_add_another_retry_layer(settings, providers, tmp_path):
    settings.enrich_enabled = True
    document = tmp_path / "资料.txt"
    document.write_text("有效内容。", encoding="utf-8")
    providers.responses.append(RuntimeError("service unavailable"))
    built = ERAGEngine(settings, providers).build_knowledge_base([document], "教材")
    assert len(providers.json_calls) == 1
    assert built["enrichment_failures"] == [
        {"source": str(document), "index": 0, "error_type": "RuntimeError"}
    ]


def test_empty_structured_answer_retries_plain_text(settings, providers, chunks):
    engine = make_engine(settings, providers, chunks)
    providers.responses.extend(
        [{"key_query": "问题", "sub_queries": ["问题"]}, {"answer": " "}, {"answer": "这是回答。[1]"}]
    )
    result = engine.ask("问题", "教材")
    assert result.answer == "这是回答。[1]"
    assert providers.json_calls[-1][1]["draft"] == "这是回答。"
    assert "直接输出正文" in providers.chat_calls[0][0]["content"]


def test_invalid_citation_is_visible(settings, providers, chunks):
    engine = make_engine(settings, providers, chunks)
    providers.responses.extend(
        [{"key_query": "问题", "sub_queries": ["问题"]}, {"answer": "结论[999]"}, {"answer": "仍错误[999]"}]
    )
    assert any("不存在的来源编号" in w for w in engine.ask("问题", "教材").warnings)
    assert len(providers.json_calls) == 3  # One bounded repair, never an unbounded generation loop.


@pytest.mark.parametrize("draft", ["结论", "结论[999]"])
def test_citations_are_repaired_using_same_evidence(settings, providers, chunks, draft):
    engine = make_engine(settings, providers, chunks)
    providers.responses.extend(
        [{"key_query": "问题", "sub_queries": ["问题"]}, {"answer": draft}, {"answer": "结论[1]"}]
    )
    result = engine.ask("问题", "教材")
    assert result.answer == "结论[1]" and not result.warnings
    assert result.trace[-1] == {"role": "citation_repair", "valid": True}
    original, repair = [payload for _, payload in providers.json_calls[1:]]
    assert repair["context"] == original["context"]
    assert repair["draft"] == draft


def test_failed_citation_repair_preserves_answer_and_warning(settings, providers, chunks):
    engine = make_engine(settings, providers, chunks)
    providers.responses.extend(
        [{"key_query": "问题", "sub_queries": ["问题"]}, {"answer": "结论"}, RuntimeError("offline")]
    )
    result = engine.ask("问题", "教材")
    assert result.answer == "结论"
    assert any("未标注来源" in warning for warning in result.warnings)
    assert result.trace[-1] == {"role": "citation_repair", "valid": False}


def test_context_manager_closes_clients(settings, providers):
    with ERAGEngine(settings, providers) as engine:
        assert engine.providers is providers
    assert providers.closed


def test_resolved_subquery_is_used_for_reranking(settings, providers, chunks, monkeypatch):
    settings.rerank_enabled = True
    providers.responses.append({"key_query": "它的分值呢？", "sub_queries": ["形成性评价的分值是多少？"]})
    seen = []
    monkeypatch.setattr(providers, "rerank", lambda q, hits: seen.append(q) or hits)
    result = make_engine(settings, providers, chunks).ask("它的分值呢？", "教材")
    assert result.plan.key_query == "形成性评价的分值是多少？"
    assert seen == ["形成性评价的分值是多少？"]


def test_unresolved_followup_gets_one_bounded_repair(settings, providers, chunks, monkeypatch):
    settings.rerank_enabled = True
    providers.responses.extend(
        [
            {"key_query": "它何时进行？", "sub_queries": [{"query": "它何时进行？", "chapter": "错误章节"}]},
            {"query": "形成性评价何时进行？"},
        ]
    )
    seen = []
    monkeypatch.setattr(providers, "rerank", lambda q, hits: seen.append(q) or hits)
    history = [
        {"role": "user", "content": "形成性评价占多少分？"},
        {"role": "assistant", "content": "35分[1]。"},
    ]
    result = make_engine(settings, providers, chunks).ask("它何时进行？", "教材", history=history)
    assert result.plan.key_query == "形成性评价何时进行？"
    assert result.plan.sub_queries[0].chapter == ""
    assert seen == ["形成性评价何时进行？"]
    assert providers.json_calls[1][1]["previous_user"] == history[0]["content"]


def test_failed_reference_repair_preserves_plan_and_warns(settings, providers, chunks):
    providers.responses.extend([{"key_query": "它呢？", "sub_queries": ["它呢？"]}, RuntimeError("offline")])
    result = make_engine(settings, providers, chunks).ask(
        "它呢？", "教材", history=[{"role": "user", "content": "形成性评价"}]
    )
    assert result.plan.key_query == "它呢？"
    assert any("指代修复失败" in warning for warning in result.warnings)
    assert len(providers.json_calls) == 3  # Planner, one repair, answer.


def test_gap_ranking_promotes_existing_candidate_without_mixing_score_scales(
    settings, providers, chunks, monkeypatch
):
    settings.tct_enabled = settings.rerank_enabled = True
    settings.final_top_k = 2
    third = chunks[1].model_copy(update={"id": "c", "content": "方案 C19 需提交实验日志"})
    all_chunks = [*chunks, third]
    engine = make_engine(settings, providers, all_chunks)
    pool = [hit(c, 0.9 - i * 0.1) for i, c in enumerate(all_chunks)]
    monkeypatch.setattr(engine._kb("教材"), "search_many", lambda *args, **kwargs: pool)
    monkeypatch.setattr(
        providers, "rerank", lambda q, hits: list(reversed(hits)) if q == "方案 C19" else hits
    )
    providers.responses.extend(
        [
            {"key_query": "问题", "sub_queries": ["问题"]},
            {"complete": False, "missing_queries": ["方案 C19"]},
            {"complete": True},
        ]
    )
    result = engine.ask("问题", "教材")
    assert [h.chunk.id for h in result.sources] == ["a", "c"]
    assert any(t["role"] == "coverage_ranker" for t in result.trace)
    assert not any(t.get("stop") == "no_new_evidence" for t in result.trace)


def test_content_ids_survive_source_relocation_and_duplicate_copies(settings, providers, tmp_path):
    original, copy = tmp_path / "原件.txt", tmp_path / "副本.txt"
    for path in [original, copy]:
        path.write_text("相同教学资料。", encoding="utf-8")
    engine = ERAGEngine(settings, providers)
    engine.build_knowledge_base([original], "教材")
    identity = engine._kb("教材").chunks[0].id
    engine.build_knowledge_base([copy, original], "教材")
    assert len(engine._kb("教材").chunks) == 1
    assert engine._kb("教材").chunks[0].id == identity


def test_empty_file_is_rejected_before_any_paid_call(settings, providers, tmp_path):
    settings.enrich_enabled = True
    valid, empty = tmp_path / "资料.txt", tmp_path / "空.txt"
    valid.write_text("有效资料。", encoding="utf-8")
    empty.touch()
    with pytest.raises(ValueError, match="没有可提取文本"):
        ERAGEngine(settings, providers).build_knowledge_base([valid, empty], "教材")
    assert not providers.json_calls and not providers.embed_calls


def test_focused_review_passes_unresolved_gaps_to_answer_and_citation_repair(settings, providers, chunks):
    settings.tct_enabled = True
    settings.tct_mode = "focused"
    providers.responses.extend(
        [
            {"key_query": "问题", "sub_queries": ["问题"]},
            {"complete": False, "missing_queries": ["问题"]},
            {"answer": "资料没有说明原因"},
            {"answer": "资料没有说明原因[1]"},
        ]
    )
    result = make_engine(settings, providers, chunks).ask("问题", "教材")
    assert result.answer == "资料没有说明原因[1]"
    assert providers.json_calls[-2][1]["unresolved_questions"] == ["问题"]
    assert providers.json_calls[-1][1]["unresolved_questions"] == ["问题"]


def test_raw_build_without_enrichment_reuses_full_content_embedding(settings, providers, tmp_path):
    settings.retrieval_mode = "raw"
    settings.enrich_enabled = False
    source = tmp_path / "long.txt"
    source.write_text("这是原文细节。" * 40, encoding="utf-8")
    engine = ERAGEngine(settings, providers)
    result = engine.build_knowledge_base([source], "教材")
    assert not result["warnings"] and not providers.json_calls
    assert len(providers.embed_calls) == 2  # Content/summary share encoding; tags remain available.
    kb = engine._kb("教材")
    assert kb.chunks[0].summary == kb.chunks[0].content
