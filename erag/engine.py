"""One-model ERAG orchestration.

The thesis names Router, Retrieval Agent and Generation Agent as logical roles. This
implementation keeps those boundaries in the trace, while allowing all calls to use
one configured model (``ERAG_RETRIEVAL_MODEL`` can still override it for cost/latency).
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from .knowledge import KnowledgeBase, read_document, split_semantic
from .providers import Providers
from .schemas import Answer, Chunk, QueryPlan, Review
from .settings import Settings

log = logging.getLogger(__name__)


PLAN_PROMPT = """你是教学助手的动态检索路由与检索规划器。请分析用户问题和对话历史，保留指代情境，采用分治法拆成1-5个可检索子问题，并给出不含情境的关键查询。无需外部资料即可回答的闲聊/常识问题将 needs_retrieval 设为 false。只输出符合字段的JSON对象。"""
REVIEW_PROMPT = """你是教学助手的证据审查器。逐一检查子查询是否被对应资料完整支持；若资料不足，列出需要再次检索的查询。不要凭外部知识补齐。只输出JSON：complete(bool), missing_queries(string[])。"""
ANSWER_PROMPT = """你是严谨、友好的教学助手。仅依据给定资料回答；资料不足时明确说明，不编造。覆盖原问题所有子任务，必要时分点、比较或给出步骤。答案中用[1]、[2]标记对应资料来源。只输出JSON对象：answer(string)。"""
ENRICH_PROMPT = """你是知识库构建专家。为输入知识块生成50-100字摘要和1-3个可检索的中文关键问题标签。保持技术术语准确，不要编造。只输出JSON对象：summary(string), tags(string[])。"""


class ERAGEngine:
    def __init__(self, settings: Settings | None = None, providers: Providers | None = None):
        self.settings = settings or Settings()
        self.providers = providers or Providers(self.settings)

    def _kb(self, name: str) -> KnowledgeBase:
        return KnowledgeBase(self.settings.knowledge_base_root, name, self.providers, self.settings).load()

    def build_knowledge_base(self, files: list[str | Path], name: str | None = None) -> dict:
        if not files: raise ValueError("至少提供一个 PDF/TXT/MD/CSV 文件")
        kb_name = name or Path(files[0]).stem
        chunks: list[Chunk] = []
        for source in files:
            path = Path(source)
            text = read_document(path)
            for i, (content, chapter, page) in enumerate(split_semantic(text)):
                summary, tags = content[:160], [chapter] if chapter else []
                try:
                    enriched = self.providers.json(ENRICH_PROMPT, {"content": content, "chapter": chapter})
                    summary = str(enriched.get("summary") or summary)
                    tags = [str(x) for x in enriched.get("tags", tags) if str(x).strip()][:3]
                except Exception as exc:
                    log.warning("块 %s enrichment 失败，使用离线摘要: %s", i, exc)
                chunks.append(Chunk(id=str(uuid.uuid4()), content=content, summary=summary,
                    tags=tags, source=str(path), chapter=chapter, page=page))
        kb = KnowledgeBase(self.settings.knowledge_base_root, kb_name, self.providers, self.settings)
        count = kb.build(chunks)
        return {"name": kb_name, "chunks": count, "path": str(kb.root), "embedding": self.settings.embedding_identity()}

    def _plan(self, query: str, history: list[dict] | None) -> QueryPlan:
        payload = {"query": query, "history": (history or [])[-self.settings.history_turns:]}
        try:
            data = self.providers.json(PLAN_PROMPT, payload, retrieval=True)
            # Accept both thesis Chinese keys and the new stable schema.
            data = {"needs_retrieval": data.get("needs_retrieval", data.get("是否检索", True)),
                "question_type": data.get("question_type", data.get("问题性质", "其他")),
                "key_query": data.get("key_query", data.get("关键查询", query)),
                "sub_queries": data.get("sub_queries", data.get("子查询序列", [query]))}
            subs = [{"query": x if isinstance(x, str) else x.get("query", query), "chapter": "" if isinstance(x, str) else x.get("chapter", "")} for x in data["sub_queries"]]
            data["sub_queries"] = subs
            return QueryPlan.model_validate(data)
        except Exception as exc:
            log.warning("规划失败，回退单查询: %s", exc)
            return QueryPlan(needs_retrieval=True, key_query=query, sub_queries=[{"query": query}])

    def _answer(self, query: str, plan: QueryPlan, hits, history) -> str:
        context = "\n\n".join(f"[{i}] {h.chunk.content}" for i, h in enumerate(hits, 1))
        context = context[:self.settings.context_max_chars]
        data = {"query": query, "plan": plan.model_dump(), "history": (history or [])[-self.settings.history_turns:], "context": context}
        try:
            return str(self.providers.json(ANSWER_PROMPT, data).get("answer", "")).strip()
        except Exception:
            # A plain completion is useful with providers that do not support JSON mode.
            return self.providers.chat([{"role": "system", "content": ANSWER_PROMPT}, {"role": "user", "content": json.dumps(data, ensure_ascii=False)}])

    def ask(self, query: str, kb_name: str | None = None, history: list[dict] | None = None) -> Answer:
        if not query or not query.strip(): raise ValueError("问题不能为空")
        plan = self._plan(query.strip(), history)
        trace = [{"role": "router", "needs_retrieval": plan.needs_retrieval}]
        if not plan.needs_retrieval or not kb_name:
            answer = self.providers.chat([{"role": "system", "content": "你是专业教学助手，简洁准确回答。"}, {"role": "user", "content": query}])
            return Answer(answer=answer, plan=plan, trace=trace, warnings=["本轮未使用知识库"])
        kb = self._kb(kb_name); hits = []
        for sub in plan.sub_queries:
            hits.extend(kb.search(sub.query, key_query=plan.key_query, chapter=sub.chapter, top_k=self.settings.per_query_top_k))
        # Stable de-duplication and global ordering after chapter-local retrieval.
        unique = {h.chunk.id: h for h in sorted(hits, key=lambda h: h.score, reverse=True)}
        hits = list(unique.values())[:self.settings.final_top_k]
        if self.settings.rerank_enabled and hits:
            try:
                hits = self.providers.rerank(plan.key_query, hits)[:self.settings.final_top_k]
                trace.append({"role": "reranker", "enabled": True, "hits": len(hits)})
            except Exception as exc:
                # Retrieval remains useful when a local reranker is not launched.
                log.warning("重排序不可用，保留向量排序: %s", exc)
                trace.append({"role": "reranker", "enabled": False, "warning": str(exc)})
        trace.append({"role": "retriever", "round": 1, "hits": len(hits)})
        if not self.settings.tct_enabled:
            trace.append({"role": "reviewer", "enabled": False})
        for round_no in range(1, self.settings.max_retrieval_rounds if self.settings.tct_enabled else 1):
            try:
                review = Review.model_validate(self.providers.json(REVIEW_PROMPT, {"query": query, "sub_queries": [s.model_dump() for s in plan.sub_queries], "hits": [h.chunk.model_dump() for h in hits]}))
            except Exception:
                review = Review(complete=True)
            trace.append({"role": "reviewer", "round": round_no, "complete": review.complete, "missing": review.missing_queries})
            if review.complete or not review.missing_queries: break
            for missing in review.missing_queries:
                hits.extend(kb.search(missing, key_query=plan.key_query, top_k=self.settings.per_query_top_k))
            hits = list({h.chunk.id: h for h in sorted(hits, key=lambda h: h.score, reverse=True)}.values())[:self.settings.final_top_k]
            trace.append({"role": "retriever", "round": round_no + 1, "hits": len(hits)})
        answer = self._answer(query, plan, hits, history)
        return Answer(answer=answer, sources=hits, plan=plan, trace=trace)

    def list_knowledge_bases(self) -> list[str]:
        return KnowledgeBase(self.settings.knowledge_base_root, "", self.providers, self.settings).list_names()
