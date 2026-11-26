"""Retrieval-only evaluation: explicit labels, no model-generated judge scores."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from pydantic import BaseModel, Field

from .knowledge import KnowledgeBase
from .schemas import QueryText


class EvaluationCase(BaseModel):
    query: QueryText
    relevant_chunk_ids: set[str] = Field(min_length=1)


def ranking_metrics(retrieved: list[str], relevant: set[str], k: int) -> dict[str, float]:
    if k < 1 or not relevant:
        raise ValueError("k 和相关证据数量必须大于 0")
    ranked = list(dict.fromkeys(retrieved))[:k]
    matched = [i for i, identity in enumerate(ranked, 1) if identity in relevant]
    dcg = sum(1 / math.log2(i + 1) for i in matched)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return {
        "recall": len(matched) / len(relevant),
        "precision": len(matched) / k,
        "hit_rate": float(bool(matched)),
        "mrr": 1 / matched[0] if matched else 0.0,
        "ndcg": dcg / ideal,
    }


def evaluate_retrieval(
    kb: KnowledgeBase, cases_path: Path, *, k: int = 5, use_tags: bool = True, rerank: bool = False
) -> dict:
    if k < 1:
        raise ValueError("top-k 必须大于 0")
    raw = cases_path.read_bytes()
    cases = [
        EvaluationCase.model_validate(json.loads(line))
        for line in raw.decode("utf-8-sig").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError("评测集不能为空")
    kb.load()
    known = {chunk.id for chunk in kb.chunks}
    missing = set().union(*(case.relevant_chunk_ids for case in cases)) - known
    if missing:
        raise ValueError(f"评测标签包含当前索引不存在的块 ID: {sorted(missing)[:5]}；请核对索引版本")
    # Capture this snapshot before remote calls; a concurrent rebuild may replace the manifest.
    manifest = kb.manifest
    rows = []
    for case in cases:
        start = perf_counter()
        # Same candidate budget across ablations, independent of the final metric cutoff.
        hits = kb.search(
            case.query, key_query=case.query if use_tags else None, top_k=max(k, kb.settings.final_top_k)
        )
        if rerank and hits:
            # Fail the experiment on service failure instead of silently measuring a fallback.
            hits = kb.providers.rerank(case.query, hits)
        identities = [hit.chunk.id for hit in hits[:k]]
        rows.append(
            {
                "query": case.query,
                "relevant_chunk_ids": sorted(case.relevant_chunk_ids),
                "retrieved_chunk_ids": identities,
                "latency_ms": (perf_counter() - start) * 1000,
                **ranking_metrics(identities, case.relevant_chunk_ids, k),
            }
        )
    metrics = {
        name: float(np.mean([row[name] for row in rows]))
        for name in ("recall", "precision", "hit_rate", "mrr", "ndcg")
    }
    return {
        "count": len(cases),
        "k": k,
        "metrics": metrics,
        "latency_ms": {
            "p50": float(np.percentile([r["latency_ms"] for r in rows], 50)),
            "p95": float(np.percentile([r["latency_ms"] for r in rows], 95)),
        },
        "configuration": {
            "use_tags": use_tags,
            "rerank": rerank,
            "rerank_model": kb.settings.rerank_model if rerank else None,
            "candidate_top_k": max(k, kb.settings.final_top_k),
            "summary_min_score": kb.settings.summary_min_score,
            "tag_min_score": kb.settings.tag_min_score,
        },
        "index": manifest,
        "cases_sha256": hashlib.sha256(raw).hexdigest(),
        "cases": rows,
    }
