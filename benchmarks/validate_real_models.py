"""Run a reproducible real-service smoke test; save raw evidence without auto-grading prose."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from erag import ERAGEngine, Settings
from erag.evaluation import ranking_metrics
from erag.knowledge import KnowledgeBase, read_document, split_semantic
from erag.providers import Providers
from erag.schemas import Hit, SubQuery

ROOT = Path(__file__).resolve().parents[1]


class MeasuredProviders(Providers):
    """Count logical provider calls; SDK-internal HTTP retries are not counted separately."""

    def __init__(self, settings):
        super().__init__(settings)
        self.events = []

    def _measure(self, operation, function, *args, **kwargs):
        started = perf_counter()
        event = {"operation": operation, "success": False}
        try:
            result = function(*args, **kwargs)
            event["success"] = True
            return result
        finally:
            event["elapsed_ms"] = (perf_counter() - started) * 1000
            self.events.append(event)

    def chat(self, *args, **kwargs):
        return self._measure("chat", super().chat, *args, **kwargs)

    def embed(self, *args, **kwargs):
        return self._measure("embed", super().embed, *args, **kwargs)

    def rerank(self, *args, **kwargs):
        return self._measure("rerank", super().rerank, *args, **kwargs)


def legacy_filter(kb: KnowledgeBase, query: str, top_k: int) -> list[Hit]:
    """Reproduce dbf05d8's summary-top-3k then AND-tag filter on the SAME index.

    This is an algorithm ablation, not a claim to run the entire historical project.
    Two separate query encodings intentionally preserve the old call pattern.
    """
    q = kb.providers.embed([query], query=True)
    key = kb.providers.embed([query], query=True)
    scores = (q @ kb.summary_vectors.T)[0]
    tag_scores = (key @ kb.tag_vectors.T)[0]
    candidates = np.argsort(-scores, kind="stable")[: top_k * 3]
    hits = [
        Hit(
            chunk=kb.chunks[int(i)],
            score=float(0.7 * scores[i] + 0.3 * tag_scores[i]),
            sub_query=query,
            route="legacy-and",
        )
        for i in candidates
        if scores[i] >= kb.settings.summary_min_score and tag_scores[i] >= kb.settings.tag_min_score
    ]
    return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]


def summarize(rows: list[dict]) -> dict:
    return {
        "measurements": len(rows),
        "metrics": {
            name: float(np.mean([row[name] for row in rows]))
            for name in ("recall", "precision", "hit_rate", "mrr", "ndcg")
        },
        "latency_ms": {
            "p50": float(np.percentile([row["latency_ms"] for row in rows], 50)),
            "p95": float(np.percentile([row["latency_ms"] for row in rows], 95)),
        },
    }


def compare_retrieval(kb, cases, *, k=3, repeats=3, seed=27):
    providers = kb.providers
    budget = max(k, kb.settings.final_top_k)
    methods = {
        "legacy_and": lambda q: legacy_filter(kb, q, budget),
        "summary": lambda q: kb.search(q, top_k=budget),
        "dual": lambda q: kb.search(q, key_query=q, top_k=budget),
        "dual_rerank": lambda q: providers.rerank(q, kb.search(q, key_query=q, top_k=budget)),
    }
    # Warm up each path; interleave method order afterwards to reduce ordering bias.
    for method in methods.values():
        method(cases[0]["query"])
    rows = {name: [] for name in methods}
    rng = random.Random(seed)
    for repeat in range(repeats):
        order = list(range(len(cases)))
        rng.shuffle(order)
        for case_index in order:
            case = cases[case_index]
            names = list(methods)
            rng.shuffle(names)
            for name in names:
                event_start, started = len(providers.events), perf_counter()
                identities = [h.chunk.id for h in methods[name](case["query"])[:k]]
                rows[name].append(
                    {
                        "query": case["query"],
                        "case_index": case_index,
                        "repeat": repeat,
                        "retrieved_chunk_ids": identities,
                        "latency_ms": (perf_counter() - started) * 1000,
                        "logical_calls": dict(
                            Counter(e["operation"] for e in providers.events[event_start:])
                        ),
                        **ranking_metrics(identities, set(case["relevant_chunk_ids"]), k),
                    }
                )
    # Repeats are latency samples, not additional independent labeled questions.
    return {
        "unique_queries": len(cases),
        "repeats": repeats,
        "k": k,
        "seed": seed,
        "candidate_budget": budget,
        "methods": {name: {"aggregate": summarize(values), "rows": values} for name, values in rows.items()},
    }


def compare_batching(kb, queries, repeats=3, key="星桥课程的评价与实验规则"):
    providers, result = kb.providers, {"sequential": [], "batched": []}
    subs = [SubQuery(query=q) for q in queries[:5]]
    for repeat in range(repeats + 1):
        order = ["sequential", "batched"] if repeat % 2 == 0 else ["batched", "sequential"]
        for method in order:
            event_start, started = len(providers.events), perf_counter()
            if method == "sequential":
                hits = [h for sub in subs for h in kb.search(sub.query, key_query=key)]
            else:
                hits = kb.search_many(subs, key_query=key)
            row = {
                "elapsed_ms": (perf_counter() - started) * 1000,
                "embedding_calls": sum(e["operation"] == "embed" for e in providers.events[event_start:]),
                "hits": [(h.sub_query, h.chunk.id) for h in hits],
            }
            if repeat:
                result[method].append(row)
    result["same_evidence"] = all(
        a["hits"] == b["hits"] for a, b in zip(result["sequential"], result["batched"], strict=True)
    )
    return result


def run_dialogue(settings, questions, kb_name, *, history_turns, tct_enabled):
    configured = settings.model_copy(update={"history_turns": history_turns, "tct_enabled": tct_enabled})
    providers, history, rows = MeasuredProviders(configured), [], []
    with ERAGEngine(configured, providers) as engine:
        for query in questions:
            start, event_start = perf_counter(), len(providers.events)
            answer = engine.ask(query, kb_name, history)
            rows.append(
                {
                    "query": query,
                    "elapsed_ms": (perf_counter() - start) * 1000,
                    "logical_calls": dict(Counter(e["operation"] for e in providers.events[event_start:])),
                    **answer.model_dump(),
                }
            )
            history.extend(
                [{"role": "user", "content": query}, {"role": "assistant", "content": answer.answer}]
            )
    return rows


def verify_reused_fixture(kb, path):
    expected = split_semantic(
        read_document(path), max_chars=kb.settings.chunk_max_chars, overlap=kb.settings.chunk_overlap
    )
    if [(chunk.content, chunk.chapter, chunk.page) for chunk in kb.chunks] != expected:
        raise ValueError("复用索引的原文/分块与当前验证语料不同，不能混用评测标签")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="必须是新的目录，防止覆盖已有实验")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fixture-dir", type=Path, default=ROOT / "examples" / "validation")
    parser.add_argument("--retrieval-only", action="store_true", help="只做建库和检索消融，不运行对话")
    parser.add_argument("--reuse-index", type=Path, help="复用已有知识库目录，在相同索引上比较回答逻辑")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("repeats 必须大于 0")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fixture = json.loads((args.fixture_dir / "cases.json").read_text(encoding="utf-8"))
    settings = Settings(
        _env_file=args.env_file,
        knowledge_base_root=args.output_dir / "knowledge_bases",
        retrieval_mode="dual",
        enrich_enabled=True,
        per_query_top_k=4,
        final_top_k=15,
        context_max_chars=24000,
    )
    providers = MeasuredProviders(settings)
    report = {
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "scope": fixture["description"],
        "python": platform.python_version(),
        "fixture_sha256": hashlib.sha256((args.fixture_dir / "course.md").read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256((args.fixture_dir / "cases.json").read_bytes()).hexdigest(),
        "manual_review_criteria": fixture.get("manual_review_criteria", {}),
        "benchmark_selection": fixture.get("benchmark_selection"),
        "settings": {
            "llm_model": settings.llm_model,
            "embedding": settings.embedding_identity(),
            "rerank_model": settings.rerank_model,
            "summary_min_score": settings.summary_min_score,
            "tag_min_score": settings.tag_min_score,
            "per_query_top_k": settings.per_query_top_k,
            "final_top_k": settings.final_top_k,
            "context_max_chars": settings.context_max_chars,
            "max_retrieval_rounds": settings.max_retrieval_rounds,
            "max_tokens": settings.max_tokens,
            "enable_thinking": settings.enable_thinking,
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted([*ROOT.glob("erag/*.py"), Path(__file__).resolve()])
        },
    }
    resource_lock = ROOT / "resources" / "resources.lock.json"
    if resource_lock.exists():
        report["downloaded_resources"] = json.loads(resource_lock.read_text(encoding="utf-8"))
    report_path = args.output_dir / "report.json"

    def save():
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(report_path)

    try:
        save()
        with ERAGEngine(settings, providers) as engine:
            print("[1/5] " + ("复用已有索引快照" if args.reuse_index else "构建真实模型知识库"), flush=True)
            started = perf_counter()
            if args.reuse_index:
                destination = settings.knowledge_base_root / "validation-course"
                shutil.copytree(args.reuse_index, destination)
                loaded = engine._kb(
                    "validation-course"
                )  # Validate embedding identity and snapshot integrity.
                verify_reused_fixture(loaded, args.fixture_dir / "course.md")
                built = {
                    "name": "validation-course",
                    "chunks": len(loaded.chunks),
                    "reused_from": str(args.reuse_index.resolve()),
                    "warnings": [],
                }
            else:
                built = engine.build_knowledge_base([args.fixture_dir / "course.md"], "validation-course")
            report["build"] = {
                **built,
                "elapsed_ms": (perf_counter() - started) * 1000,
                "logical_calls": dict(Counter(e["operation"] for e in providers.events)),
            }
            if built["warnings"]:
                raise RuntimeError(f"构建发生降级，不能作为完整真实模型验证: {built['warnings']}")
            kb = engine._kb("validation-course")
            cases = []
            for item in fixture["retrieval"]:
                relevant = [c.id for c in kb.chunks if c.chapter in item["chapters"]]
                if len({c.chapter for c in kb.chunks if c.id in relevant}) != len(set(item["chapters"])):
                    raise ValueError(f"语料分块与人工章节标签不一致: {item['query']}")
                cases.append({"query": item["query"], "relevant_chunk_ids": relevant})
            (args.output_dir / "retrieval-cases.jsonl").write_text(
                "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n", encoding="utf-8"
            )
            report["index"] = kb.manifest
            save()
            print(f"[2/5] 检索消融：{len(cases)} 个问题，每种方法重复 {args.repeats} 次", flush=True)
            report["retrieval"] = compare_retrieval(kb, cases, repeats=args.repeats)
            save()
            print("[3/5] 比较逐个检索与批量检索", flush=True)
            report["batching"] = compare_batching(
                kb,
                [c["query"] for c in cases],
                repeats=args.repeats,
                key=fixture.get("batch_key", "星桥课程的评价与实验规则"),
            )
            save()
        report["dialogues"] = {}
        modes = (
            []
            if args.retrieval_only
            else [
                ("without_history", 0, False),
                ("with_history", 5, False),
                ("with_history_tct", 5, True),
            ]
        )
        for name, history_turns, tct_enabled in modes:
            print(f"[4/5] 多轮问答：{name}", flush=True)
            report["dialogues"][name] = run_dialogue(
                settings,
                fixture["dialogue"],
                "validation-course",
                history_turns=history_turns,
                tct_enabled=tct_enabled,
            )
            save()
        if not args.retrieval_only:
            print("[5/5] 资料不足问题", flush=True)
            report["insufficient_evidence"] = run_dialogue(
                settings, [fixture["unanswerable"]], "validation-course", history_turns=0, tct_enabled=True
            )
        report["status"] = (
            "completed_retrieval_only" if args.retrieval_only else "completed_needs_answer_review"
        )
        report["finished_at_utc"] = datetime.now(UTC).isoformat()
        # Capture versions without dumping environment variables or credentials.
        report["packages"] = sorted(
            f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions()
        )
        save()
        print(
            json.dumps(
                {
                    "report": str(report_path),
                    "status": report["status"],
                    "retrieval": {
                        name: data["aggregate"] for name, data in report["retrieval"]["methods"].items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        save()
        raise
    finally:
        providers.close()


if __name__ == "__main__":
    main()
