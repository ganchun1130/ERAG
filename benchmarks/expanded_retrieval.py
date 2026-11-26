"""Expanded retrieval ablations: fixed public labels, hard negatives, held-out queries.

Run prepare/build first; evaluate freezes all variants before opening test results.
JSONL checkpoints retain every real model output and allow interrupted builds to resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from queue import Queue
from time import perf_counter

import numpy as np

from benchmarks.prepare_t2_fixture import parquet_rows
from erag import ERAGEngine, Settings
from erag.evaluation import ranking_metrics
from erag.knowledge import KnowledgeBase
from erag.lexical import BM25, reciprocal_rank_fusion
from erag.providers import Providers
from erag.resources import downloaded_path
from erag.schemas import Chunk, Hit


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    corpus_root = downloaded_path(args.resources_root, "retrieval-data")
    labels_root = downloaded_path(args.resources_root, "retrieval-labels")
    corpus = {
        str(r["id"]): " ".join(r["text"].split()) for r in parquet_rows(corpus_root, "data/corpus-*.parquet")
    }
    corpus = {k: v for k, v in corpus.items() if 0 < len(v) <= 650}
    queries = {str(r["id"]): r["text"] for r in parquet_rows(corpus_root, "data/queries-*.parquet")}
    relevant = defaultdict(set)
    for row in parquet_rows(labels_root, "data/dev-*.parquet"):
        if row["score"] > 0:
            relevant[str(row["qid"])].add(str(row["pid"]))
    excluded = set()
    if args.exclude:
        excluded.update(json.loads(args.exclude.read_text())["benchmark_selection"]["query_ids"])
    eligible = sorted(
        q
        for q, ids in relevant.items()
        if q not in excluded and ids <= corpus.keys() and 0 < len(queries[q]) <= 1000
    )
    rng = random.Random(args.seed)
    selected = rng.sample(eligible, args.queries)
    ids = sorted(corpus)
    lexical = BM25([corpus[i] for i in ids])
    positives = set().union(*(relevant[q] for q in selected))
    hard = set()
    for q in selected:
        ranking = np.argsort(-lexical.scores(queries[q]), kind="stable")
        negatives = [ids[int(i)] for i in ranking if ids[int(i)] not in relevant[q]][: args.hard_negatives]
        hard.update(negatives)
    random_docs = set(rng.sample(sorted(corpus.keys() - positives - hard), args.distractors))
    documents = sorted(positives | hard | random_docs)
    fixture = {
        "seed": args.seed,
        "official_split": "dev",
        "limitation": "Length-filtered reduced corpus; unjudged passages count as nonrelevant. Internal holdout, not official test.",
        "selection": {
            "queries": len(selected),
            "documents": len(documents),
            "positive": len(positives),
            "hard": len(hard - positives),
            "random": len(random_docs),
            "max_chars": 650,
            "excluded_previous_queries": sorted(excluded),
        },
        "documents": [{"id": i, "text": corpus[i]} for i in documents],
        "queries": [
            {
                "id": q,
                "text": queries[q],
                "relevant": sorted(relevant[q]),
                "split": "development" if n < args.development else "test",
            }
            for n, q in enumerate(selected)
        ],
    }
    save(args.output / "fixture.json", fixture)
    print(json.dumps(fixture["selection"]), flush=True)


def build(args, settings):
    fixture = json.loads((args.output / "fixture.json").read_text())
    checkpoint = args.output / "enrichment.jsonl"
    completed = (
        {row["id"]: row for row in (json.loads(line) for line in checkpoint.read_text().splitlines())}
        if checkpoint.exists()
        else {}
    )
    pending = [d for d in fixture["documents"] if d["id"] not in completed]

    models = args.models.split(",") if args.models else [settings.llm_model]
    events = Queue()

    def enrich_shard(model, documents):
        configured = settings.model_copy(update={"llm_model": model, "timeout": 300})
        try:
            with ERAGEngine(configured) as engine:
                for document in documents:
                    started = perf_counter()
                    try:
                        enriched, repairs = engine._enrich(document["text"], "")
                        events.put(
                            {
                                "id": document["id"],
                                "model_uid": model,
                                **enriched.model_dump(),
                                "repairs": repairs,
                                "elapsed_s": perf_counter() - started,
                            }
                        )
                    except Exception as exc:
                        events.put({"id": document["id"], "error": str(exc), "model_uid": model})
        finally:
            events.put(None)

    failures = []
    with (
        checkpoint.open("a", encoding="utf-8") as stream,
        ThreadPoolExecutor(max_workers=min(args.workers, len(models))) as pool,
    ):
        futures = [
            pool.submit(enrich_shard, model, pending[i :: len(models)]) for i, model in enumerate(models)
        ]
        finished = 0
        while finished < len(models):
            row = events.get()
            if row is None:
                finished += 1
                continue
            if "error" in row:
                failures.append(row)
                save(args.output / "enrichment-failures.json", failures)
                continue
            completed[row["id"]] = row
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if len(completed) % 25 == 0:
                print(f"enrichment {len(completed)}/{len(fixture['documents'])}", flush=True)
        for future in futures:
            future.result()
    if failures:
        raise RuntimeError(
            f"{len(failures)} enrichment failures; successful outputs saved, rerun build to retry missing documents"
        )
    chunks = [
        Chunk(
            id=d["id"],
            content=d["text"],
            summary=completed[d["id"]]["summary"],
            tags=completed[d["id"]]["tags"],
            source="T2Retrieval",
            chapter="",
        )
        for d in fixture["documents"]
    ]
    providers = Providers(settings)
    try:
        kb = KnowledgeBase(args.output, "index", providers, settings)
        kb.build(chunks)
        np.save(args.output / "content.npy", kb.content_vectors)
        np.save(
            args.output / "queries.npy", providers.embed([q["text"] for q in fixture["queries"]], query=True)
        )
        save(
            args.output / "build.json",
            {
                "chunks": len(chunks),
                "repairs": sum(r["repairs"] for r in completed.values()),
                "embedding": settings.embedding_identity(),
                "fixture_sha256": hashlib.sha256((args.output / "fixture.json").read_bytes()).hexdigest(),
            },
        )
    finally:
        providers.close()


def bootstrap(deltas, seed=27):
    rng = np.random.default_rng(seed)
    values = np.asarray(deltas)
    estimates = values[rng.integers(0, len(values), size=(4000, len(values)))].mean(axis=1)
    return {
        "difference": float(values.mean()),
        "ci95": np.percentile(estimates, [2.5, 97.5]).tolist(),
        "wins": int((values > 1e-8).sum()),
        "ties": int((abs(values) <= 1e-8).sum()),
        "losses": int((values < -1e-8).sum()),
    }


def evaluate(args, settings):
    fixture = json.loads((args.output / "fixture.json").read_text())
    fixture_sha = hashlib.sha256((args.output / "fixture.json").read_bytes()).hexdigest()
    build_info = json.loads((args.output / "build.json").read_text())
    if (
        build_info["fixture_sha256"] != fixture_sha
        or build_info["embedding"] != settings.embedding_identity()
    ):
        raise ValueError("Fixture or embedding settings changed since index construction")
    providers = Providers(settings)
    kb = KnowledgeBase(args.output, "index", providers, settings).load()
    content, queries = np.load(args.output / "content.npy"), np.load(args.output / "queries.npy")
    if [c.id for c in kb.chunks] != [d["id"] for d in fixture["documents"]]:
        raise ValueError("Index order differs from the fixture")
    if content.shape != kb.content_vectors.shape or queries.shape != (
        len(fixture["queries"]),
        content.shape[1],
    ):
        raise ValueError("Embedding checkpoint dimensions do not match the fixture")
    if not np.array_equal(content, kb.content_vectors):
        raise ValueError("Raw embedding checkpoint differs from the index")
    lexical = BM25([c.content for c in kb.chunks])
    path = args.output / f"results-{args.split}.jsonl"
    source_root = Path(__file__).resolve().parents[1]
    provenance = {
        "fixture_sha256": fixture_sha,
        "source_sha256": {
            str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted([*source_root.glob("erag/*.py"), Path(__file__).resolve()])
        },
        "settings": {k: v for k, v in settings.model_dump(mode="json").items() if "api_key" not in k},
        "packages": {name: version(name) for name in ["numpy", "openai", "httpx"]},
        "candidate_budgets": [15, 50],
        "rrf_constant": 60,
        "seed": args.seed,
        "shards": args.shards,
        "shard_index": args.shard_index,
    }
    metadata_path = args.output / f"provenance-{args.split}.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != provenance:
            raise ValueError("Configuration/source changed; use a new output directory")
    elif path.exists():
        raise ValueError("Existing results have no provenance; use a new output directory")
    save(metadata_path, provenance)
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done = {r["qid"] for r in rows}
    variants = ["dual15", "dual50", "raw50", "hybrid50"]
    try:
        with path.open("a", encoding="utf-8") as stream:
            for qi, query in enumerate(fixture["queries"]):
                if (
                    query["split"] != args.split
                    or query["id"] in done
                    or qi % args.shards != args.shard_index
                ):
                    continue
                started = perf_counter()
                vector = queries[qi]
                summary_scores, tag_scores = vector @ kb.summary_vectors.T, vector @ kb.tag_vectors.T
                raw_scores = vector @ content.T
                dense = [
                    int(i)
                    for i in np.argsort(-raw_scores, kind="stable")[:50]
                    if raw_scores[i] >= settings.summary_min_score
                ]
                lexical_scores = lexical.scores(query["text"])
                sparse = [
                    int(i) for i in np.argsort(-lexical_scores, kind="stable")[:50] if lexical_scores[i] > 0
                ]
                hybrid = [i for i, _ in reciprocal_rank_fusion([dense, sparse])][:50]
                methods = {}
                for budget in [15, 50]:
                    chosen = kb._candidates(summary_scores, tag_scores, np.arange(len(kb.chunks)), budget)
                    blended = 0.7 * summary_scores + 0.3 * tag_scores
                    chosen = sorted(chosen.tolist(), key=lambda i: -blended[i])[:budget]
                    methods[f"dual{budget}"] = chosen
                methods.update(raw50=dense, hybrid50=hybrid)
                row = {
                    "qid": query["id"],
                    "query": query["text"],
                    "relevant": query["relevant"],
                    "methods": {},
                }
                order = variants.copy()
                random.Random(args.seed + qi).shuffle(order)
                for method in order:
                    indices = methods[method]
                    hits = [
                        Hit(
                            chunk=kb.chunks[i],
                            score=float(raw_scores[i]),
                            sub_query=query["text"],
                            route=method,
                        )
                        for i in indices
                    ]
                    before = [h.chunk.id for h in hits]
                    start = perf_counter()
                    ranked = providers.rerank(query["text"], hits)
                    elapsed = (perf_counter() - start) * 1000
                    after = [h.chunk.id for h in ranked]
                    relevant = set(query["relevant"])
                    for label, ranking in [(method, before), (method + "_rerank", after)]:
                        row["methods"][label] = {
                            "ranking": ranking,
                            "metrics": {str(k): ranking_metrics(ranking, relevant, k) for k in [3, 5, 10]},
                            "candidate_recall": len(set(before) & relevant) / len(relevant),
                            "rerank_ms": elapsed if label.endswith("_rerank") else 0,
                            "scores": [h.score for h in ranked] if label.endswith("_rerank") else [],
                        }
                row["elapsed_s"] = perf_counter() - started
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{args.split} {len(rows)} qid={query['id']} {row['elapsed_s']:.1f}s", flush=True)
    finally:
        providers.close()
    summary = {"count": len(rows), "split": args.split, "methods": {}, "paired_vs_dual15": {}}
    for method in rows[0]["methods"]:
        summary["methods"][method] = {
            "metrics": {
                str(k): {
                    m: float(np.mean([r["methods"][method]["metrics"][str(k)][m] for r in rows]))
                    for m in ["recall", "mrr", "ndcg", "hit_rate"]
                }
                for k in [3, 5, 10]
            },
            "candidate_recall": float(np.mean([r["methods"][method]["candidate_recall"] for r in rows])),
            "rerank_ms_p50": float(np.median([r["methods"][method]["rerank_ms"] for r in rows])),
        }
        summary["paired_vs_dual15"][method] = bootstrap(
            [
                r["methods"][method]["metrics"]["5"]["ndcg"] - r["methods"]["dual15"]["metrics"]["5"]["ndcg"]
                for r in rows
            ]
        )
    save(args.output / f"summary-{args.split}.json", summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "build", "evaluate"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resources-root", type=Path, default=Path("resources"))
    p.add_argument("--env-file", type=Path, default=Path(".env"))
    p.add_argument("--exclude", type=Path)
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--development", type=int, default=40)
    p.add_argument("--hard-negatives", type=int, default=2)
    p.add_argument("--distractors", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--models", help="Comma-separated UIDs of identical pinned-model replicas")
    p.add_argument("--seed", type=int, default=20261126)
    p.add_argument("--split", choices=["development", "test"], default="development")
    p.add_argument(
        "--shards", type=int, default=1, help="Independent query shards, using identical model replicas"
    )
    p.add_argument("--shard-index", type=int, default=0)
    args = p.parse_args()
    if not 0 <= args.shard_index < args.shards:
        p.error("shard-index must be in [0, shards)")
    if args.stage == "prepare":
        prepare(args)
    else:
        settings = Settings(_env_file=args.env_file)
        (build if args.stage == "build" else evaluate)(args, settings)


if __name__ == "__main__":
    main()
