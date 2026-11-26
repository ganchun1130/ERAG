"""Create a fixed, short-document T2Retrieval subset using original public qrels.

This is a bounded diagnostic, NOT the complete C-MTEB benchmark. Selection uses
only IDs, labels, document lengths and a fixed seed, never model scores.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from erag.resources import downloaded_path


def select_subset(corpus, queries, relevant, *, count=32, distractors=64, max_chars=650, seed=27):
    corpus = {str(k): " ".join(v.split()) for k, v in corpus.items()}
    short = {k for k, text in corpus.items() if 0 < len(text) <= max_chars}
    eligible = sorted(
        qid
        for qid, ids in relevant.items()
        if ids and ids <= short and qid in queries and 0 < len(queries[qid].strip()) <= 1000
    )
    if len(eligible) < count:
        raise ValueError(f"只有 {len(eligible)} 个满足全部相关文档完整保留的查询，需要 {count}")
    rng = random.Random(seed)
    selected = rng.sample(eligible, count)
    positives = set().union(*(relevant[qid] for qid in selected))
    pool = sorted(short - positives)
    if len(pool) < distractors:
        raise ValueError("可用的干扰文档数量不足")
    documents = positives | set(rng.sample(pool, distractors))
    metadata = {
        "source_split": "dev",
        "seed": seed,
        "max_document_chars": max_chars,
        "eligible_queries": len(eligible),
        "query_ids": selected,
        "document_ids": sorted(documents),
        "positive_documents": len(positives),
        "distractor_documents": distractors,
        "normalization": "Collapse whitespace; prepend 正文： to prevent accidental heading parsing.",
        "limitation": "Only queries whose ALL positive documents fit the length bound; reduced corpus. "
        "Not a full-corpus benchmark score or evidence of teaching-answer accuracy.",
    }
    course = "\n\n".join(f"# 文档 {pid}\n\n正文：{corpus[pid]}" for pid in sorted(documents)) + "\n"
    cases = [
        {"query": queries[qid], "chapters": [f"文档 {pid}" for pid in sorted(relevant[qid])]}
        for qid in selected
    ]
    return course, cases, metadata


def parquet_rows(root, pattern):
    import pyarrow.parquet as pq

    files = sorted(Path(root).glob(pattern))
    if not files:
        raise FileNotFoundError(f"没有找到 {root}/{pattern}")
    for path in files:
        yield from pq.read_table(path).to_pylist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources-root", type=Path, default=Path("resources"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--distractors", type=int, default=64)
    parser.add_argument("--max-chars", type=int, default=650)
    parser.add_argument("--seed", type=int, default=27)
    args = parser.parse_args()
    if args.queries < 1 or args.distractors < 0 or not 1 <= args.max_chars <= 750:
        parser.error("queries > 0，distractors >= 0，max-chars 在 1 到 750 之间")
    corpus_root = downloaded_path(args.resources_root, "retrieval-data")
    labels_root = downloaded_path(args.resources_root, "retrieval-labels")
    corpus = {row["id"]: row["text"] for row in parquet_rows(corpus_root, "data/corpus-*.parquet")}
    queries = {row["id"]: row["text"] for row in parquet_rows(corpus_root, "data/queries-*.parquet")}
    relevant = defaultdict(set)
    for row in parquet_rows(labels_root, "data/dev-*.parquet"):
        if row["score"] > 0:
            relevant[row["qid"]].add(row["pid"])
    course, cases, metadata = select_subset(
        corpus,
        queries,
        relevant,
        count=args.queries,
        distractors=args.distractors,
        max_chars=args.max_chars,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fixture = {
        "description": "T2Retrieval dev 固定抽样、短文档及缩小候选语料诊断；原始公开相关性标签；非完整 benchmark。",
        "retrieval": cases,
        "batch_key": "检索与查询直接相关的文档",
        "benchmark_selection": metadata,
    }
    (args.output_dir / "course.md").write_text(course, encoding="utf-8")
    (args.output_dir / "cases.json").write_text(
        json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"queries": len(cases), "documents": len(metadata["document_ids"])}, indent=2))


if __name__ == "__main__":
    main()
