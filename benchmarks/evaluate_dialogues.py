"""Paired multi-turn component ablations using fixed MTRAG histories and labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmarks.expanded_retrieval import save
from benchmarks.validate_real_models import MeasuredProviders
from erag import ERAGEngine, Settings
from erag.evaluation import ranking_metrics
from erag.knowledge import KnowledgeBase
from erag.providers import Providers
from erag.schemas import Chunk


def word_f1(answer, reference):
    def words(text):
        text = re.sub(r"\[\d+\]", "", text.lower())
        return Counter(re.findall(r"[a-z0-9]+", text))

    a, b = words(answer), words(reference)
    matched = sum((a & b).values())
    return 2 * matched / max(sum(a.values()) + sum(b.values()), 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["build", "evaluate"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--env-file", type=Path, default=Path(".env"))
    p.add_argument("--split", choices=["development", "test"], default="development")
    p.add_argument("--model", default="qwen3-eval")
    p.add_argument("--variants", help="Comma-separated variant subset, for independent pinned-model replicas")
    args = p.parse_args()
    settings = Settings(
        _env_file=args.env_file,
        llm_model=args.model,
        retrieval_model=args.model,
        knowledge_base_root=args.output,
        retrieval_mode="raw",
        per_query_top_k=15,
        final_top_k=5,
        context_max_chars=16000,
        max_tokens=2048,
        timeout=180,
    )
    fixture = json.loads((args.output / "fixture.json").read_text())
    if args.stage == "build":
        # TCT ablations keep representation fixed to complete raw passages; no LLM labels or summaries.
        providers = Providers(settings)
        try:
            chunks = [
                Chunk(
                    id=d["id"],
                    content=d["text"],
                    summary=d["text"],
                    tags=[d["title"] or d["text"][:100]],
                    source="MTRAG",
                    chapter="",
                )
                for d in fixture["documents"]
            ]
            KnowledgeBase(args.output, "index", providers, settings).build(chunks)
        finally:
            providers.close()
        return
    variants = {
        "no_history": {"history_turns": 0, "tct_enabled": False, "contextual_ranking": False},
        "history": {"history_turns": 5, "tct_enabled": False, "contextual_ranking": False},
        "resolved_history": {"history_turns": 5, "tct_enabled": False, "contextual_ranking": True},
        "legacy_tct": {
            "history_turns": 5,
            "tct_enabled": True,
            "tct_mode": "legacy",
            "contextual_ranking": False,
        },
        "resolved_legacy_tct": {
            "history_turns": 5,
            "tct_enabled": True,
            "tct_mode": "legacy",
            "contextual_ranking": True,
        },
        "focused_tct": {
            "history_turns": 5,
            "tct_enabled": True,
            "tct_mode": "focused",
            "contextual_ranking": True,
        },
    }
    if args.variants:
        chosen = args.variants.split(",")
        if not set(chosen) <= variants.keys():
            p.error("Unknown variant")
        variants = {name: variants[name] for name in chosen}
    source_root = Path(__file__).resolve().parents[1]
    provenance = {
        "fixture_sha256": hashlib.sha256((args.output / "fixture.json").read_bytes()).hexdigest(),
        "source_sha256": {
            str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted([*source_root.glob("erag/*.py"), Path(__file__).resolve()])
        },
        "settings": {
            key: value for key, value in settings.model_dump(mode="json").items() if "api_key" not in key
        },
        "variants": variants,
    }
    metadata_path = args.output / f"provenance-{args.split}.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != provenance:
        raise ValueError("Configuration/source changed since this run began; use a new output directory")
    save(metadata_path, provenance)
    path = args.output / f"results-{args.split}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done = {(r["session"], r["turn"], r["variant"]) for r in rows}
    with path.open("a", encoding="utf-8") as stream:
        for session in fixture["sessions"]:
            if session["split"] != args.split:
                continue
            for turn_no, turn in enumerate(session["turns"], 1):
                order = list(variants)
                random.Random(f"47-{session['id']}-{turn_no}").shuffle(order)
                for variant in order:
                    if (session["id"], turn_no, variant) in done:
                        continue
                    configured = settings.model_copy(update=variants[variant])
                    providers = MeasuredProviders(configured)
                    with ERAGEngine(configured, providers) as engine:
                        started = perf_counter()
                        answer = engine.ask(turn["query"], "index", turn["history"])
                        elapsed = perf_counter() - started
                    retrieved = [hit.chunk.id for hit in answer.sources]
                    relevant = set(turn["relevant"])
                    row = {
                        "session": session["id"],
                        "turn": turn_no,
                        "variant": variant,
                        "query": turn["query"],
                        "reference_answer": turn["reference_answer"],
                        "relevant": turn["relevant"],
                        "labels": turn["labels"],
                        "answer": answer.answer,
                        "source_ids": retrieved,
                        "plan": answer.plan.model_dump(),
                        "trace": answer.trace,
                        "warnings": answer.warnings,
                        "elapsed_s": elapsed,
                        "word_f1": word_f1(answer.answer, turn["reference_answer"]),
                        "calls": dict(Counter(e["operation"] for e in providers.events)),
                        "retrieval": ranking_metrics(retrieved, relevant, 5) if relevant else None,
                        "all_evidence": set(retrieved) >= relevant if relevant else None,
                    }
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(
                        f"{session['id']} turn={turn_no} {variant} {elapsed:.1f}s recall={row['retrieval']}",
                        flush=True,
                    )
    summary = {"turns": len(rows) // len(variants), "variants": {}}
    for variant in variants:
        subset = [r for r in rows if r["variant"] == variant]
        judged = [r for r in subset if r["retrieval"] is not None]
        summary["variants"][variant] = {
            "count": len(subset),
            "labeled_turns": len(judged),
            "recall5": float(np.mean([r["retrieval"]["recall"] for r in judged])),
            "all_evidence": float(np.mean([r["all_evidence"] for r in judged])),
            "word_f1": float(np.mean([r["word_f1"] for r in subset])),
            "chat_calls": sum(r["calls"].get("chat", 0) for r in subset),
            "elapsed_s": sum(r["elapsed_s"] for r in subset),
            "review_queries": sum(
                len(t.get("missing", [])) for r in subset for t in r["trace"] if t["role"] == "reviewer"
            ),
        }
    save(args.output / f"summary-{args.split}.json", summary)


if __name__ == "__main__":
    main()
