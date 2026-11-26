"""Check complete paired coverage and summarize expanded real-model experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from benchmarks.expanded_retrieval import bootstrap, save


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def summarize_dialogues(directories, output):
    fixtures = [(d / "fixture.json").read_bytes() for d in directories]
    if len({hashlib.sha256(f).hexdigest() for f in fixtures}) != 1:
        raise ValueError("Different dialogue fixtures")
    fixture = json.loads(fixtures[0])
    provenance = [json.loads((d / "provenance-test.json").read_text()) for d in directories]
    if any(p["source_sha256"] != provenance[0]["source_sha256"] for p in provenance):
        raise ValueError("Different dialogue code")
    rows = [r for d in directories for r in load_rows(d / "results-test.jsonl")]
    variants = sorted({v for p in provenance for v in p["variants"]})
    if "history" not in variants:
        raise ValueError("The history baseline must be included")
    expected = {
        (s["id"], n, v)
        for s in fixture["sessions"]
        if s["split"] == "test"
        for n in range(1, len(s["turns"]) + 1)
        for v in variants
    }
    keyed = {(r["session"], r["turn"], r["variant"]): r for r in rows}
    if set(keyed) != expected or len(keyed) != len(rows):
        raise ValueError("Missing or duplicate dialogue measurements")
    sessions = sorted({r["session"] for r in rows})
    report = {
        "conversations": len(sessions),
        "turns": len(rows) // len(variants),
        "variants": {},
        "paired_vs_history": {},
        "fixture_sha256": hashlib.sha256(fixtures[0]).hexdigest(),
    }
    metrics = {
        "recall5": lambda r: r["retrieval"]["recall"] if r["retrieval"] else None,
        "all_evidence": lambda r: r["all_evidence"],
        "word_f1": lambda r: r["word_f1"],
    }
    rng = np.random.default_rng(47)
    samples = rng.integers(0, len(sessions), size=(4000, len(sessions)))
    for variant in variants:
        subset = [r for r in rows if r["variant"] == variant]
        report["variants"][variant] = {
            "count": len(subset),
            "labeled_turns": sum(r["retrieval"] is not None for r in subset),
            "chat_calls": sum(r["calls"].get("chat", 0) for r in subset),
            "elapsed_s": sum(r["elapsed_s"] for r in subset),
            "warnings": sum(bool(r["warnings"]) for r in subset),
        }
        differences = {}
        for name, metric in metrics.items():
            pairs = [(r, metric(r), metric(keyed[r["session"], r["turn"], "history"])) for r in subset]
            pairs = [(r, float(a), float(b)) for r, a, b in pairs if a is not None and b is not None]
            report["variants"][variant][name] = float(np.mean([a for _, a, _ in pairs]))
            sums = np.array([sum(a - b for r, a, b in pairs if r["session"] == s) for s in sessions])
            counts = np.array([sum(r["session"] == s for r, _, _ in pairs) for s in sessions])
            estimates = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
            differences[name] = {
                "difference": float(sums.sum() / counts.sum()),
                "cluster_ci95": np.percentile(estimates, [2.5, 97.5]).tolist(),
            }
        report["paired_vs_history"][variant] = differences
    save(output / "dialogue-summary.json", report)
    # Preselected final turn of each held-out conversation; names hidden from the reviewer.
    documents = {d["id"]: d for d in fixture["documents"]}
    audit, key = [], {}
    for session in fixture["sessions"]:
        if session["split"] != "test":
            continue
        n, turn = len(session["turns"]), session["turns"][-1]
        order = ["history", "resolved_history", "focused_tct"]
        random.Random("blind47-" + session["id"]).shuffle(order)
        item = {
            "session": session["id"],
            "turn": n,
            "query": turn["query"],
            "history": turn["history"],
            "reference": turn["reference_answer"],
            "answers": [],
        }
        for i, variant in enumerate(order):
            identity = f"{session['id']}-{chr(65 + i)}"
            row = keyed[session["id"], n, variant]
            item["answers"].append(
                {
                    "id": identity,
                    "answer": row["answer"],
                    "sources": [documents[k] for k in row["source_ids"]],
                }
            )
            key[identity] = variant
        audit.append(item)
    save(output / "blind-review.json", audit)
    save(output / "blind-review-key.json", key)


def summarize_retrieval(directories, output):
    fixtures = [(d / "fixture.json").read_bytes() for d in directories]
    if len({hashlib.sha256(f).hexdigest() for f in fixtures}) != 1:
        raise ValueError("Different retrieval fixtures")
    metadata = [json.loads((d / "provenance-test.json").read_text()) for d in directories]
    if any(m["source_sha256"] != metadata[0]["source_sha256"] for m in metadata):
        raise ValueError("Different retrieval code")
    rows = [r for d in directories for r in load_rows(d / "results-test.jsonl")]
    fixture = json.loads(fixtures[0])
    expected = {q["id"] for q in fixture["queries"] if q["split"] == "test"}
    if {r["qid"] for r in rows} != expected or len(rows) != len(expected):
        raise ValueError("Missing or duplicate retrieval measurements")
    report = {"count": len(rows), "split": "test", "methods": {}, "paired_vs_dual15": {}}
    for method in rows[0]["methods"]:
        report["methods"][method] = {
            "metrics": {
                str(k): {
                    metric: float(np.mean([r["methods"][method]["metrics"][str(k)][metric] for r in rows]))
                    for metric in ["recall", "mrr", "ndcg", "hit_rate"]
                }
                for k in [3, 5, 10]
            },
            "candidate_recall": float(np.mean([r["methods"][method]["candidate_recall"] for r in rows])),
            "rerank_ms_p50": float(np.median([r["methods"][method]["rerank_ms"] for r in rows])),
        }
        report["paired_vs_dual15"][method] = bootstrap(
            [
                r["methods"][method]["metrics"]["5"]["ndcg"] - r["methods"]["dual15"]["metrics"]["5"]["ndcg"]
                for r in rows
            ]
        )
    report["rerank_effect"] = {}
    for method in ["dual15", "dual50", "raw50", "hybrid50"]:
        report["rerank_effect"][method] = bootstrap(
            [
                r["methods"][method + "_rerank"]["metrics"]["5"]["ndcg"]
                - r["methods"][method]["metrics"]["5"]["ndcg"]
                for r in rows
            ]
        )
    report["paired_vs_dual15_rerank"] = {
        method: bootstrap(
            [
                r["methods"][method]["metrics"]["5"]["ndcg"]
                - r["methods"]["dual15_rerank"]["metrics"]["5"]["ndcg"]
                for r in rows
            ]
        )
        for method in rows[0]["methods"]
    }
    save(output / "retrieval-summary.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dialogues", nargs="+", type=Path)
    parser.add_argument("--retrieval", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.dialogues:
        summarize_dialogues(args.dialogues, args.output)
    if args.retrieval:
        summarize_retrieval(args.retrieval, args.output)


if __name__ == "__main__":
    main()
