"""Repair explicitly identified fallback chunks from a failed experiment into a NEW snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

from benchmarks.validate_real_models import MeasuredProviders
from erag import ERAGEngine, Settings
from erag.knowledge import KnowledgeBase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--failed-report", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True, help="日志中失败块的零起始索引")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.failed_report.read_text(encoding="utf-8"))
    if original["status"] != "failed" or not original.get("build", {}).get("warnings"):
        parser.error("只修复明确记录了构建降级的失败实验")
    source = Path(original["build"]["path"])
    settings = Settings(_env_file=args.env_file, knowledge_base_root=args.output_dir / "knowledge_bases")
    providers = MeasuredProviders(settings)
    try:
        kb = KnowledgeBase(source.parent, source.name, providers, settings)
        kb.load()
        indices = sorted(set(args.indices))
        for index in indices:
            if not 0 <= index < len(kb.chunks):
                parser.error(f"块索引越界: {index}")
            chunk = kb.chunks[index]
            if chunk.summary != chunk.content[:160] or chunk.tags != (
                [chunk.chapter] if chunk.chapter else []
            ):
                parser.error(f"块 {index} 不是可确认的降级摘要，请核对失败日志")
        fallback_indices = {
            i
            for i, chunk in enumerate(kb.chunks)
            if chunk.summary == chunk.content[:160]
            and chunk.tags == ([chunk.chapter] if chunk.chapter else [])
        }
        if set(indices) != fallback_indices:
            parser.error(f"必须核对并修复全部降级块，当前可识别索引为 {sorted(fallback_indices)}")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        report = {
            "status": "running",
            "source_report": str(args.failed_report.resolve()),
            "source_report_sha256": hashlib.sha256(args.failed_report.read_bytes()).hexdigest(),
            "source_index": kb.manifest,
            "original_warnings": original["build"]["warnings"],
            "repaired_chunks": [],
        }
        report_path = args.output_dir / "repair-report.json"

        def save():
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        save()
        started = perf_counter()
        try:
            with ERAGEngine(settings, providers) as engine:
                chunks = [chunk.model_copy(deep=True) for chunk in kb.chunks]
                for index in indices:
                    chunk = chunks[index]
                    enriched, retries = engine._enrich(chunk.content, chunk.chapter)
                    chunk.summary, chunk.tags = enriched.summary, enriched.tags
                    report["repaired_chunks"].append(
                        {"index": index, "chunk_id": chunk.id, "retries": retries}
                    )
                    save()
                target = KnowledgeBase(settings.knowledge_base_root, source.name, providers, settings)
                target.build(chunks)
                report.update(
                    status="completed",
                    path=str(target.root.resolve()),
                    chunks=len(chunks),
                    elapsed_ms=(perf_counter() - started) * 1000,
                    logical_calls=dict(Counter(event["operation"] for event in providers.events)),
                    all_other_chunks_unchanged=all(
                        chunk == kb.chunks[i] for i, chunk in enumerate(chunks) if i not in indices
                    ),
                )
                save()
                print(json.dumps(report, ensure_ascii=False, indent=2))
        except Exception as exc:
            report.update(status="failed", error=str(exc))
            save()
            raise
    finally:
        providers.close()


if __name__ == "__main__":
    main()
