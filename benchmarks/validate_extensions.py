"""Opt-in real-model extension smoke test using synthetic, explicitly labelled facts.

Starts no model server and downloads no data. Requires configured chat, embedding,
and (if enabled) reranking APIs. Temporary knowledge bases are deleted on exit.
The HTTP bridge uses a local BM25 service, not a mock model or a deployed Dify instance.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from erag import Chunk, ERAGEngine, Settings
from erag.engine import citation_issue
from erag.lexical import BM25
from erag.schemas import Hit, SubQuery

FACTS = [
    (
        "scholarship",
        "青禾课程奖学金",
        "青禾课程奖学金每人每学期上限为720元。凭证须在每月3日提交给教务服务台。",
    ),
    ("camera", "星岚实验室相机", "星岚实验室相机的借用期是7天，最多续借一次。续借须在到期前一天申请。"),
    (
        "maintenance",
        "春雨平台维护",
        "春雨平台每周三20:30至21:15维护。维护期间只读查询仍可用，写入请求须等维护结束。",
    ),
    ("application", "星河研学报名", "星河研学的报名截止日期为2031年5月18日。申请材料由线下服务台签收。"),
]
QUESTIONS = [
    ("青禾课程奖学金每人每学期的上限是多少？", [r"720|七百二十"], "scholarship"),
    ("它的凭证什么时候交，交给谁？", [r"3日|3号|三日|三号", r"教务服务台"], "scholarship"),
    ("星岚实验室的相机可以借多久，最多续借几次？", [r"7天|7 天|七天", r"一次|1次|1 次"], "camera"),
]


def run_dialogue(engine, kb):
    history, rows = [], []
    for question, patterns, expected in QUESTIONS:
        answer = engine.ask(question, kb, history=history)
        row = answer.model_dump()
        row["question"] = question
        row["facts_present"] = all(re.search(p, answer.answer) for p in patterns)
        row["valid_citations"] = citation_issue(answer.answer, len(answer.sources)) is None
        expected_title = next(title for identifier, title, _ in FACTS if identifier == expected)
        row["expected_evidence"] = any(expected_title in h.chunk.content for h in answer.sources)
        row["no_service_fallback"] = not any("失败" in w or "不可用" in w for w in answer.warnings)
        rows.append(row)
        history.extend(
            [{"role": "user", "content": question}, {"role": "assistant", "content": answer.answer}]
        )
    return rows


def validate(settings):
    report = {"data": "Synthetic extension test fixtures; not a retrieval-quality benchmark", "routes": {}}
    with tempfile.TemporaryDirectory(prefix="erag-extension-check-") as directory:
        root = Path(directory)
        file = root / "synthetic-notes.md"
        file.write_text("\n\n".join(f"# {title}\n{text}" for _, title, text in FACTS), encoding="utf-8")
        base = settings.model_copy(
            update={
                "knowledge_base_root": root / "default",
                "extension_factory": "",
                "extension_options": {},
                "enrich_enabled": False,
                "tct_enabled": False,
                "summary_min_score": 0.0,
                "final_top_k": 3,
                "per_query_top_k": 8,
                "max_tokens": 1024,
            }
        )
        with ERAGEngine(base) as engine:
            if base.rerank_enabled:
                query = "中国的首都是哪里？"
                candidates = [
                    Hit(
                        chunk=Chunk(id=str(i), content=text, source="synthetic-health-check"),
                        score=0,
                        sub_query=query,
                        route="health-check",
                    )
                    for i, text in enumerate(["北京是中国的首都。", "鲸鱼生活在海洋中。"])
                ]
                ranked = engine.providers.rerank(query, candidates)
                if ranked[0].chunk.id != "0":
                    raise ValueError("重排语义检查失败，请先核对模型的输入模板并运行 erag-check")
                report["rerank_semantics"] = [{"id": h.chunk.id, "score": h.score} for h in ranked]
            engine.build_knowledge_base([file], "notes")
            report["routes"]["default_documents"] = run_dialogue(engine, "notes")

        chunks = [
            Chunk(
                id=identifier,
                content=text,
                summary=text,
                source="custom-manual.pdf",
                chapter=title,
                page=i + 1,
                metadata={"pipeline": "user-chunks-v1", "section": identifier},
            )
            for i, (identifier, title, text) in enumerate(FACTS)
        ]
        prepared = root / "prepared.jsonl"
        prepared.write_text("\n".join(c.model_dump_json() for c in chunks), encoding="utf-8")
        custom = base.model_copy(
            update={
                "knowledge_base_root": root / "prepared",
                "extension_factory": "erag.integrations.chunks:create_extensions",
                "embedding_space": "synthetic-test-space-v1",
            }
        )
        with ERAGEngine(custom) as engine:
            engine.build_knowledge_base([prepared], "notes")
            report["routes"]["custom_jsonl"] = run_dialogue(engine, "notes")
            assert engine.open_knowledge_base("notes").chunks == chunks
            report["metadata_and_ids_preserved"] = True
        if "127.0.0.1" in custom.embedding_base_url:
            migrated = custom.model_copy(
                update={"embedding_base_url": custom.embedding_base_url.replace("127.0.0.1", "localhost")}
            )
            with ERAGEngine(migrated) as engine:
                assert engine.open_knowledge_base("notes").search_many(
                    [SubQuery(query=QUESTIONS[0][0])], top_k=2
                )
            report["same_space_endpoint_migration"] = True

        lexical, requests = BM25([c.content for c in chunks]), []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(body)
                scores = lexical.scores(body["query"])
                order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
                rows = [
                    {
                        "content": chunks[i].content,
                        "title": chunks[i].chapter,
                        "score": float(scores[i] / (1 + scores[i])),
                        "metadata": {**chunks[i].metadata, "source": "external-manual.pdf"},
                    }
                    for i in order[: body["retrieval_setting"]["top_k"]]
                    if scores[i] > 0
                ]
                data = json.dumps({"records": rows}, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            external = base.model_copy(
                update={
                    "extension_factory": "erag.integrations.external_knowledge:create_extensions",
                    "extension_options": {
                        "endpoint": f"http://127.0.0.1:{server.server_port}/retrieval",
                        "knowledge_bases": {"external": "synthetic-v1"},
                    },
                    "embedding_base_url": "http://127.0.0.1:1/v1",
                    "rerank_enabled": False,
                    "knowledge_base_root": root / "must-not-exist",
                }
            )
            with ERAGEngine(external) as engine:
                report["routes"]["external_bm25_http"] = run_dialogue(engine, "external")
                assert not external.knowledge_base_root.exists()
            report["external_uses_no_local_index_or_embedding"] = True
            report["external_requests"] = len(requests)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
    report["passed"] = all(
        row[key]
        for rows in report["routes"].values()
        for row in rows
        for key in ["facts_present", "valid_citations", "expected_evidence", "no_service_fallback"]
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--env-file")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    options = {"config_file": args.config} if args.config else {}
    if args.env_file:
        options["_env_file"] = args.env_file
    report = validate(Settings(**options))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"passed": report["passed"], "routes": list(report["routes"]), "turns": 9}, ensure_ascii=False
        )
    )
    if not report["passed"]:
        raise SystemExit("Extension smoke checks failed; inspect the saved report")


if __name__ == "__main__":
    main()
