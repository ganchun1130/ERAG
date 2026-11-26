"""Build a real teaching KB and replay autonomous Chinese dialogue regression cases."""

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.expanded_retrieval import save
from benchmarks.validate_real_models import run_dialogue
from erag import ERAGEngine, Settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--fixture", type=Path, default=Path("examples/validation"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fixture = json.loads((args.fixture / "cases.json").read_text())
    settings = Settings(
        _env_file=args.env_file,
        knowledge_base_root=args.output / "knowledge_bases",
        contextual_ranking=True,
        enrich_enabled=True,
        final_top_k=5,
        context_max_chars=16000,
        max_tokens=2048,
    )
    root = Path(__file__).resolve().parents[1]
    report = {
        "status": "running",
        "dialogues": {},
        "fixture_sha256": hashlib.sha256((args.fixture / "course.md").read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256((args.fixture / "cases.json").read_bytes()).hexdigest(),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted([*root.glob("erag/*.py"), Path(__file__).resolve()])
        },
        "settings": {k: v for k, v in settings.model_dump(mode="json").items() if "api_key" not in k},
    }
    with ERAGEngine(settings) as engine:
        report["build"] = engine.build_knowledge_base([args.fixture / "course.md"], "validation-course")
    save(args.output / "report.json", report)
    if report["build"]["warnings"]:
        raise RuntimeError("Real build degraded; inspect preserved warnings")
    for mode, budget, tct in [("dual", 15, False), ("raw", 50, False), ("raw", 50, True)]:
        name = f"{mode}{budget}" + ("_focused_tct" if tct else "")
        configured = settings.model_copy(update={"retrieval_mode": mode, "per_query_top_k": budget})
        report["dialogues"][name] = run_dialogue(
            configured, fixture["dialogue"], "validation-course", history_turns=5, tct_enabled=tct
        )
        save(args.output / "report.json", report)
        print(f"completed {name}", flush=True)
    report["unanswerable"] = run_dialogue(
        settings.model_copy(update={"retrieval_mode": "raw", "per_query_top_k": 50}),
        [fixture["unanswerable"]],
        "validation-course",
        history_turns=0,
        tct_enabled=False,
    )
    report["status"] = "completed_needs_answer_review"
    save(args.output / "report.json", report)


if __name__ == "__main__":
    main()
