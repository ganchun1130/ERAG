import json

import pytest

from benchmarks.analyze_expanded import summarize_dialogues


def test_dialogue_summary_checks_pairing_and_bootstraps_conversations(tmp_path):
    directory, output = tmp_path / "run", tmp_path / "summary"
    directory.mkdir()
    output.mkdir()
    sessions, rows = [], []
    for session in ["a", "b"]:
        sessions.append(
            {
                "id": session,
                "split": "test",
                "turns": [{"query": "question", "history": [], "reference_answer": "reference"}],
            }
        )
        for variant, value in [("history", 0.4), ("resolved_history", 0.6), ("focused_tct", 0.6)]:
            rows.append(
                {
                    "session": session,
                    "turn": 1,
                    "variant": variant,
                    "retrieval": {"recall": value},
                    "all_evidence": False,
                    "word_f1": value,
                    "calls": {"chat": 2},
                    "elapsed_s": 1,
                    "warnings": [],
                    "source_ids": [],
                    "answer": "answer",
                }
            )
    (directory / "fixture.json").write_text(json.dumps({"sessions": sessions, "documents": []}))
    (directory / "provenance-test.json").write_text(
        json.dumps(
            {"source_sha256": {}, "variants": {v: {} for v in ["history", "resolved_history", "focused_tct"]}}
        )
    )
    path = directory / "results-test.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    summarize_dialogues([directory], output)
    result = json.loads((output / "dialogue-summary.json").read_text())
    delta = result["paired_vs_history"]["resolved_history"]["recall5"]
    assert delta["difference"] == pytest.approx(0.2)
    assert delta["cluster_ci95"] == pytest.approx([0.2, 0.2])
    path.write_text("\n".join(json.dumps(r) for r in rows + rows[:1]))
    with pytest.raises(ValueError, match="duplicate"):
        summarize_dialogues([directory], output)
    path.write_text("\n".join(json.dumps(r) for r in rows if r["variant"] != "focused_tct"))
    with pytest.raises(ValueError, match="Missing"):
        summarize_dialogues([directory], output)
