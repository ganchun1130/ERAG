import json
import sys
from types import ModuleType

import pytest

from erag import Extensions
from erag.services.check_models import main


@pytest.mark.parametrize("chat_only", [False, True])
def test_health_check_uses_factory_and_skips_unneeded_models(tmp_path, monkeypatch, capsys, chat_only):
    module = ModuleType("health_check_extension")
    events = []

    class Chat:
        def chat(self, messages, *, structured=False, retrieval=False):
            assert structured
            events.append("retrieval" if retrieval else "chat")
            return '{"sum":42}'

        def close(self):
            events.append("closed")

    module.create = lambda settings: Extensions(chat=Chat(), requires_embeddings=chat_only)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = tmp_path / "health.json"
    config.write_text(
        json.dumps(
            {
                "extension_factory": module.__name__ + ":create",
                "retrieval_model": "planner",
                "rerank_enabled": chat_only,
                "embedding_base_url": "http://127.0.0.1:1",
                "rerank_base_url": "http://127.0.0.1:1",
            }
        )
    )
    argv = ["erag-check", "--config", str(config), "--env-file", str(tmp_path / "missing.env")]
    monkeypatch.setattr(sys, "argv", argv + (["--chat-only"] if chat_only else []))
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "passed"
    assert result["embedding"]["status"] == result["reranker"]["status"] == "skipped"
    assert events == ["chat", "retrieval", "closed"]
