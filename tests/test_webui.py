import uuid

import pytest

from erag.schemas import Hit


@pytest.fixture
def ui(monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "false")
    from erag.web import ui

    return ui


def test_interface_constructs_on_installed_gradio(ui, monkeypatch, tmp_path):
    monkeypatch.setattr(ui, "list_knowledge_bases", lambda: [])
    demo = ui.create_interface(history_path=tmp_path / "history.sqlite3")
    assert demo.get_config_file()["components"]
    demo.close()


def test_followup_receives_only_completed_raw_history(ui, tmp_path, chunks):
    store = ui.ChatStore(tmp_path / "history.sqlite3")
    owner = uuid.uuid4().hex
    session = store.create(owner)
    turn, _ = store.begin(owner, session, "问题")
    sources = [Hit(chunk=chunks[0], score=1, sub_query="问题", route="global")]
    store.finish(owner, session, turn, "回答[1]", [s.model_dump() for s in sources])
    failed, _ = store.begin(owner, session, "失败问题")
    store.finish(owner, session, failed, "后端错误", [], error=True)
    store.begin(owner, session, "还在生成")
    assert "<details>" in ui.format_response_with_sources("回答[1]", sources)
    assert store.model_history(owner, session) == [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "回答[1]"},
    ]


def test_build_selects_actual_default_name(ui, monkeypatch):
    monkeypatch.setattr(
        ui, "build_knowledge_base", lambda files, name: {"name": "教材", "chunks": 2, "warnings": []}
    )
    monkeypatch.setattr(ui, "list_knowledge_bases", lambda: ["教材"])
    _, update = ui.build(["教材.pdf"], "")
    assert update["value"] == "教材"


def test_source_html_is_escaped(chunks):
    from erag.web.functions import format_response_with_sources

    chunk = chunks[0].model_copy(update={"content": "<script>alert(1)</script>"})
    text = format_response_with_sources("回答", [Hit(chunk=chunk, score=1, sub_query="问题", route="global")])
    assert "<script>" not in text and "&lt;script&gt;" in text and "第1页" in text
