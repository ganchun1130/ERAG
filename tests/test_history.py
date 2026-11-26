import uuid
from pathlib import Path

import pytest

from erag.web.history import ChatStore


@pytest.fixture
def store(tmp_path):
    return ChatStore(tmp_path / "chat.sqlite3")


def test_conversations_survive_restart_and_are_isolated(store):
    first, second = uuid.uuid4().hex, uuid.uuid4().hex
    session = store.create(first, kb="教材")
    turn, _ = store.begin(first, session, "解释形成性评价")
    assert store.finish(first, session, turn, "反馈[1]", [])
    restored = ChatStore(store.path)
    assert restored.get(first, session)["turns"][0]["answer"] == "反馈[1]"
    assert restored.list(second) == []
    with pytest.raises(ValueError, match="不存在"):
        restored.get(second, session)
    with pytest.raises(ValueError, match="不存在"):
        restored.delete(second, session)
    assert restored.browser_secret() == store.browser_secret()


def test_stop_rejects_late_result_and_allows_next_question(store):
    owner = uuid.uuid4().hex
    session = store.create(owner)
    turn, _ = store.begin(owner, session, "第一问")
    with pytest.raises(ValueError, match="正在生成"):
        store.begin(owner, session, "并发消息")
    store.stop(owner, session)
    assert not store.finish(owner, session, turn, "迟到的回答", [])
    second, _ = store.begin(owner, session, "第二问")
    assert store.finish(owner, session, second, "有效回答", [])
    assert [t["status"] for t in store.get(owner, session)["turns"]] == ["stopped", "complete"]


def test_delete_during_generation_never_resurrects_conversation(store):
    owner = uuid.uuid4().hex
    session = store.create(owner)
    turn, _ = store.begin(owner, session, "问题")
    store.delete(owner, session)
    assert not store.finish(owner, session, turn, "回答", [])
    assert store.list(owner) == []
    assert store.undo_delete(owner) == session
    assert store.get(owner, session)["turns"][0]["status"] == "stopped"
    assert store.undo_delete(uuid.uuid4().hex) is None


def test_retry_export_rename_and_crash_recovery(store):
    owner = uuid.uuid4().hex
    session = store.create(owner)
    turn, _ = store.begin(owner, session, "原问题")
    store.finish(owner, session, turn, "原回答", [])
    retried, question = store.begin(owner, session, "", retry=True)
    assert question == "原问题" and retried != turn
    assert len(store.get(owner, session)["turns"]) == 1
    store.recover_interrupted()
    assert store.get(owner, session)["turns"][0]["status"] == "stopped"
    store.rename(owner, session, "考试复习")
    assert store.list(owner, "考试")[0]["title"] == "考试复习"
    assert "原问题" in Path(store.export(owner, session)).read_text()
