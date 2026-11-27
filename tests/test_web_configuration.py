import json
import stat
import uuid

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from erag.web.configuration import ProfileStore, WebProfile, WorkspaceRuntime, service_error


def custom_profile(**updates):
    values = {
        "chat": {
            "protocol": "openai",
            "url": "https://chat.invalid/v1",
            "model": "my-chat",
            "key": "private-chat-key",
        },
        "embedding": {"protocol": "openai", "url": "https://embed.invalid/v1", "model": "my-embedding"},
        "rerank": {"protocol": "disabled"},
        "method": "jsonl",
    }
    values.update(updates)
    return WebProfile.model_validate(values)


def test_profiles_persist_privately_and_never_inherit_another_owner(tmp_path):
    path = tmp_path / "settings" / "profiles.sqlite3"
    owner, other = uuid.uuid4().hex, uuid.uuid4().hex
    store = ProfileStore(path, WebProfile())
    store.save(owner, custom_profile())
    reloaded = ProfileStore(path, WebProfile())
    version, profile = reloaded.load(owner)
    assert version == 1 and profile.chat.key.get_secret_value() == "private-chat-key"
    assert reloaded.load(other)[0] == 0
    assert not reloaded.load(other)[1].chat.key.get_secret_value()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_blank_keys_preserve_only_same_endpoint_and_explicit_clear_wins(tmp_path):
    store = ProfileStore(tmp_path / "settings.sqlite3", WebProfile())
    owner = uuid.uuid4().hex
    store.save(owner, custom_profile())
    draft = custom_profile().model_dump()
    draft["chat"]["key"] = ""
    draft["chat"]["model"] = "new-model-same-service"
    assert store.prepare(owner, draft).chat.key.get_secret_value() == "private-chat-key"
    draft["chat"]["url"] = "https://another.invalid/v1"
    assert store.prepare(owner, draft).chat.key.get_secret_value() == ""
    draft["chat"]["key"] = "password-manager-value"
    assert store.prepare(owner, draft, ["chat"]).chat.key.get_secret_value() == ""


@pytest.mark.parametrize(
    "updates",
    [
        {"chat": {"protocol": "openai", "url": "file:///etc/passwd", "model": "m"}},
        {"chat": {"protocol": "openai", "url": "https://user:secret@example.com", "model": "m"}},
        {"chat": {"protocol": "openai", "url": "https://example.com", "model": ""}},
        {"chunk_size": 200, "chunk_overlap": 201},
        {"extension_factory": "untrusted:create"},
        {"method": "external", "external_url": "https://example.com/retrieval", "external_id": ""},
    ],
)
def test_invalid_web_config_cannot_be_published(updates):
    with pytest.raises(ValidationError):
        custom_profile(**updates)


def test_custom_service_does_not_forward_server_credentials_or_planner(settings, tmp_path):
    settings.llm_api_key = SecretStr("server-secret")
    settings.llm_headers = {"X-Server": SecretStr("server-header-secret")}
    settings.llm_extra_body = {"vendor_private": True}
    settings.retrieval_model = "server-private-planner"
    settings.retrieval_api_key = SecretStr("server-planner-secret")
    runtime = WorkspaceRuntime(settings, tmp_path)
    owner, other = uuid.uuid4().hex, uuid.uuid4().hex
    runtime.apply(owner, custom_profile())
    with runtime.use(owner) as workspace:
        config = workspace.engine.settings
        assert config.llm_api_key.get_secret_value() == "private-chat-key"
        assert not config.llm_headers and not config.llm_extra_body
        assert not config.retrieval_model and not config.retrieval_api_key.get_secret_value()
        assert workspace.engine.upload_extensions == [".jsonl"]
    with runtime.use(other) as workspace:
        assert workspace.engine.settings.llm_api_key.get_secret_value() == "server-secret"
    assert settings.llm_api_key.get_secret_value() == "server-secret"
    runtime.close()


def test_embedding_namespaces_keep_old_indexes_and_isolate_owners(settings, tmp_path):
    runtime = WorkspaceRuntime(settings, tmp_path)
    owner, other = uuid.uuid4().hex, uuid.uuid4().hex
    profile = custom_profile()
    runtime.apply(owner, profile)
    with runtime.use(owner) as workspace:
        old_root, old_scope = workspace.engine.settings.knowledge_base_root, workspace.scope
    old_root.mkdir(parents=True)
    marker = old_root / "preserved.txt"
    marker.write_text("existing index")
    updated = profile.model_copy(deep=True)
    updated.embedding.model = "new-embedding"
    runtime.apply(owner, updated)
    with runtime.use(owner) as workspace:
        assert workspace.engine.settings.knowledge_base_root != old_root
        assert workspace.scope != old_scope
    runtime.apply(owner, profile)
    with runtime.use(owner) as workspace:
        assert workspace.engine.settings.knowledge_base_root == old_root
        assert marker.read_text() == "existing index"
    runtime.apply(other, profile)
    with runtime.use(other) as workspace:
        assert workspace.engine.settings.knowledge_base_root != old_root
    runtime.close()


def test_configuration_change_waits_for_live_operation_and_pool_is_bounded(settings, tmp_path):
    runtime = WorkspaceRuntime(settings, tmp_path, cache_size=1)
    first, second = uuid.uuid4().hex, uuid.uuid4().hex
    with runtime.use(first, operation=True) as active:
        with pytest.raises(ValueError, match="仍在处理"):
            runtime.apply(first, custom_profile())
        with runtime.use(second):
            assert active.users == 1
        assert len(runtime._engines) == 1
    runtime.apply(first, custom_profile())
    with runtime.use(first) as current:
        assert current.engine is not active.engine
    assert len(runtime._engines) == 1
    runtime.close()


def test_saving_cannot_overtake_an_operation_acquiring_its_configuration(settings, tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    runtime = WorkspaceRuntime(settings, tmp_path)
    owner = uuid.uuid4().hex
    reading, resume, running, release, saving, saved = (Event() for _ in range(6))
    load = runtime.profiles.load

    def pause_first_load(owner):
        result = load(owner)
        if not reading.is_set():
            reading.set()
            assert resume.wait(5)
        return result

    def operation():
        with runtime.use(owner, operation=True):
            running.set()
            assert release.wait(5)

    def save():
        saving.set()
        try:
            runtime.apply(owner, custom_profile())
        finally:
            saved.set()

    monkeypatch.setattr(runtime.profiles, "load", pause_first_load)
    with ThreadPoolExecutor(max_workers=2) as pool:
        request = pool.submit(operation)
        try:
            assert reading.wait(5)
            update = pool.submit(save)
            assert saving.wait(5)
            assert not saved.wait(0.1)
            resume.set()
            assert running.wait(5)
            with pytest.raises(ValueError, match="仍在处理"):
                update.result(timeout=5)
        finally:
            resume.set()
            release.set()
            request.result(timeout=5)
            runtime.close()
    assert not runtime._operations


def test_external_service_uses_workspace_key_without_mutating_environment(settings, tmp_path, monkeypatch):
    monkeypatch.setenv("KEEP_EXISTING_KEY", "server-key")
    profile = custom_profile(
        method="external",
        external_url="https://knowledge.invalid/retrieval",
        external_name="资料库",
        external_id="remote-42",
        external_key="owner-key",
    )
    runtime = WorkspaceRuntime(settings, tmp_path)
    workspace = runtime.create(uuid.uuid4().hex, profile)
    repository = workspace.engine.extensions.knowledge
    assert repository.client.headers["Authorization"] == "Bearer owner-key"
    assert workspace.engine.list_knowledge_bases() == ["资料库"]
    assert not workspace.engine.extensions.can_build
    assert not workspace.engine.extensions.requires_embeddings
    import os

    assert os.environ["KEEP_EXISTING_KEY"] == "server-key"
    workspace.engine.close()
    runtime.close()


def test_provider_error_body_is_not_sent_to_browser_or_history():
    response = httpx.Response(
        401,
        json={"error": "secret-token-from-provider"},
        request=httpx.Request("POST", "https://example.com"),
    )
    error = httpx.HTTPStatusError("secret-token-from-provider", request=response.request, response=response)
    assert "secret-token" not in service_error(error)
    assert "认证" in service_error(error)


def test_ui_save_reload_reset_and_old_conversation_scope(ui, settings, tmp_path):
    runtime = WorkspaceRuntime(settings, tmp_path)
    demo = ui.create_interface(history_path=tmp_path / "history.sqlite3", runtime=runtime)
    functions = {dependency.fn.__name__: dependency.fn for dependency in demo.fns.values() if dependency.fn}
    state = functions["boot"](None)[0]
    owner = state["owner"]
    # Locate the actual editor to exercise its serialization, callbacks and return shapes.
    import gradio as gr

    from erag.web.settings_panel import SettingsPanel

    with gr.Blocks():
        panel = SettingsPanel()
    profile = custom_profile()
    values = [update["value"] for update in panel.values(profile)]
    key_index = list(panel.fields).index("chat.key")
    values[key_index] = "private-chat-key"
    result = functions["save_settings"](state, *values)
    assert "已保存" in result[0]
    assert "private-chat-key" not in json.dumps(result, ensure_ascii=False, default=str)
    state = result[1 + len(panel.fields)]
    updates = functions["load_settings"](state)
    assert updates[key_index]["value"] == ""
    assert "已保存" in updates[key_index]["placeholder"]
    with runtime.use(owner) as workspace:
        scope = workspace.scope
    store = ui.ChatStore(tmp_path / "history.sqlite3")
    store.configure(owner, state["active"], kb="旧资料", rerank=False, tct=False, scope=scope)
    turn, _ = store.begin(owner, state["active"], "旧问题")
    store.finish(owner, state["active"], turn, "旧答案", [])
    restored = functions["reset_settings"](state)
    new_state = restored[1 + len(panel.fields)]
    assert new_state["active"] != state["active"]
    assert store.get(owner, state["active"])["turns"][0]["answer"] == "旧答案"
    assert runtime.profiles.load(owner)[1].chat.protocol == "inherit"
    assert not runtime.profiles.load(owner)[1].chat.key.get_secret_value()
    old_render = functions["select_chat"](state["active"], new_state)
    assert not old_render[10]["interactive"]
    assert "另一配置" in old_render[15]
    runtime.close()
    demo.close()


@pytest.fixture
def ui(monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "false")
    from erag.web import ui

    return ui
