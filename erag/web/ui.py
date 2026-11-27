"""Persistent Gradio chat workspace; run with erag-ui."""

from __future__ import annotations

import argparse
import atexit
import uuid
from html import escape
from pathlib import Path

import gradio as gr

from erag.schemas import Hit
from erag.settings import Settings

from .configuration import WorkspaceRuntime, service_error
from .css import complete_css
from .functions import (
    build_knowledge_base,
    format_response_with_sources,
    get_engine,
    list_knowledge_bases,
)
from .history import ChatStore, identifier
from .settings_panel import SettingsPanel


def build(files, name, *, engine=None):
    try:
        if engine is None:
            result = build_knowledge_base(files, name)
            names = list_knowledge_bases()
        else:
            paths = [f if isinstance(f, (str, Path)) else f.name for f in (files or [])]
            result = engine.build_knowledge_base(paths, name.strip() or None)
            names = engine.list_knowledge_bases()
    except Exception as exc:
        raise gr.Error(service_error(exc)) from None
    for warning in result["warnings"]:
        gr.Warning(warning)
    return (
        f"已完成：{result['name']}，写入 {result['chunks']} 个知识块",
        gr.update(choices=names, value=result["name"]),
    )


def create_interface(*, history_path: Path | None = None, runtime: WorkspaceRuntime | None = None):
    settings = get_engine().settings
    store = ChatStore(history_path or settings.knowledge_base_root.parent / "chat_history.sqlite3")
    store.recover_interrupted()
    runtime = runtime or WorkspaceRuntime(settings, store.path.parent)
    atexit.register(runtime.close)
    initial = list_knowledge_bases()
    default_kb = initial[0] if initial else ""
    with gr.Blocks(title="ERAG · 知识小站") as demo:
        browser = gr.BrowserState(None, storage_key="erag.workspace.v1", secret=store.browser_secret())
        with gr.Sidebar(label="工作空间", width=292, elem_id="workspace-sidebar"):
            gr.HTML(
                '<div class="brand"><span class="brand-mark">✳</span><div><b>ERAG</b><small>知识小站</small></div></div>'
            )
            new = gr.Button("＋  新建对话", variant="primary", elem_id="new-chat")
            search = gr.Textbox(
                placeholder="搜索历史对话", show_label=False, container=False, elem_id="history-search"
            )
            sessions = gr.Radio([], label="最近聊过", elem_id="history-list", value=None)
            with gr.Accordion("对话管理", open=False, elem_classes="sidebar-section"):
                name = gr.Textbox(label="对话名称", max_lines=1)
                rename = gr.Button("重命名", size="sm")
                with gr.Row():
                    delete = gr.Button("删除当前对话", size="sm")
                    export = gr.DownloadButton("导出 Markdown", size="sm")
                undo = gr.Button("撤销最近一次删除", size="sm")
            gr.Markdown("### 我的知识空间", elem_id="tools-title")
            kb = gr.Dropdown([("普通聊天", ""), *initial], value=default_kb, label="知识库")
            panel = SettingsPanel()
            with gr.Accordion("检索设置", open=False, elem_classes="sidebar-section"):
                rerank = gr.Checkbox(value=settings.rerank_enabled, label="重排检索结果")
                tct = gr.Checkbox(
                    value=settings.tct_enabled,
                    label="补充检索（TCT）",
                    info="证据不足时尝试补查，回答可能需要更长时间。",
                )
            with gr.Accordion(
                "添加知识",
                open=False,
                elem_classes="sidebar-section",
                visible=get_engine().extensions.can_build,
            ) as upload_panel:
                files = gr.File(
                    file_count="multiple",
                    file_types=get_engine().upload_extensions,
                    label="上传资料",
                    height=130,
                )
                kb_name = gr.Textbox(label="知识库名称", placeholder="例如：课程笔记")
                build_btn = gr.Button("构建知识库", size="sm")
                build_status = gr.Markdown()
                refresh_btn = gr.Button("刷新知识库", size="sm")
            external_note = gr.Markdown(
                "知识库由外部服务管理，请在该服务中更新资料。", visible=not get_engine().extensions.can_build
            )
            gr.HTML(
                '<div class="sidebar-note">每一次好奇，都值得被记住。<br>对话自动保存，可随时回来继续。</div>'
            )
        with gr.Column(elem_id="chat-main"):
            title = gr.HTML('<div class="chat-heading"><b>新对话</b><span>ERAG 知识助手</span></div>')
            chatbot = gr.Chatbot(
                height="calc(100dvh - 235px)",
                min_height=350,
                show_label=False,
                container=False,
                elem_id="conversation",
                layout="bubble",
                sanitize_html=True,
                placeholder='<div class="welcome"><div class="welcome-icon">✳</div><div class="welcome-eyebrow">A LITTLE CURIOSITY, A NEW DISCOVERY</div><h1>今天，想了解点什么？</h1><p>读懂资料，理清思路。<br>陪你把每一个问题，慢慢聊明白。</p></div>',
                buttons=["copy"],
                feedback_options=None,
            )
            status = gr.Markdown("", elem_id="chat-status")
            with gr.Row(elem_id="composer"):
                message = gr.Textbox(
                    placeholder="写下你的问题，或接着刚才的话题…",
                    show_label=False,
                    container=False,
                    lines=2,
                    max_lines=8,
                    scale=9,
                    elem_id="message-input",
                )
                send = gr.Button("↑", variant="primary", min_width=48, scale=0, elem_id="send-message")
            with gr.Row(elem_id="composer-actions"):
                gr.Markdown(
                    "Enter 发送 · Shift + Enter 换行　｜　重要信息请结合来源核对", elem_id="composer-hint"
                )
                retry = gr.Button("重新生成", size="sm", scale=0, min_width=90)
                stop = gr.Button("停止生成", size="sm", scale=0, min_width=90, interactive=False)

        outputs = [
            browser,
            sessions,
            title,
            chatbot,
            kb,
            rerank,
            tct,
            name,
            new,
            delete,
            send,
            retry,
            stop,
            search,
            message,
            status,
            export,
            undo,
            build_btn,
            refresh_btn,
            upload_panel,
            files,
            external_note,
            panel.save,
            panel.reset,
        ]

        def state_value(value):
            if not isinstance(value, dict):
                raise gr.Error("页面尚未就绪，请刷新后重试")
            return identifier(value.get("owner")), identifier(value.get("active"))

        def render(value, *, notice="", clear=False):
            owner, active = state_value(value)
            current = store.get(owner, active)
            busy = any(turn["status"] == "running" for turn in current["turns"])
            displayed = []
            for turn in current["turns"]:
                displayed.append({"role": "user", "content": turn["question"]})
                if turn["status"] == "complete":
                    answer = format_response_with_sources(
                        turn["answer"], [Hit.model_validate(s) for s in turn["sources"]]
                    )
                elif turn["status"] == "running":
                    answer = "正在查找资料并组织回答…"
                elif turn["status"] == "error":
                    answer = "暂时无法完成回答。可以点击“重新生成”再试一次。"
                else:
                    answer = "*已停止生成。*"
                displayed.append({"role": "assistant", "content": answer})
            rows = store.list(owner)
            choices = [(row["title"], row["id"]) for row in rows]
            with runtime.use(owner) as workspace:
                names = workspace.engine.list_knowledge_bases()
                scope = workspace.scope
                can_build = workspace.engine.extensions.can_build
                extensions = workspace.engine.upload_extensions
                rank_disabled = runtime.profiles.load(owner)[1].rerank.protocol == "disabled"
            mismatch = bool(current["kb"] and current["scope"] != scope)
            selected = current["kb"] if not mismatch and current["kb"] in names else ""
            heading = (
                '<div class="chat-heading"><b>'
                + escape(current["title"])
                + "</b><span>"
                + escape(selected or "普通聊天")
                + "</span></div>"
            )

            def control(enabled=True):
                return gr.update(interactive=enabled and not busy)

            return (
                value,
                gr.update(choices=choices, value=active, interactive=not busy),
                heading,
                displayed,
                gr.update(choices=[("普通聊天", ""), *names], value=selected, interactive=not busy),
                gr.update(
                    value=bool(current["rerank"]) and not rank_disabled,
                    interactive=not busy and not rank_disabled,
                ),
                gr.update(value=bool(current["tct"]), interactive=not busy),
                gr.update(value=current["title"], interactive=not busy),
                control(),
                control(),
                control(not mismatch),
                control(bool(current["turns"]) and not mismatch),
                gr.update(interactive=busy),
                control(),
                gr.update(**({"value": ""} if clear else {}), interactive=not busy),
                notice
                or (
                    "此对话的知识库属于另一配置。恢复原设置后可继续，或新建对话。"
                    if mismatch
                    else "正在生成 · 可以随时停止"
                    if busy
                    else ""
                ),
                gr.update(value=store.export(owner, active) if not busy else None, interactive=not busy),
                control(),
                control(),
                control(),
                gr.update(visible=can_build),
                gr.update(file_types=extensions),
                gr.update(visible=not can_build),
                control(),
                control(),
            )

        def boot(value):
            try:
                owner = identifier((value or {}).get("owner"))
            except ValueError:
                owner = uuid.uuid4().hex
            rows = store.list(owner)
            active = (value or {}).get("active")
            if active not in {row["id"] for row in rows}:
                active = rows[0]["id"] if rows else create_chat(owner)
            return render({"owner": owner, "active": active})

        def create_chat(owner, selected=None, rank=None, feedback=None):
            with runtime.use(owner) as workspace:
                engine = workspace.engine
                names = engine.list_knowledge_bases()
                selected = selected if selected in names else (names[0] if selected is None and names else "")
                return store.create(
                    owner,
                    kb=selected or None,
                    scope=workspace.scope,
                    rerank=engine.settings.rerank_enabled if rank is None else rank,
                    tct=engine.settings.tct_enabled if feedback is None else feedback,
                )

        def new_chat(value, selected, rank, feedback):
            owner, _ = state_value(value)
            active = create_chat(owner, selected, rank, feedback)
            return render({"owner": owner, "active": active}, clear=True)

        def select_chat(active, value):
            owner, _ = state_value(value)
            return render({"owner": owner, "active": active}, clear=True)

        def delete_chat(value):
            owner, active = state_value(value)
            store.delete(owner, active)
            return boot({"owner": owner})

        def rename_chat(value, text):
            owner, active = state_value(value)
            try:
                store.rename(owner, active, text)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            return render(value)

        def undo_delete(value):
            owner, _ = state_value(value)
            restored = store.undo_delete(owner)
            if restored:
                return render({"owner": owner, "active": restored}, notice="对话已恢复")
            return render(value, notice="没有可以恢复的对话")

        def configure(value, selected, rank, feedback):
            owner, active = state_value(value)
            current = store.get(owner, active)
            with runtime.use(owner) as workspace:
                scope = workspace.scope
            if ((current["kb"] or "") != selected or current["scope"] != scope) and current["turns"]:
                return new_chat(value, selected, rank, feedback)
            store.configure(owner, active, kb=selected or None, rerank=rank, tct=feedback, scope=scope)
            return render(value)

        def generate(text, value, selected, rank, feedback, *, regenerate=False):
            owner, active = state_value(value)
            with runtime.use(owner, rerank=rank, tct=feedback, operation=True) as workspace:
                current = store.get(owner, active)
                if current["kb"] and current["scope"] != workspace.scope:
                    raise gr.Error("请先恢复此对话的知识库配置，或新建对话")
                store.configure(
                    owner, active, kb=selected or None, rerank=rank, tct=feedback, scope=workspace.scope
                )
                try:
                    turn_id, text = store.begin(owner, active, text or "", retry=regenerate)
                except ValueError as exc:
                    raise gr.Error(str(exc)) from exc
                yield render(value, clear=True)
                history = store.model_history(owner, active)
                try:
                    result = workspace.engine.ask(text, selected or None, history=history)
                    accepted = store.finish(
                        owner, active, turn_id, result.answer, [h.model_dump() for h in result.sources]
                    )
                    notice = " · ".join(result.warnings)
                except Exception as exc:
                    notice = service_error(exc)
                    accepted = store.finish(owner, active, turn_id, notice, [], error=True)
            if accepted:
                yield render(value, notice=notice)

        def regenerate(value, selected, rank, feedback):
            yield from generate("", value, selected, rank, feedback, regenerate=True)

        def stop_chat(value):
            owner, active = state_value(value)
            store.stop(owner, active)
            return render(value, notice="已停止。本轮结果不会写入对话。")

        def filter_sessions(value, text):
            owner, active = state_value(value)
            rows = store.list(owner, text)
            return gr.update(
                choices=[(row["title"], row["id"]) for row in rows],
                value=active if active in {row["id"] for row in rows} else None,
            )

        def build_and_select(files, name, value, rank, feedback):
            owner, _ = state_value(value)
            with runtime.use(owner, operation=True) as workspace:
                working = list(render(value, notice="正在构建知识库…"))
                for index in [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 23, 24]:
                    working[index] = gr.update(interactive=False)
                yield ("正在读取资料并构建索引…", *working)
                try:
                    notice, selection = build(files, name, engine=workspace.engine)
                except Exception:
                    yield ("构建失败，请检查资料格式与嵌入模型连接。", *render(value))
                    return
                yield (notice, *configure(value, selection["value"], rank, feedback))

        def load_settings(value):
            owner, _ = state_value(value)
            return panel.values(runtime.profiles.load(owner)[1])

        def save_settings(value, *values):
            owner, active = state_value(value)
            profile = panel.draft(runtime, owner, values)
            try:
                runtime.apply(owner, profile)
            except ValueError:
                raise gr.Error("设置未保存，请确认当前请求已完成，且项目扩展与新设置兼容。") from None
            current = store.get(owner, active)
            with runtime.use(owner) as workspace:
                changed = current["scope"] != workspace.scope
            if changed:
                active = create_chat(owner)
            result = render(
                {"owner": owner, "active": active},
                notice="设置已保存。"
                + ("已开启新对话，原对话和知识库仍保留。" if changed else "后续请求将使用新设置。"),
            )
            return ("已保存到当前工作空间。", *panel.values(profile), *result)

        def reset_settings(value):
            owner, _ = state_value(value)
            defaults = runtime.profiles.defaults
            # Reuse the publication path, explicitly clearing all previously stored credentials.
            values = [item["value"] for item in panel.values(defaults)]
            for index, field in enumerate(panel.fields):
                if field.endswith(".clear") or field == "external_clear":
                    values[index] = True
            return save_settings(value, *values)

        def test_settings(role, value, *values):
            owner, _ = state_value(value)
            profile = panel.draft(runtime, owner, values)
            try:
                return runtime.test(owner, profile, role)
            except Exception as exc:
                return service_error(exc) + " 测试不会保存设置。"

        demo.load(boot, [browser], outputs).then(
            load_settings, [browser], panel.inputs, api_visibility="private"
        )
        settings_outputs = [panel.status, *panel.inputs, *outputs]
        panel.save.click(save_settings, [browser, *panel.inputs], settings_outputs, api_visibility="private")
        panel.reset.click(reset_settings, [browser], settings_outputs, api_visibility="private")
        for role, button in [
            ("chat", panel.test_chat),
            ("knowledge", panel.test_knowledge),
            ("rerank", panel.test_rerank),
        ]:

            def check(value, *values, role=role):
                return test_settings(role, value, *values)

            button.click(check, [browser, *panel.inputs], [panel.status], api_visibility="private")
        new.click(new_chat, [browser, kb, rerank, tct], outputs)
        sessions.input(select_chat, [sessions, browser], outputs)
        delete.click(delete_chat, [browser], outputs)
        undo.click(undo_delete, [browser], outputs)
        rename.click(rename_chat, [browser, name], outputs)
        search.input(filter_sessions, [browser, search], [sessions])
        for component in [kb, rerank, tct]:
            component.input(configure, [browser, kb, rerank, tct], outputs)
        generation_inputs = [message, browser, kb, rerank, tct]
        jobs = [
            send.click(
                generate,
                generation_inputs,
                outputs,
                concurrency_id="chat",
                concurrency_limit=8,
                show_progress="hidden",
            ),
            message.submit(
                generate,
                generation_inputs,
                outputs,
                concurrency_id="chat",
                concurrency_limit=8,
                show_progress="hidden",
            ),
            retry.click(
                regenerate,
                [browser, kb, rerank, tct],
                outputs,
                concurrency_id="chat",
                concurrency_limit=8,
                show_progress="hidden",
            ),
        ]
        stop.click(stop_chat, [browser], outputs, cancels=jobs, queue=False, show_progress="hidden")
        chatbot.clear(new_chat, [browser, kb, rerank, tct], outputs)
        build_btn.click(build_and_select, [files, kb_name, browser, rerank, tct], [build_status, *outputs])
        refresh_btn.click(lambda value: render(value), inputs=[browser], outputs=outputs)
    demo.erag_runtime = runtime
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="ERAG 知识小站")
    parser.add_argument("--config", type=Path, help="与 CLI 共用的 JSON 配置文件")
    parser.add_argument("--env-file", help="可选 .env 路径")
    args = parser.parse_args()
    options = {"_env_file": args.env_file} if args.env_file else {}
    if args.config:
        options["config_file"] = args.config
    settings = get_engine(Settings(**options)).settings
    create_interface().launch(
        server_name=settings.ui_host,
        server_port=settings.ui_port,
        css=complete_css,
        # A custom data root can put conversation exports outside the working directory.
        allowed_paths=[str((settings.knowledge_base_root.parent / "exports").resolve())],
    )


if __name__ == "__main__":
    main()
