"""Gradio 5/6 interface for the one-model ERAG engine.

Run from the repository root with ``python -m WebUI.ui``.
"""
from __future__ import annotations

import gradio as gr

from .functions import build_knowledge_base, format_response_with_sources, get_engine, list_knowledge_bases, process_query
from .css import complete_css


def create_interface():
    initial = list_knowledge_bases()
    with gr.Blocks(title="ERAG 教学助手", css=complete_css) as demo:
        gr.Markdown("# ERAG 教学助手\n基于论文提出的情境改写、双向意图映射与 TCT 反思流程。")
        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                kb = gr.Dropdown(initial, value=initial[0] if initial else None, label="知识库", allow_custom_value=False)
                files = gr.File(file_count="multiple", file_types=[".pdf", ".txt", ".md", ".csv"], label="构建/更新知识库")
                kb_name = gr.Textbox(label="知识库名称（可选）")
                build_btn = gr.Button("开始构建", variant="secondary")
                build_status = gr.Markdown()
                refresh_btn = gr.Button("刷新知识库列表")
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(type="messages", height=620, label="对话")
                message = gr.Textbox(placeholder="输入教学问题，支持复杂、多轮查询…", label="问题")
                with gr.Row():
                    send = gr.Button("发送", variant="primary")
                    clear = gr.ClearButton([message, chatbot], value="新建对话")

        def respond(text, history, selected):
            if not text or not text.strip(): return "", history
            if not selected:
                return "", history + [{"role": "assistant", "content": "请先选择或构建一个知识库。"}]
            answer, sources = process_query(text, selected)
            history = history + [{"role": "user", "content": text}, {"role": "assistant", "content": format_response_with_sources(answer, sources)}]
            return "", history

        def build(files_, name):
            status = build_knowledge_base(files_, name or None)
            return status, gr.update(choices=list_knowledge_bases(), value=(name or None))

        def refresh():
            names = list_knowledge_bases()
            return gr.update(choices=names, value=names[0] if names else None)

        send.click(respond, [message, chatbot, kb], [message, chatbot])
        message.submit(respond, [message, chatbot, kb], [message, chatbot])
        build_btn.click(build, [files, kb_name], [build_status, kb])
        refresh_btn.click(refresh, outputs=[kb])
    return demo


if __name__ == "__main__":
    settings = get_engine().settings
    create_interface().launch(server_name=settings.ui_host, server_port=settings.ui_port)
