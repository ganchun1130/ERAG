# coding:utf-8
# @File  : ui.py
# @Author: ganchun
# @Date  :  2025/04/22
# @Description: 实现聊天界面UI

import gradio as gr
import time
import os
from css import complete_css
from functions import create_history_item, process_query, format_response_with_sources
import threading

# 全局变量跟踪取消状态
processing_cancelled = False


def create_interface():
    with gr.Blocks(title="智能问答", css=complete_css) as demo:
        # 添加JS脚本，控制知识来源的显示/隐藏和处理弹窗的取消
        gr.HTML("""
        <script>
        function toggleSources(messageId) {
            const sourcesEl = document.getElementById('sources-' + messageId);
            const buttonEl = document.getElementById('toggle-' + messageId);

            if (sourcesEl.style.display === 'none') {
                sourcesEl.style.display = 'block';
                buttonEl.innerHTML = '↑ 隐藏参考来源';
            } else {
                sourcesEl.style.display = 'none';
                buttonEl.innerHTML = '↓ 显示参考来源';
            }
        }

        function cancelProcessing() {
            // 隐藏弹窗
            const modalEl = document.querySelector('.modal-mask');
            if (modalEl) {
                modalEl.classList.remove('show');
            }

            // 触发取消处理的按钮点击
            const cancelBtn = document.getElementById('cancel_processing_btn');
            if (cancelBtn) {
                cancelBtn.click();
            }
        }
        </script>
        """)

        # 状态变量
        current_kb = gr.State("计算教育学")
        kb_options = gr.State(["计算教育学", "基础数学理论", "机器学习基础"])

        # 添加模态弹窗 - 处理中提示框
        modal = gr.HTML(visible=False)

        # 隐藏的取消按钮，用于从JS触发Python函数
        cancel_btn = gr.Button("取消", visible=False, elem_id="cancel_processing_btn")

        with gr.Row():
            # 左侧边栏
            with gr.Column(scale=1, min_width=300):
                # 智能体列表
                gr.Markdown("###  智能体列表")
                with gr.Column(elem_classes="agent-list"):
                    gr.HTML("""
                                        <div class="agent-card"> 动态检索路由 智能体</div>
                                        <div class="agent-card"> 检索 智能体</div>
                                        <div class="agent-card"> 生成 智能体</div>
                                    """)

                # 知识库列表
                gr.Markdown("###  知识库列表")
                knowledge_radio = gr.Radio(
                    choices=["计算教育学", "基础数学理论", "机器学习基础"],
                    label="可用知识库",
                    elem_classes="knowledge-list",
                    interactive=True,
                    show_label=False,
                    value="计算教育学"
                )

                # 文件上传
                gr.Markdown("###  知识库构建")
                with gr.Column(elem_classes="upload-section"):
                    upload_btn = gr.UploadButton(
                        "选择文件上传",
                        file_types=["text", ".pdf", ".txt"],
                        file_count="multiple"
                    )
                    gr.Markdown("支持TXT/PDF格式", elem_classes="format-hint")

                # 新建对话按钮
                new_chat_btn = gr.Button(
                    " 新建对话",
                    variant="primary",
                    elem_classes="new-chat-btn"
                )

            # 主聊天区
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=600)
                with gr.Group():
                    input_box = gr.Textbox(placeholder="输入问题...")
                    hint_area = gr.HTML(
                        f"🧠 教学助手正在基于 <strong>计算教育学<ron您聊天",
                        elem_classes="knowledge-hint"
                    )
                send_btn = gr.Button("发送", variant="primary")

        # 显示处理中状态提示框
        def show_processing_modal(files):
            if not files:
                return gr.HTML(visible=False)

            # 重置取消标记
            global processing_cancelled
            processing_cancelled = False

            # 获取文件名
            file_name = files[0].name if files else "未知文件"
            file_basename = os.path.basename(file_name)

            modal_html = f"""
            <div class="modal-mask show">
                <div class="modal-content" style="width: 400px;">
                    <div class="loading-spinner"></div>
                    <h3>正在处理文件</h3>
                    <p>{file_basename}</p>
                    <p>正在构建知识库，请耐心等待...</p>
                    <p style="color: #6b7280; font-size: 12px;">这个过程可能需要几分钟</p>
                    <button onclick="cancelProcessing()"
                            style="margin-top: 20px; padding: 8px 16px; background: #f3f4f6;
                            border: 1px solid #d1d5db; border-radius: 4px; cursor: pointer;">
                        取消处理
                    <tton>
                </div>
            </div>
            """

            # 启动处理线程
            file_data = files[0]
            threading.Thread(target=process_file, args=(file_data, current_kb.value), daemon=True).start()

            return gr.HTML(modal_html, visible=True)

        # 处理取消按钮点击
        def cancel_processing():
            global processing_cancelled
            processing_cancelled = True
            return gr.HTML(visible=False)

        # 模拟文件处理过程
        def process_file(file_data, kb_name):
            if not file_data:
                return

            # 每隔1秒检查一次是否被取消
            for _ in range(300):  # 处理5分钟
                if processing_cancelled:
                    print("处理已被用户取消")
                    return
                time.sleep(1)

            # 处理完成，更新知识库列表
            # 注意：由于Gradio的限制，这里无法直接更新UI组件
            # 在实际应用中，可以使用数据库或文件存储更新后的选项，然后在UI加载时读取

        # 更新知识库提示
        def update_hint(knowledge):
            if knowledge:
                return gr.HTML(
                    f"🧠 教学助手正在基于 <strong>{knowledge}<ron您聊天",
                    elem_classes="knowledge-hint"
                ), knowledge
            return gr.HTML(visible=False), knowledge

        # 处理用户输入
        def respond(message, chat_history, knowledge_base):
            if not message.strip():
                return "", chat_history

            # 处理用户查询
            response, sources = process_query(message, kb_name='chapter_knowledge_base')

            # 格式化回答，包含知识来源
            formatted_response = format_response_with_sources(response, sources)

            # 更新聊天历史
            chat_history.append([message, formatted_response])

            return "", chat_history

        # 新建对话
        def create_new_chat():
            return []

        # 事件绑定
        # 上传按钮处理
        upload_btn.upload(
            show_processing_modal,
            inputs=[upload_btn],
            outputs=[modal]
        )

        # 取消按钮处理
        cancel_btn.click(
            cancel_processing,
            outputs=[modal]
        )

        # 知识库选择
        knowledge_radio.change(
            update_hint,
            inputs=[knowledge_radio],
            outputs=[hint_area, current_kb]
        )

        # 新建对话按钮
        new_chat_btn.click(
            create_new_chat,
            outputs=[chatbot]
        )

        # 发送按钮
        send_btn.click(
            respond,
            inputs=[input_box, chatbot, current_kb],
            outputs=[input_box, chatbot]
        )

        # 回车发送
        input_box.submit(
            respond,
            inputs=[input_box, chatbot, current_kb],
            outputs=[input_box, chatbot]
        )

    return demo


if __name__ == "__main__":
    interface = create_interface()
    interface.launch()