# coding:utf-8
# @File  : chat_with_kb_db.py
# @Author: ganchun
# @Date  :  2025/04/24
# @Description: 基于豆包API的知识库对话系统，支持命令行和Web界面

import os
import json
import time
import argparse
from typing import List, Dict, Any, Tuple, Optional
import jsonlines
import gradio as gr
from openai import OpenAI
from datetime import datetime

# 导入自定义模块
from Prompt.prompt_templates import query_rewrite_prompt
from RAG.retrieval.contextual_rewrite import retrieve_from_knowledge_base, list_knowledge_bases

# 豆包API配置参数
DOUBAO_API_URL = "https://ark.cn-beijing.volces.com/api/v3"
DOUBAO_API_KEY = "ce6fac43-3d59-4dfe-8949-ea1029f42a32"  # 请替换为你的API密钥
DOUBAO_MODEL_ID = "ep-20250306115053-vzsng"  # Doubao-1.5-pro-32k

# 对话历史配置
CHAT_HISTORY_DIR = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\Conversation"
MAX_HISTORY_TURNS = 5  # 最大对话历史轮次

# 确保历史记录目录存在
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)


class ChatSession:
    """聊天会话管理类"""

    def __init__(self, kb_name: str, session_id: Optional[str] = None, keep_history: bool = False):
        """
        初始化聊天会话

        参数:
            kb_name: 知识库名称
            session_id: 会话ID，如果为None则自动生成
        """
        self.kb_name = kb_name
        self.session_id = session_id if session_id else f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.history_file = os.path.join(CHAT_HISTORY_DIR, f"{self.session_id}.jsonl")
        self.keep_history = keep_history
        self.history = self._load_history() if keep_history else []
        self.client = OpenAI(api_key=DOUBAO_API_KEY, base_url=DOUBAO_API_URL)

    def _load_history(self) -> List[Dict[str, Any]]:
        """加载历史对话记录"""
        if not os.path.exists(self.history_file):
            return []

        history = []
        try:
            with jsonlines.open(self.history_file, 'r') as reader:
                for item in reader:
                    history.append(item)
            return history
        except Exception as e:
            print(f"加载历史记录出错: {e}")
            return []

    def save_message(self, role: str, content: str, references: Optional[List[Dict]] = None):
        """
        保存消息到历史记录

        参数:
            role: 角色('user'或'assistant')
            content: 消息内容
            references: 引用的知识片段
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

        if role == "assistant" and references:
            message["references"] = references

        # 无论是否保存到文件，都临时添加到历史中供当前对话使用
        self.history.append(message)

        # 只有在keep_history为True时才保存到文件
        if self.keep_history:
            try:
                with jsonlines.open(self.history_file, 'a') as writer:
                    writer.write(message)
            except Exception as e:
                print(f"保存消息出错: {e}")

    def get_chat_messages(self) -> List[Dict[str, str]]:
        """获取用于LLM上下文的历史消息"""
        # 如果不保留历史，只返回当前对话轮次的消息（最多一问一答）
        if not self.keep_history:
            # 只返回当前这一轮的用户问题，不包含历史
            messages = []
            for msg in self.history:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            return messages

        # 保留历史时的逻辑，只保留最近的几轮对话
        recent_history = self.history[-MAX_HISTORY_TURNS * 2:] if len(
            self.history) > MAX_HISTORY_TURNS * 2 else self.history

        # 转换为OpenAI消息格式
        messages = []
        for msg in recent_history:
            if msg["role"] in ["user", "assistant"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        return messages

    def chat(self, query: str) -> Tuple[str, List[Dict]]:
        """
        处理用户输入并生成回复

        参数:
            query: 用户输入

        返回:
            (回复内容, 引用的知识片段)
        """
        # 保存用户消息
        self.save_message("user", query)

        # 从知识库检索相关信息
        references = retrieve_from_knowledge_base(self.kb_name, query, query_rewrite_prompt)

        # 构建提示上下文
        context = ""
        if references:
            for i, ref in enumerate(references, 1):
                context += f"[{i}] {ref.get('总结', '')}:\n{ref.get('内容', '')[:800]}...\n\n"

        # 准备消息
        messages = self.get_chat_messages()

        # 插入系统提示
        system_prompt = (
            "你是一个专业、友好的智能助手。请根据提供的参考资料回答用户问题。"
            "如果参考资料中没有相关信息，请基于你的知识谨慎回答，并明确指出这是你的判断而非来自参考资料。"
            "回答应当准确、清晰，并尽量使用参考资料中的原文表述。"
            "不要在回答中直接提及参考资料编号，要自然地融入信息。"
        )

        # 如果有上下文，则添加到系统提示中
        if context:
            system_prompt += f"\n\n以下是相关的参考资料：\n{context}"

        # 构建豆包API请求的消息列表
        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(messages)

        try:
            # 调用豆包API生成回复
            print("正在调用豆包API...")
            start_time = time.time()

            response = self.client.chat.completions.create(
                model=DOUBAO_MODEL_ID,
                messages=api_messages,
                temperature=0.7,
                max_tokens=2048
            )

            reply = response.choices[0].message.content.strip()
            print(f"API响应时间: {time.time() - start_time:.2f}秒")

            # 保存助手回复
            self.save_message("assistant", reply, references)

            return reply, references
        except Exception as e:
            error_msg = f"豆包API调用出错: {e}"
            print(error_msg)
            self.save_message("system", error_msg)
            return error_msg, []


def chat_command_line(kb_name: str, keep_history: bool = True):
    """命令行聊天界面"""
    print(f"正在使用知识库: {kb_name}")
    print(f"当前使用的豆包模型: {DOUBAO_MODEL_ID}")
    print(f"历史对话记录: {'启用' if keep_history else '禁用'}")
    print("输入'q'或'quit'退出对话\n")

    session = ChatSession(kb_name, keep_history=keep_history)


    while True:
        query = input("\n用户: ")
        if query.lower() in ['q', 'quit', 'exit']:
            print("对话已结束")
            break

        print("\n思考中...")
        reply, references = session.chat(query)

        print(f"\n助手: {reply}\n")

        if references:
            print("\n参考来源:")
            for i, ref in enumerate(references, 1):
                tags = ", ".join(ref.get("标签", []))
                print(f"[{i}] {tags} - {ref.get('总结', '')[:100]}...")
            print()


def chat_web_interface(kb_name: str, keep_history: bool = True):
    """Gradio Web聊天界面"""
    session = ChatSession(kb_name, keep_history=keep_history)
    references_store = []

    def predict(message, history):
        reply, refs = session.chat(message)
        references_store.clear()
        references_store.extend(refs)
        return reply

    def show_references():
        if not references_store:
            return "没有找到相关参考资料"

        refs_text = "### 参考来源\n\n"
        for i, ref in enumerate(references_store, 1):
            tags = ", ".join(ref.get("标签", []))
            summary = ref.get("总结", "")
            content = ref.get("内容", "")[:500] + "..."

            refs_text += f"**来源 {i}**\n\n"
            refs_text += f"**标签**: {tags}\n\n"
            refs_text += f"**摘要**: {summary}\n\n"
            refs_text += f"**内容**: {content}\n\n"
            refs_text += "---\n\n"

        return refs_text

    with gr.Blocks(title=f"豆包知识库对话 - {kb_name}") as demo:
        history_status = "保留" if keep_history else "不保留"
        gr.Markdown(f"# 基于豆包API的知识库对话系统\n\n当前知识库: **{kb_name}**\n当前模型: **{DOUBAO_MODEL_ID}**")

        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(height=500)
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="请输入您的问题...",
                    container=False
                )

            with gr.Column(scale=3):
                with gr.Accordion("参考来源", open=False) as references_accordion:
                    references_output = gr.Markdown()
                    ref_btn = gr.Button("显示/更新参考来源")
                    ref_btn.click(show_references, outputs=[references_output])

        msg.submit(predict, [msg, chatbot], [chatbot])

        gr.Markdown("""
        ### 使用说明
        1. 在输入框中输入您的问题
        2. 系统会从知识库中检索相关信息并通过豆包API生成回答
        3. 点击"显示/更新参考来源"查看AI回答所依据的资料
        """)

    demo.launch(share=False, server_name="0.0.0.0")


def main():
    parser = argparse.ArgumentParser(description="基于豆包API的知识库对话系统")
    parser.add_argument("--kb", type=str, help="知识库名称", required=True)
    parser.add_argument("--mode", type=str, choices=["cli", "web"], default="cli",
                        help="交互模式: cli表示命令行, web表示网页界面")
    parser.add_argument("--model", type=str, default=DOUBAO_MODEL_ID,
                        help="豆包模型ID，默认为Doubao-1.5-pro-32k")
    parser.add_argument("--no-history", action="store_true",
                        help="不保留历史对话，每次对话都是单轮的")

    args = parser.parse_args()

    # 是否保留历史对话
    keep_history = not args.no_history

    # 如果指定了模型ID，则更新全局变量
    global DOUBAO_MODEL_ID
    if args.model:
        DOUBAO_MODEL_ID = args.model

    # 获取所有可用知识库
    kb_list = list_knowledge_bases()

    if not kb_list:
        print("错误: 没有找到可用的知识库")
        return

    if args.kb not in kb_list:
        print(f"错误: 知识库'{args.kb}'不存在")
        print(f"可用知识库: {kb_list}")
        return

    # 根据模式选择界面，传递keep_history参数
    if args.mode == "cli":
        chat_command_line(args.kb, keep_history)
    else:
        chat_web_interface(args.kb, keep_history)


if __name__ == "__main__":
    main()