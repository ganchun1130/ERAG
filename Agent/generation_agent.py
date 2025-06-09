# coding:utf-8
# @File  : generation_agent.py
# @Author: ganchun
# @Date  :  2025/04/23
# @Description: 基于通义千问的生成智能体

import json
from typing import Dict, Any, List, Optional, Union
from xinference.client import RESTfulClient
from .retrieval_agent import RetrievalAgent


class GenerationAgent:
    def __init__(
            self,
            model_name: str = "qwen",
            endpoint: str = "http://localhost:9997",
            system_prompt: str = "你是一个智能助手，请基于提供的信息回答问题。",
            retrieval_agent: Optional[RetrievalAgent] = None
    ):
        """初始化生成智能体

        Args:
            model_name: 使用的模型名称
            endpoint: Xinference 服务端点
            system_prompt: 系统提示词
            retrieval_agent: 检索智能体实例
        """
        self.client = RESTfulClient(endpoint)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.retrieval_agent = retrieval_agent

        # 检查模型是否存在，不存在则部署
        models = self.client.list_models()
        model_exists = any(model["model_name"] == model_name for model in models)

        if not model_exists:
            print(f"模型 {model_name} 不存在，请确保已部署模型")

    def set_retrieval_agent(self, retrieval_agent: RetrievalAgent):
        """设置检索智能体

        Args:
            retrieval_agent: 检索智能体实例
        """
        self.retrieval_agent = retrieval_agent

    def generate(
            self,
            query: str,
            use_retrieval: bool = True,
            temperature: float = 0.7,
            max_tokens: int = 2048
    ) -> str:
        """生成回答

        Args:
            query: 用户问题
            use_retrieval: 是否使用检索增强
            temperature: 生成温度
            max_tokens: 最大生成长度

        Returns:
            生成的回答
        """
        context = ""
        if use_retrieval and self.retrieval_agent:
            try:
                context = self.retrieval_agent.get_context(query)
            except Exception as e:
                print(f"检索出错: {e}")

        if context:
            prompt = f"""请基于以下信息回答用户的问题。如果无法从提供的信息中找到答案，请坦诚回答不知道。

参考信息:
{context}

用户问题: {query}
"""
        else:
            prompt = query

        response = self.client.chat_completion(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response["choices"][0]["message"]["content"]

    def chat(
            self,
            messages: List[Dict[str, str]],
            use_retrieval: bool = True,
            temperature: float = 0.7,
            max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """多轮对话

        Args:
            messages: 对话历史
            use_retrieval: 是否使用检索增强
            temperature: 生成温度
            max_tokens: 最大生成长度

        Returns:
            模型响应
        """
        if use_retrieval and self.retrieval_agent and messages:
            last_user_msg = next((msg["content"] for msg in reversed(messages)
                                  if msg["role"] == "user"), None)

            if last_user_msg:
                try:
                    context = self.retrieval_agent.get_context(last_user_msg)
                    if context:
                        # 在系统消息中添加检索信息
                        system_msg = {
                            "role": "system",
                            "content": f"{self.system_prompt}\n\n参考信息:\n{context}"
                        }
                        messages = [system_msg] + [m for m in messages if m["role"] != "system"]
                except Exception as e:
                    print(f"检索出错: {e}")

        # 确保有系统消息
        has_system = any(msg["role"] == "system" for msg in messages)
        if not has_system:
            messages = [{"role": "system", "content": self.system_prompt}] + messages

        response = self.client.chat_completion(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response