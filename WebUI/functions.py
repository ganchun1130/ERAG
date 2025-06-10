# coding:utf-8
# @File  : functions.py
# @Author: ganchun
# @Date  :  2025/04/22
# @Description: 实现知识库检索和API调用等功能

import os
import time
import faiss
import pickle
import numpy as np
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Tuple
from openai import OpenAI
import re

# 配置
EMBEDDING_API_URL = "http://localhost:9997/v1"
EMBEDDING_MODEL_UID = "bge-m3"
KNOWLEDGE_BASE_ROOT = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\Knowledge_Base\jisuanjiaoyuxue"
VECTOR_DIMENSION = 1024
API_KEY = ""
LLM_MODEL = "qwen3-32b"


def get_embeddings(texts: List[str]) -> np.ndarray:
    """使用本地API获取文本的嵌入向量"""
    client = OpenAI(api_key="not empty", base_url=EMBEDDING_API_URL)

    batch_size = 32
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="生成嵌入向量"):
        batch_texts = texts[i:i + batch_size]
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL_UID,
                input=batch_texts
            )
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)

            # 添加短暂延迟避免请求过于频繁
            if i + batch_size < len(texts):
                time.sleep(0.1)

        except Exception as e:
            print(f"获取嵌入向量出错: {e}")
            # 出错时添加零向量作为占位符
            all_embeddings.extend([[0.0] * VECTOR_DIMENSION] * len(batch_texts))

    return np.array(all_embeddings, dtype=np.float32)


def search_knowledge_base(query: str, kb_name: str, top_k: int = 2) -> List[Dict]:
    """从知识库中检索相关内容"""
    kb_dir = os.path.join(KNOWLEDGE_BASE_ROOT, kb_name)

    if not os.path.exists(kb_dir):
        print(f"知识库 {kb_name} 不存在")
        return []

    # 加载索引
    index_path = os.path.join(kb_dir, "index.faiss")
    if not os.path.exists(index_path):
        print(f"索引文件 {index_path} 不存在")
        return []

    index = faiss.read_index(index_path)

    # 加载元数据
    metadata_path = os.path.join(kb_dir, "metadata.pkl")
    if not os.path.exists(metadata_path):
        print(f"元数据文件 {metadata_path} 不存在")
        return []

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    # 获取查询的嵌入向量
    query_vector = get_embeddings([query])

    # 归一化查询向量
    faiss.normalize_L2(query_vector)

    # 执行检索
    distances, indices = index.search(query_vector, top_k)

    # 获取检索结果
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:  # FAISS在没有足够结果时会返回-1
            continue

        chunk_text = metadata["chunks"][idx]
        chapter_idx = metadata.get("chunk_to_chapter_map", {}).get(idx, -1)

        results.append({
            "text": chunk_text,
            "chapter_idx": chapter_idx,
            "source_title": f"知识库{kb_name}-章节{chapter_idx + 1}" if chapter_idx >= 0 else f"知识库{kb_name}",
            "similarity": float(distances[0][i])
        })

    return results


def call_qwen3_api(query: str, context: List[Dict]) -> str:
    """调用Qwen3 API生成回答"""
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 构建系统提示词，包含知识库内容
    system_prompt = "你是一个基于知识库的智能助手，请根据提供的知识回答用户问题。"

    # 构建上下文
    context_text = "\n\n".join([f"参考内容{i + 1}：{item['text']}" for i, item in enumerate(context)])

    full_prompt = f"{query}\n\n参考知识库内容：\n{context_text}"

    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt},
            ],
            extra_body={"enable_thinking": False},
        )
        response = completion.choices[0].message.content
        return response
    except Exception as e:
        print(f"API调用出错: {e}")
        return f"抱歉，我无法处理您的请求。错误信息: {str(e)}"


def format_response_with_sources(response: str, sources: List[Dict]) -> str:
    """格式化回答，包含知识来源"""
    msg_id = int(time.time())  # 使用时间戳作为消息ID

    # 格式化回答HTML，包含知识来源
    formatted_response = f"""{response}

<div class="message-buttons">
    <button id="toggle-{msg_id}" class="toggle-sources" onclick="toggleSources({msg_id})">
        ↑ 隐藏参考来源
    </button>
</div>

<div id="sources-{msg_id}" class="knowledge-sources">"""

    # 添加检索到的知识来源
    for i, source in enumerate(sources):
        formatted_response += f"""
    <div class="source-item">
        <div class="source-title">{source["source_title"]}</div>
        <div class="source-text">{source["text"]}</div>
    </div>"""

    formatted_response += "\n</div>"

    return formatted_response


def create_history_item(title, date):
    """创建历史记录项"""
    return f"""
    <div class="history-item">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M20 12V5H4V19H13" stroke="#4B5563" stroke-width="1.5" stroke-linecap="round"/>
            <path d="M18 14V17M18 17V20M18 17H15M18 17H21" stroke="#3B82F6" stroke-width="1.5"/>
        </svg>
        <div class="history-content">
            <div class="history-title">{title}</div>
            <div class="history-date">{date}</div>
        </div>
        <div class="history-actions">
            <img  title="删除">
            <img  title="导出">
        </div>
    </div>
    """


def process_query(query: str, kb_name: str = "chapter_knowledge_base") -> Tuple[str, List[Dict]]:
    """处理用户查询：检索知识库并调用API生成回答"""
    # 1. 从知识库检索相关内容
    search_results = search_knowledge_base(query, kb_name)

    # 2. 如果没有检索到内容，返回提示
    if not search_results:
        return "对不起，我在知识库中找不到相关信息。", []

    # 3. 调用API生成回答
    response = call_qwen3_api(query, search_results)

    return response, search_results
