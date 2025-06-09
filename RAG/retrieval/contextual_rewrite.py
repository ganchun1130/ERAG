# coding:utf-8
# @File  : contextual_rewrite.py
# @Author: ganchun
# @Date  :  2025/04/22
# @Description: 双层检索知识库查询

import os
import json
import pickle
import numpy as np
import faiss
import time
from typing import List, Dict, Any
import concurrent.futures
from openai import OpenAI
from xinference.client import Client

from Prompt.prompt_templates import query_rewrite_prompt
from scripts.use_doubao_api import use_doubao_api_custom
from config import (DOUBAO_API_KEY,SUB_QUERY_TOP_K,SUB_QUERY_TOP_P,
                    DOUBAO_MODEL,KEY_QUERY_TOP_K,KEY_QUERY_TOP_P,
                    EMBEDDING_API_URL,FINAL_DOCS_TOP_K,RERANKER_API_URL,
                    EMBEDDING_MODEL_UID,KNOWLEDGE_BASE_ROOT,RERANKER_MODEL_UID
                    )


def rewrite_query(query: str) -> str:

    try:
        prompt = query_rewrite_prompt.format(input_text=query)
        response = use_doubao_api_custom(
            api_key=DOUBAO_API_KEY,
            model=DOUBAO_MODEL,
            prompt=prompt
        )

        return response

    except Exception as e:
        print(f"查询重写API调用出错: {e}")
        # 出错时返回简单格式，仍确保后续流程能继续
        return json.dumps({
            "原始查询": query,
            "子查询序列": [query],
            "关键查询": query
        }, ensure_ascii=False)


def process_json_response(json_str: str, max_retries=1, retry_callback=None) -> Dict[str, Any]:
    """
    处理API返回的JSON结果，提取查询重写结果

    参数:
        json_str: API返回的响应文本
        max_retries: 最大重试次数
        retry_callback: 重试回调函数

    返回:
        Dict: 解析后的查询重写结果
    """
    retry_count = 0
    original_json_str = json_str

    while retry_count <= max_retries:
        try:
            # 处理可能的错误格式
            json_str = json_str.replace('""', '"').strip()
            json_str = json_str.replace('{{', '{').strip()
            json_str = json_str.replace('}}', '}').strip()

            # 尝试直接解析整个响应
            try:
                data = json.loads(json_str)
                if isinstance(data, dict) and (
                        "原始查询" in data or "子查询序列" in data or "关键查询" in data):
                    return data
            except json.JSONDecodeError:
                pass

            # 检查是否包含Markdown格式的JSON
            if "```json" in json_str and "```" in json_str:
                start = json_str.find("```json") + len("```json")
                end = json_str.find("```", start)
                if start != -1 and end > start:
                    json_content = json_str[start:end].strip()
                    try:
                        data = json.loads(json_content)
                        if isinstance(data, dict):
                            return data
                    except:
                        pass

            # 尝试查找JSON对象部分
            if '{' in json_str and '}' in json_str:
                start = json_str.find('{')
                end = json_str.rfind('}') + 1
                if start != -1 and end > start:
                    json_content = json_str[start:end]
                    try:
                        data = json.loads(json_content)
                        if isinstance(data, dict):
                            return data
                    except:
                        pass

            # 所有解析尝试失败，如果有回调函数则重试
            if retry_callback and retry_count < max_retries:
                print(f"JSON解析失败，尝试重新请求 (尝试 {retry_count + 1}/{max_retries})")
                retry_count += 1
                json_str = retry_callback()  # 获取新的响应
                continue
            else:
                # 解析失败时构造默认格式
                print(f"JSON解析失败，返回默认格式")
                return {
                    "原始查询": original_json_str,
                    "子查询序列": [original_json_str],
                    "关键查询": original_json_str,
                    "是否检索": True
                }

        except Exception as e:
            if retry_callback and retry_count < max_retries:
                print(f"处理失败({e})，尝试重新请求 (尝试 {retry_count + 1}/{max_retries})")
                retry_count += 1
                json_str = retry_callback()
            else:
                print(f"JSON解析最终失败: {e}")
                print(f"原始响应: {original_json_str[:200]}...")
                return {
                    "原始查询": original_json_str,
                    "子查询序列": [original_json_str],
                    "关键查询": original_json_str,
                    "是否检索": True
                }

    # 超过最大重试次数
    return {
        "原始查询": original_json_str,
        "子查询序列": [original_json_str],
        "关键查询": original_json_str,
        "是否检索": True
    }


def get_embedding(text: str) -> np.ndarray:
    """获取单个文本的嵌入向量"""
    client = OpenAI(api_key="not empty", base_url=EMBEDDING_API_URL)
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL_UID,
            input=text
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(embedding)
        return embedding
    except Exception as e:
        print(f"获取嵌入向量出错: {e}")
        return np.zeros((1, 1024), dtype=np.float32)

def load_knowledge_base(kb_name: str) -> Dict[str, Any]:

    kb_dir = os.path.join(KNOWLEDGE_BASE_ROOT, kb_name)

    if not os.path.exists(kb_dir):
        raise ValueError(f"知识库 '{kb_name}' 不存在")

    # 加载元数据
    with open(os.path.join(kb_dir, "metadata.pkl"), 'rb') as f:
        metadata = pickle.load(f)

    # 加载索引
    summary_index = faiss.read_index(os.path.join(kb_dir, "summary_index.faiss"))
    tag_index = faiss.read_index(os.path.join(kb_dir, "tag_index.faiss"))

    return {
        "metadata": metadata,
        "summary_index": summary_index,
        "tag_index": tag_index
    }


def retrieve_from_knowledge_base(kb_name: str,
                                 query: str,
                                 do_rerank: bool = True) -> Dict[str, Any]:
    """
    从知识库中检索相关文档，并返回查询信息和检索结果

    参数:
        kb_name: 知识库名称
        query: 用户原始查询
        do_rerank: 是否进行重排序

    返回:
        包含查询信息和检索结果的字典
    """
    # 获取重写的查询
    print(f"重写查询: {query}")
    rewritten_query_str = rewrite_query(query)
    rewritten_query = process_json_response(rewritten_query_str)

    print("查询重写结果:")
    print(f"原始查询: {rewritten_query.get('原始查询', query)}")
    print(f"子查询序列: {rewritten_query.get('子查询序列', [])}")
    print(f"关键查询: {rewritten_query.get('关键查询', '')}")
    print(f"是否检索: {rewritten_query.get('是否检索')}")

    # 提取查询信息
    original_query = query
    key_query = rewritten_query.get("关键查询", query)
    sub_queries = rewritten_query.get("子查询序列", [query])

    # 加载知识库
    print(f"加载知识库: {kb_name}")
    kb_data = load_knowledge_base(kb_name)
    metadata = kb_data["metadata"]
    summary_index = kb_data["summary_index"]
    tag_index = kb_data["tag_index"]
    documents = metadata["documents"]
    all_tags = metadata["all_tags"]
    tag_to_doc_map = metadata["tag_to_doc_map"]

    # 开始双层检索
    start_time = time.time()

    # 第一阶段: 基于子查询与摘要的相似度
    first_stage_results = set()
    sub_queries = rewritten_query.get("子查询序列", [query])

    def process_sub_query(sub_query, top_k):
        # 对单个子查询进行嵌入和检索
        sub_query_embedding = get_embedding(sub_query)
        scores, indices = summary_index.search(sub_query_embedding, top_k * 2)

        # 根据top_p进行过滤
        results = []
        if scores[0].size > 0:
            max_score = scores[0][0]
            threshold = max_score * SUB_QUERY_TOP_P

            for score, doc_idx in zip(scores[0], indices[0]):
                if score >= threshold and doc_idx < len(documents):
                    results.append((doc_idx, score))

        # 只返回top_k个结果
        return results[:top_k]

    # 并行处理每个子查询
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_sub_query, query, SUB_QUERY_TOP_K)
                   for query in sub_queries]

        for future in concurrent.futures.as_completed(futures):
            for doc_idx, score in future.result():
                first_stage_results.add(doc_idx)

    print(f"第一阶段检索完成，找到{len(first_stage_results)}个候选文档")

    # 第二阶段: 基于关键查询与标签的相似度
    key_query = rewritten_query.get("关键查询", query)
    key_query_embedding = get_embedding(key_query)

    # 对标签进行搜索
    tag_scores, tag_indices = tag_index.search(key_query_embedding,
                                               min(len(all_tags), KEY_QUERY_TOP_K * 3))

    # 根据key_query_top_p过滤标签
    relevant_docs_from_tags = set()
    if tag_scores[0].size > 0:
        max_tag_score = tag_scores[0][0]
        tag_threshold = max_tag_score * KEY_QUERY_TOP_P

        for score, tag_idx in zip(tag_scores[0], tag_indices[0]):
            if score >= tag_threshold and tag_idx < len(all_tags):
                # 如果标签存在于映射中，添加关联文档
                if tag_idx in tag_to_doc_map:
                    relevant_docs_from_tags.update(tag_to_doc_map[tag_idx])

    print(f"第二阶段检索完成，找到{len(relevant_docs_from_tags)}个标签匹配的文档")

    # 取两个阶段的并集，并去重
    final_doc_indices = first_stage_results.union(relevant_docs_from_tags)

    # 准备结果文档
    retrieve_results = []
    for doc_idx in list(final_doc_indices):
        if doc_idx < len(documents):
            doc = documents[doc_idx].copy()
            retrieve_results.append(doc)

    # 对结果进行重排序
    if do_rerank and retrieve_results:
        print("对检索结果进行重排序...")
        final_docs = reranker(query, retrieve_results)
        # 限制返回数量
        final_docs = final_docs[:FINAL_DOCS_TOP_K]
    else:
        # 如果不进行重排序，直接限制返回数量
        final_docs = retrieve_results[:FINAL_DOCS_TOP_K]

    print(f"检索完成，用时{time.time() - start_time:.2f}秒，返回{len(final_docs)}条结果")

    # 返回包含查询信息和检索结果的字典
    return {
        "query_info": {
            "原始查询": original_query,
            "关键查询": key_query,
            "子查询序列": sub_queries
        },
        "retrieved_docs": final_docs
    }


def list_knowledge_bases() -> List[str]:

    kb_mapping_path = os.path.join(KNOWLEDGE_BASE_ROOT, "kb_mapping.json")

    if not os.path.exists(kb_mapping_path):
        return []

    with open(kb_mapping_path, 'r', encoding='utf-8') as f:
        kb_mapping = json.load(f)

    return list(kb_mapping.keys())


def reranker(query: str, documents: List[Dict[str, Any]]
            ) -> List[Dict[str, Any]]:
    if not documents:
        return []

    try:
        # 连接到重排序服务
        client = Client(RERANKER_API_URL)
        model = client.get_model(RERANKER_MODEL_UID)

        # 准备语料库，提取文档的摘要或内容
        corpus = [doc.get('内容', '')  for doc in documents]
        # 调用重排序模型
        rerank_result = model.rerank(corpus, query)

        # 处理重排序结果
        if 'results' not in rerank_result:
            print("重排序结果格式不正确")
            return documents

        # 根据相关性得分重新排序文档
        reranked_docs = []
        for item in sorted(rerank_result['results'], key=lambda x: x['relevance_score'], reverse=True):
            idx = item['index']
            if 0 <= idx < len(documents):
                # 将相关性得分添加到文档中
                doc = documents[idx].copy()
                doc['relevance_score'] = item['relevance_score']
                reranked_docs.append(doc)

        return reranked_docs

    except Exception as e:
        print(f"重排序过程出错: {e}")
        return documents  # 出错时返回原始文档列表


if __name__ == "__main__":

    # 列出所有知识库
    knowledge_bases = list_knowledge_bases()
    print(f"可用的知识库: {knowledge_bases}")

    query = "比较Qwen2.5和Qwen1.5模型架构"
    kb_name = "All"  # 替换为实际的知识库名称
    do_rerank = True  # 是否进行重排序

    # 调用检索函数
    results = retrieve_from_knowledge_base(kb_name, query, do_rerank)

    # 打印查询信息
    print("\n查询信息:")
    print(f"原始查询: {results['query_info']['原始查询']}")
    print(f"关键查询: {results['query_info']['关键查询']}")
    print(f"子查询序列: {results['query_info']['子查询序列']}")

    # 打印检索结果
    print(f"\n检索到的文档({len(results['retrieved_docs'])}):")
