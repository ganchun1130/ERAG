# coding:utf-8
# @File  : naive.py
# @Author: ganchun
# @Date  :  2025/04/27
# @Description: 朴素RAG实现，直接对原始查询检索，只检索摘要

import os
import time
from typing import List, Dict, Any

# 从contextual_rewrite.py导入必要函数
from RAG.retrieval.contextual_rewrite import (
    get_embedding,
    load_knowledge_base,
    list_knowledge_bases,
    reranker
)
from config import SUB_QUERY_TOP_K, SUB_QUERY_TOP_P, FINAL_DOCS_TOP_K


def naive_retrieve_from_knowledge_base(kb_name: str,
                                       query: str,
                                       top_k: int = SUB_QUERY_TOP_K,
                                       top_p: float = SUB_QUERY_TOP_P,
                                       do_rerank: bool = True) -> List[Dict[str, Any]]:
    """
    朴素检索函数，直接对原始查询进行嵌入并检索

    参数:
        kb_name: 知识库名称
        query: 用户查询
        top_k: 检索返回的结果数量
        top_p: 检索保留的结果比例
        do_rerank: 是否进行重排序

    返回:
        检索到的文档列表
    """
    # 加载知识库
    print(f"加载知识库: {kb_name}")
    kb_data = load_knowledge_base(kb_name)
    metadata = kb_data["metadata"]
    summary_index = kb_data["summary_index"]
    documents = metadata["documents"]

    # 开始检索
    start_time = time.time()

    # 对原始查询进行嵌入
    query_embedding = get_embedding(query)

    # 检索摘要
    scores, indices = summary_index.search(query_embedding, top_k * 2)

    # 根据top_p过滤结果
    retrieve_results = []
    if scores[0].size > 0:
        max_score = scores[0][0]
        threshold = max_score * top_p

        for score, doc_idx in zip(scores[0], indices[0]):
            if score >= threshold and doc_idx < len(documents):
                doc = documents[doc_idx].copy()
                doc['relevance_score'] = float(score)
                retrieve_results.append(doc)

    # 限制结果数量
    retrieve_results = retrieve_results[:top_k]

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

    return final_docs


if __name__ == "__main__":
    # 列出所有知识库
    knowledge_bases = list_knowledge_bases()
    print(f"可用的知识库: {knowledge_bases}")

    query = "比较Qwen2.5和Qwen1.5模型架构"
    kb_name = "All"  # 替换为实际的知识库名称

    # 调用检索函数
    retrieved_docs = naive_retrieve_from_knowledge_base(kb_name, query)

    # 打印检索结果
    for i, doc in enumerate(retrieved_docs):
        print(f"结果 #{i + 1} - 相关性: {doc.get('relevance_score', 0):.4f}")
        print(f"标题: {doc.get('标题', '无标题')}")
        print(f"摘要: {doc.get('总结', '')[:100]}...")
        print('-' * 50)