# coding:utf-8
# @File  : 17_build_kb.py
# @Author: ganchun
# @Date  :  2025/06/06
# @Description:
# coding:utf-8
import os
import re
import json
import numpy as np
import faiss
import pickle
from tqdm import tqdm
import time
from datetime import datetime
from openai import OpenAI
from typing import List

# 配置参数
EMBEDDING_MODEL_UID = "bge-m3"  # 将在运行时获取实际UID
EMBEDDING_API_URL = "http://localhost:9997/v1"
KNOWLEDGE_BASE_ROOT = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\Knowledge_Base\jisuanjiaoyuxue"
VECTOR_DIMENSION = 1024  # bge-m3的维度
INPUT_FILE = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\Knowledge_Docs\final\chapter_merged.txt"


class ChineseTextSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        # 使用正则表达式分割中文句子
        sent_sep_pattern = re.compile(
            '([﹒﹔﹖﹗．。！？]["'"」』]{0,2}|(?=["'"「『]{1,2}|$))')
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)

        # 根据chunk_size合并句子形成文本块
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sent_list:
            sentence_length = len(sentence)

            # 如果当前块加上这个句子超过了限制，就保存当前块并开始新块
            if current_length + sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append("".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # 添加最后一个块
        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks


def get_embeddings(texts):
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


def get_embedding_model_uid():
    """获取embedding模型的UID"""
    try:
        from xinference.client import RESTfulClient
        client = RESTfulClient("http://localhost:9997")

        # 查找已加载的模型
        models = client.list_models()
        for model in models:
            if model == "bge-m3":
                print(f"已找到已加载的embedding模型：{model}")
                return model

        # 如果未找到，则加载模型
        model_uid = client.launch_model(
            model_name="bge-m3",
            model_type="embedding",
            model_path="./models/bge-m3"
        )
        return model_uid
    except Exception as e:
        print(f"获取模型UID时出错: {e}")
        return "bge-m3"  # 默认返回模型名称


def build_knowledge_base():
    """构建知识库"""
    start_time = datetime.now()
    print(f"\n开始构建知识库...")

    # 创建知识库目录
    kb_name = "chapter_knowledge_base"
    kb_dir = os.path.join(KNOWLEDGE_BASE_ROOT, kb_name)
    os.makedirs(kb_dir, exist_ok=True)

    # 读取输入文件
    print(f"读取文件 {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        chapters = f.readlines()

    print(f"共读取到 {len(chapters)} 个章节")

    # 初始化分块器
    splitter = ChineseTextSplitter(chunk_size=500, chunk_overlap=50)

    # 处理每个章节并分块
    all_chunks = []
    chunk_to_chapter_map = {}  # 记录每个块属于哪个章节

    for chapter_idx, chapter_text in enumerate(tqdm(chapters, desc="对章节进行分块")):
        chapter_chunks = splitter.split_text(chapter_text)

        # 记录每个块与章节的对应关系
        for i in range(len(chapter_chunks)):
            chunk_idx = len(all_chunks)
            all_chunks.append(chapter_chunks[i])
            chunk_to_chapter_map[chunk_idx] = chapter_idx

    print(f"分块完成，共生成 {len(all_chunks)} 个文本块")

    # 获取模型UID
    global EMBEDDING_MODEL_UID
    EMBEDDING_MODEL_UID = get_embedding_model_uid()
    print(f"使用embedding模型UID: {EMBEDDING_MODEL_UID}")

    # 生成嵌入向量
    print("为文本块生成嵌入向量...")
    chunk_vectors = get_embeddings(all_chunks)

    # 构建FAISS索引 - 使用内积相似度（归一化后等价于余弦相似度）
    print("构建FAISS索引...")
    index = faiss.IndexFlatIP(VECTOR_DIMENSION)

    # 归一化向量以使用余弦相似度
    faiss.normalize_L2(chunk_vectors)
    index.add(chunk_vectors)

    # 保存索引
    print("保存索引和元数据...")
    faiss.write_index(index, os.path.join(kb_dir, "index.faiss"))

    # 保存元数据
    metadata = {
        "name": kb_name,
        "chunks": all_chunks,
        "chunk_to_chapter_map": chunk_to_chapter_map,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(os.path.join(kb_dir, "metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)

    # 保存知识库信息
    kb_info = {
        "name": kb_name,
        "chunk_count": len(all_chunks),
        "chapter_count": len(chapters),
        "source_file": INPUT_FILE,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(os.path.join(kb_dir, "info.json"), 'w', encoding='utf-8') as f:
        json.dump(kb_info, f, ensure_ascii=False, indent=2)

    end_time = datetime.now()
    time_used = end_time - start_time
    print(f"知识库构建完成! 耗时: {time_used}")
    print(f"知识库保存路径: {kb_dir}")


if __name__ == "__main__":
    build_knowledge_base()