# coding:utf-8
# @File  : build_knowledge_base.py
# @Author: ganchun
# @Date  :  2025/04/22
# @Description: 构建本地化向量知识库，支持多个知识库命名

import os
import csv
import json
import numpy as np
import faiss
import pickle
from tqdm import tqdm
import time
import argparse
from datetime import datetime
from openai import OpenAI

# 配置参数
EMBEDDING_MODEL_UID = None  # 将在运行时从API获取
EMBEDDING_API_URL = "http://localhost:9997/v1"
KNOWLEDGE_BASE_ROOT = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\Knowledge_Base"
CSV_ROOT = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\Knowledge_Docs"
VECTOR_DIMENSION = 1024  # bge-m3的维度


def get_embeddings(texts):
    """使用OpenAI API获取文本的嵌入向量"""
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


def read_csv_file(file_path):
    """读取单个CSV文件并返回文档列表（只使用内容、总结和标签字段）"""
    documents = []

    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # 读取表头

        # 查找各字段的索引位置
        content_idx = header.index('内容') if '内容' in header else 2
        summary_idx = header.index('总结') if '总结' in header else 3
        tags_idx = header.index('标签') if '标签' in header else 4

        # 确认必要的字段存在
        required_fields = ['内容', '总结', '标签']
        missing_fields = [field for field in required_fields if field not in header]

        if missing_fields:
            print(f"警告: 文件 {os.path.basename(file_path)} 中缺少必要字段: {', '.join(missing_fields)}")
            return []

        for row in reader:
            if not row or len(row) <= max(content_idx, summary_idx, tags_idx):
                continue

            # 处理标签字段
            tags = row[tags_idx]
            if tags:
                try:
                    if isinstance(tags, str):
                        if tags.startswith('[') and tags.endswith(']'):
                            tags = json.loads(tags)
                        else:
                            tags = [tag.strip() for tag in tags.split(',')]
                except:
                    tags = [tags]
            else:
                tags = []

            doc = {
                '内容': row[content_idx],
                '总结': row[summary_idx],
                '标签': tags
            }
            documents.append(doc)

    return documents


def build_single_knowledge_base(csv_file, kb_name):
    """构建单个知识库"""
    start_time = datetime.now()
    print(f"\n开始构建知识库 '{kb_name}'...")

    # 创建知识库目录
    kb_dir = os.path.join(KNOWLEDGE_BASE_ROOT, kb_name)
    os.makedirs(kb_dir, exist_ok=True)

    # 读取文档
    documents = read_csv_file(csv_file)
    if not documents:
        print(f"错误: 无法从文件 {os.path.basename(csv_file)} 读取有效数据")
        return False

    print(f"读取了 {len(documents)} 个文档")

    # 提取摘要文本
    summaries = [doc['总结'] for doc in documents]
    print("为总结生成嵌入向量...")
    summary_vectors = get_embeddings(summaries)

    # 提取标签
    all_tags = []
    tag_to_doc_map = {}  # 标签索引到文档索引的映射

    for doc_idx, doc in enumerate(documents):
        for tag in doc['标签']:
            if tag:
                tag_idx = len(all_tags)
                all_tags.append(tag)
                if tag_idx not in tag_to_doc_map:
                    tag_to_doc_map[tag_idx] = []
                tag_to_doc_map[tag_idx].append(doc_idx)

    print(f"总共有 {len(all_tags)} 个标签")
    print("为标签生成嵌入向量...")
    tag_vectors = get_embeddings(all_tags)

    # 构建FAISS索引 - 使用内积相似度（归一化后等价于余弦相似度）
    print("构建FAISS索引...")

    # 为总结创建索引
    summary_index = faiss.IndexFlatIP(VECTOR_DIMENSION)
    # 归一化向量以使用余弦相似度
    faiss.normalize_L2(summary_vectors)
    summary_index.add(summary_vectors)

    # 为标签创建索引
    tag_index = faiss.IndexFlatIP(VECTOR_DIMENSION)
    # 归一化向量以使用余弦相似度
    faiss.normalize_L2(tag_vectors)
    tag_index.add(tag_vectors)

    # 保存索引
    print("保存索引和元数据...")
    faiss.write_index(summary_index, os.path.join(kb_dir, "summary_index.faiss"))
    faiss.write_index(tag_index, os.path.join(kb_dir, "tag_index.faiss"))

    # 保存元数据
    metadata = {
        "name": kb_name,
        "documents": documents,
        "all_tags": all_tags,
        "tag_to_doc_map": tag_to_doc_map,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(os.path.join(kb_dir, "metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)

    # 保存知识库信息
    kb_info = {
        "name": kb_name,
        "document_count": len(documents),
        "tag_count": len(all_tags),
        "source_file": os.path.basename(csv_file),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(os.path.join(kb_dir, "info.json"), 'w', encoding='utf-8') as f:
        json.dump(kb_info, f, ensure_ascii=False, indent=2)

    end_time = datetime.now()
    time_used = end_time - start_time
    print(f"知识库 '{kb_name}' 构建完成! 耗时: {time_used}")
    return True


def build_all_knowledge_bases():
    """构建所有CSV文件的知识库"""
    # 确保输出目录存在
    os.makedirs(KNOWLEDGE_BASE_ROOT, exist_ok=True)

    # 读取所有CSV文件
    csv_files = [os.path.join(CSV_ROOT, f) for f in os.listdir(CSV_ROOT)
                 if f.endswith('.csv')]

    if not csv_files:
        print(f"错误: 在目录 {CSV_ROOT} 中没有找到CSV文件")
        return

    print(f"找到 {len(csv_files)} 个CSV文件")

    # 构建知识库映射文件
    kb_mapping = {}

    # 处理每个CSV文件
    for i, csv_file in enumerate(csv_files):
        file_name = os.path.basename(csv_file)
        # 使用文件名作为知识库名称（去掉.csv扩展名）
        kb_name = os.path.splitext(file_name)[0]

        print(f"\n[{i + 1}/{len(csv_files)}] 处理文件: {file_name}")

        success = build_single_knowledge_base(
            csv_file=csv_file,
            kb_name=kb_name
        )

        if success:
            kb_mapping[kb_name] = {
                "path": os.path.join(KNOWLEDGE_BASE_ROOT, kb_name),
                "source_file": file_name
            }

    # 保存知识库映射
    with open(os.path.join(KNOWLEDGE_BASE_ROOT, "kb_mapping.json"), 'w', encoding='utf-8') as f:
        json.dump(kb_mapping, f, ensure_ascii=False, indent=2)

    print(f"\n所有知识库构建完成! 总共构建了 {len(kb_mapping)} 个知识库")
    print(f"知识库列表: {', '.join(kb_mapping.keys())}")


def build_specific_knowledge_base(csv_file_path, kb_name):
    """构建指定的CSV文件知识库"""
    if not os.path.exists(csv_file_path):
        print(f"错误: 文件 {csv_file_path} 不存在")
        return

    # 确保输出目录存在
    os.makedirs(KNOWLEDGE_BASE_ROOT, exist_ok=True)

    success = build_single_knowledge_base(
        csv_file=csv_file_path,
        kb_name=kb_name
    )

    if success:
        # 更新知识库映射
        kb_mapping_path = os.path.join(KNOWLEDGE_BASE_ROOT, "kb_mapping.json")

        if os.path.exists(kb_mapping_path):
            with open(kb_mapping_path, 'r', encoding='utf-8') as f:
                kb_mapping = json.load(f)
        else:
            kb_mapping = {}

        kb_mapping[kb_name] = {
            "path": os.path.join(KNOWLEDGE_BASE_ROOT, kb_name),
            "source_file": os.path.basename(csv_file_path)
        }

        with open(kb_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(kb_mapping, f, ensure_ascii=False, indent=2)


def get_embedding_model_uid():
    """获取embedding模型的UID"""
    from xinference.client import RESTfulClient
    client = RESTfulClient("http://localhost:9997")

    # 尝试列出已有模型，检查是否已加载
    models = client.list_models()
    for model in models:
        if model == "bge-m3":
            print(f"已找到已加载的embedding模型：{model}")
            return model

    # 如果没有找到已加载的模型，则加载模型
    model_uid = client.launch_model(
        model_name="bge-m3",
        model_type="embedding",
        model_path=r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Model\EM\bge-m3"
    )
    return model_uid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='构建向量知识库')
    parser.add_argument('--all', action='store_true', help='构建所有CSV文件的知识库')
    parser.add_argument('--csv', type=str, help='指定CSV文件路径')
    parser.add_argument('--name', type=str, help='指定知识库名称')

    args = parser.parse_args()

    # 获取embedding模型UID
    print("获取embedding模型UID...")
    EMBEDDING_MODEL_UID = get_embedding_model_uid()
    print(f"使用embedding模型UID: {EMBEDDING_MODEL_UID}")

    if args.all:
        build_all_knowledge_bases()
    elif args.csv and args.name:
        build_specific_knowledge_base(args.csv, args.name)
    else:
        print("请使用 --all 选项构建所有知识库，或使用 --csv 和 --name 选项构建特定知识库")
        print("例如: python build_knowledge_base.py --all")
        print("或者: python build_knowledge_base.py --csv 文件路径.csv --name 知识库名称")