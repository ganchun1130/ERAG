# coding:utf-8
# @File  : build_eval_data.py
# @Author: ganchun
# @Date  :  2025/04/25
# @Description: 构建评估数据集

import os
import json
import time
import pandas as pd
import argparse
import jsonlines
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from openai import OpenAI

# 导入相关函数
from Prompt.prompt_templates import query_rewrite_prompt,doubao_rag_prompt,naive_rag_prompt
from RAG.retrieval.contextual_rewrite import retrieve_from_knowledge_base, list_knowledge_bases
from config import DOUBAO_API_URL, DOUBAO_API_KEY, DOUBAO_MODEL_MAPPING

# 设置参数
EVAL_DATA_DIR = r"/Data/Evaluation/ERAG"
RAW_DATA_PATH = r"/Data/Evaluation/raw/eval_data.csv"
DEFAULT_KB = "All"  # 使用的默认知识库


def read_questions_from_csv(csv_path: str, question_type: str = 'all') -> pd.DataFrame:
    """
    从CSV文件读取问题数据

    参数:
        csv_path: CSV文件路径
        question_type: 问题类型过滤，'all'表示所有问题，'complex'表示只读取complex类型问题，'simple'表示只读取simple类型问题

    返回:
        包含过滤后问题的DataFrame
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)

        # 确保列名符合预期
        expected_columns = ['问题类型', '问题', '答案']
        if not all(col in df.columns for col in expected_columns):
            raise ValueError(f"CSV文件必须包含以下列: {expected_columns}")

        # 统一列名格式
        df = df.rename(columns={
            '问题类型': 'complex',
            '问题': 'question',
            '答案': 'answer'
        })

        # 根据问题类型过滤数据
        if question_type.lower() == 'complex':
            df = df[df['complex'] == 'complex']
        elif question_type.lower() == 'simple':
            df = df[df['complex'] == 'simple']
        # 'all'情况下不需要过滤

        return df

    except Exception as e:
        print(f"读取CSV文件出错: {e}")
        return pd.DataFrame()


def process_question(question: str, kb_name: str) -> Dict[str, Any]:
    """处理单个问题，进行检索和生成答案"""
    try:
        # 从知识库检索相关内容
        retrieval_result = retrieve_from_knowledge_base(kb_name, question, do_rerank=True)

        # 提取查询信息
        query_info = retrieval_result["query_info"]
        original_query = query_info["原始查询"]
        key_query = query_info["关键查询"]
        sub_queries = query_info["子查询序列"]

        # 提取检索到的文档内容
        retrieved_docs = retrieval_result["retrieved_docs"]
        contexts = []
        for doc in retrieved_docs:
            if "内容" in doc:
                contexts.append(doc["内容"])
            else:
                contexts.append("")

        # 调用豆包API生成回答
        client = OpenAI(api_key=DOUBAO_API_KEY, base_url=DOUBAO_API_URL)

        # 构建提示词
        prompt = doubao_rag_prompt.format(
            original_query=original_query,
            key_query=key_query,
            sub_query_list=json.dumps(sub_queries, ensure_ascii=False),
            retrieved_results=json.dumps(contexts, ensure_ascii=False)
        )

        messages = [
            {"role": "system", "content": "你是一个专业的知识助手。"},
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            model=DOUBAO_MODEL_MAPPING["Doubao-1.5-pro-32k"],
            messages=messages,
            max_tokens=1024
        )

        model_response = response.choices[0].message.content.strip()

        # 解析回答，提取最终答案字段
        try:
            response_json = json.loads(model_response)
            final_answer = response_json.get("回答", "")
        except:
            # 如果解析失败，尝试提取JSON部分
            try:
                start = model_response.find("{")
                end = model_response.rfind("}") + 1
                if start != -1 and end > start:
                    json_content = model_response[start:end]
                    response_json = json.loads(json_content)
                    final_answer = response_json.get("回答", "")
                else:
                    final_answer = model_response
            except:
                final_answer = model_response

        return {
            "retrieved_contexts": contexts,
            "response": final_answer
        }

    except Exception as e:
        print(f"处理问题出错: {e}")
        return {
            "retrieved_contexts": [],
            "response": f"处理出错: {str(e)}"
        }


def build_evaluation_dataset(kb_name: str, output_dir: str, chunk_size: int = 10):
    """构建评估数据集"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 定义输出文件路径
    all_questions_file = os.path.join(output_dir, "all_questions_eval.jsonl")
    complex_questions_file = os.path.join(output_dir, "complex_questions_eval.jsonl")

    # 读取所有问题
    print(f"正在从{RAW_DATA_PATH}读取问题...")
    df = read_questions_from_csv(RAW_DATA_PATH, 'all')

    if df.empty:
        print("没有找到有效的问题数据")
        return

    print(f"共读取到{len(df)}个问题")

    # 处理问题并构建评估数据
    all_results = []
    total_chunks = (len(df) + chunk_size - 1) // chunk_size

    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        chunk_df = df.iloc[start_idx:end_idx]

        print(f"处理第{i + 1}/{total_chunks}块，共{end_idx - start_idx}个问题")

        for _, row in tqdm(chunk_df.iterrows(), total=len(chunk_df)):
            question = row['question']
            reference = row['answer']

            # 处理问题
            result = process_question(question, kb_name)

            # 构建评估数据
            eval_data = {
                "user_input": question,
                "retrieved_contexts": result["retrieved_contexts"],
                "response": result["response"],
                "reference": reference
            }

            all_results.append(eval_data)

            # 每处理完一块问题后，就更新文件
            if _ % chunk_size == 0 or _ == len(chunk_df) - 1:
                with jsonlines.open(all_questions_file, 'w') as writer:
                    writer.write_all(all_results)

                with jsonlines.open(complex_questions_file, 'w') as writer:
                    writer.write_all([r for r in all_results if r.get("question_type") == "complex"])

            # 稍微延迟，避免API请求过于频繁
            time.sleep(0.5)

    # 处理完成后，打印统计信息
    all_count = len(all_results)
    complex_count = len([r for r in all_results if r.get("question_type") == "complex"])

    print(f"评估数据构建完成!")
    print(f"所有问题数量: {all_count}")
    print(f"复杂问题数量: {complex_count}")
    print(f"全部问题评估数据保存在: {all_questions_file}")
    print(f"复杂问题评估数据保存在: {complex_questions_file}")

def main():
    parser = argparse.ArgumentParser(description="构建评估数据集")
    parser.add_argument("--kb", type=str, default=DEFAULT_KB, help="知识库名称")
    parser.add_argument("--output", type=str, default=EVAL_DATA_DIR, help="输出目录")
    parser.add_argument("--chunk", type=int, default=5, help="每次处理的问题数量")

    args = parser.parse_args()

    # 检查知识库是否存在
    kb_list = list_knowledge_bases()
    if args.kb not in kb_list:
        print(f"错误: 知识库'{args.kb}'不存在")
        print(f"可用知识库: {kb_list}")
        return

    # 构建评估数据集
    build_evaluation_dataset(args.kb, args.output, args.chunk)


if __name__ == "__main__":
    main()
