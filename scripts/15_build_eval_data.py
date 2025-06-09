# coding:utf-8
# @File  : 15_build_eval_data.py
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

# 导入相关配置
from config import DOUBAO_API_KEY, DOUBAO_BOT_ID

# 设置参数
EVAL_DATA_DIR = r"/Data/Evaluation/naive"
RAW_DATA_PATH = r"/Data/Evaluation/raw\eval_data.csv"


def doubao_kb_bot_api(api_key, bot_model, prompt):
    """调用豆包智能体API进行问答"""
    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3/bots",
        api_key=api_key
    )

    try:
        # 调用智能体API
        completion = client.chat.completions.create(
            model=bot_model,  # bot-20XXX 为智能体ID
            messages=[
                {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
                {"role": "user", "content": prompt},
            ]
        )
        response = completion.choices[0].message.content
        reference = completion.references

        return {
            "response": response,
            "references": reference
        }

    except Exception as e:
        print(f"调用豆包API出错: {e}")
        return {
            "response": f"处理出错: {str(e)}",
            "references": []
        }


def read_questions_from_csv(csv_path: str, question_type: str = 'all') -> pd.DataFrame:
    """从CSV文件读取问题数据"""
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


def process_question(question: str, bot_id: str) -> Dict[str, Any]:
    """处理单个问题，获取智能体回复与检索内容"""
    try:
        # 使用智能体进行问答
        result = doubao_kb_bot_api(DOUBAO_API_KEY, bot_id, question)

        # 提取响应和引用内容
        response = result.get("response", "")
        references = result.get("references", [])

        # 提取检索的上下文
        contexts = []
        for ref in references:
            if isinstance(ref, dict) and "content" in ref:
                contexts.append(ref.get("content", ""))

        # 如果没有提取到上下文，尝试从响应中解析
        if not contexts and "检索结果" in response:
            try:
                start = response.find("检索结果") + len("检索结果")
                end = response.find("回答")
                if start != -1 and end > start:
                    contexts_str = response[start:end].strip()
                    # 尝试解析JSON或列表格式
                    if contexts_str.startswith("[") and contexts_str.endswith("]"):
                        contexts = json.loads(contexts_str)
            except:
                pass

        return {
            "retrieved_contexts": contexts,
            "response": response
        }

    except Exception as e:
        print(f"处理问题出错: {e}")
        return {
            "retrieved_contexts": [],
            "response": f"处理出错: {str(e)}"
        }


def build_evaluation_dataset(bot_id: str, output_dir: str, chunk_size: int = 10):
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
    complex_results = []

    total_chunks = (len(df) + chunk_size - 1) // chunk_size

    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        chunk_df = df.iloc[start_idx:end_idx]

        print(f"处理第{i + 1}/{total_chunks}块，共{end_idx - start_idx}个问题")

        for _, row in tqdm(chunk_df.iterrows(), total=len(chunk_df)):
            question = row['question']
            reference = row['answer']
            question_type = row.get('complex', 'unknown')  # 获取问题类型

            # 处理问题
            result = process_question(question, bot_id)

            # 构建评估数据
            eval_data = {
                "user_input": question,
                "retrieved_contexts": result["retrieved_contexts"],
                "response": result["response"],
                "reference": reference
            }

            all_results.append(eval_data)

            # 根据问题类型分类
            if question_type == 'complex':
                complex_results.append(eval_data)


        # 每处理完一块问题后，就更新文件
        with jsonlines.open(all_questions_file, 'w') as writer:
            writer.write_all(all_results)

        with jsonlines.open(complex_questions_file, 'w') as writer:
            writer.write_all(complex_results)

        # 稍微延迟，避免API请求过于频繁
        time.sleep(0.5)

    # 处理完成后，打印统计信息
    print(f"评估数据构建完成!")
    print(f"所有问题数量: {len(all_results)}")
    print(f"复杂问题数量: {len(complex_results)}")
    print(f"全部问题评估数据保存在: {all_questions_file}")
    print(f"复杂问题评估数据保存在: {complex_questions_file}")


def main():
    parser = argparse.ArgumentParser(description="构建评估数据集")
    parser.add_argument("--bot_id", type=str, default=DOUBAO_BOT_ID, help="豆包智能体ID")
    parser.add_argument("--output", type=str, default=EVAL_DATA_DIR, help="输出目录")
    parser.add_argument("--chunk", type=int, default=5, help="每次处理的问题数量")

    args = parser.parse_args()

    # 构建评估数据集
    build_evaluation_dataset(args.bot_id, args.output, args.chunk)


if __name__ == "__main__":
    main()
    # 直接调用函数进行测试
    # response = doubao_kb_bot_api(
    #     api_key=DOUBAO_API_KEY,
    #     bot_model=DOUBAO_BOT_ID,
    #     prompt="比较Qwen2.5和Qwen2模型架构"
    # )
    # answer = response.get("response", "")
    # reference = response.get("references", [])
    # print(answer)
    # print(reference)