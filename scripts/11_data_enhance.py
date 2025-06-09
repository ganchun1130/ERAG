# coding:utf-8
# @File  : 11_data_enhance.py
# @Author: ganchun
# @Date  :  2025/04/21
# @Description: 数据增强

import os
import time
import pandas as pd
from tqdm import tqdm
from Prompt.prompt_templates import data_enhance_prompt
from use_doubao_api import use_doubao_api_custom
from other_functions import get_cache_key, append_to_csv, process_json_qa


def data_enhance(api_key, model, input_csv, output_csv, batch_size=10):
    """
    读取输入CSV文件，使用API进行数据增强，并将结果保存到新的CSV文件

    参数:
        api_key: API密钥
        model: 模型名称
        input_csv: 输入CSV文件路径
        output_csv: 输出CSV文件路径
        batch_size: 每次处理的问答对数量
    """
    print(f"开始数据增强处理: {input_csv}")

    # 创建输出目录
    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)

    # 读取原始CSV文件
    df_original = pd.read_csv(input_csv, encoding='utf-8')
    print(f"读取原始数据: {len(df_original)}行")

    # 初始化缓存和计数器
    cache = {}
    api_call_count = 0
    cache_hit_count = 0
    total_enhanced_count = 0

    # 首先保存原始数据到新文件
    df_original.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"已保存原始数据到新文件")

    # 按批次处理数据
    total_batches = (len(df_original) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(df_original), batch_size), desc="处理批次", total=total_batches):
        # 获取当前批次的数据
        batch_df = df_original.iloc[i:i + batch_size]

        # 构建输入文本
        batch_text = []
        for _, row in batch_df.iterrows():
            qa_item = {
                "question": row['问题'],
                "difficulty": row['问题类型'] if '问题类型' in row else "simple",
                "answer": row['答案']
            }
            batch_text.append(qa_item)

        # 转换为JSON格式文本
        import json
        input_text = json.dumps(batch_text, ensure_ascii=False, indent=2)

        # 构建提示词
        prompt = data_enhance_prompt.format(input_text=input_text)

        # 检查缓存
        cache_key = get_cache_key(prompt)
        if cache_key in cache:
            print("使用缓存结果")
            response_text = cache[cache_key]
            cache_hit_count += 1
        else:
            # API调用
            api_call_count += 1
            if api_call_count % 5 == 0:
                print(f"已完成{api_call_count}次API调用，暂停3秒...")
                time.sleep(3)

            response_text = use_doubao_api_custom(api_key=api_key, model=model, prompt=prompt)
            cache[cache_key] = response_text

        # 定义重试回调函数
        def retry_api_call():
            print("重新调用API...")
            time.sleep(2)
            return use_doubao_api_custom(api_key=api_key, model=model, prompt=prompt)

        # 解析返回的JSON数据
        enhanced_qa_list = process_json_qa(
            response_text=response_text,
            max_retries=3,
            retry_callback=retry_api_call
        )

        if not enhanced_qa_list:
            print("警告: 此批次未返回有效数据")
            continue

        # 转换为DataFrame格式
        enhanced_data = []
        for qa_item in enhanced_qa_list:
            if isinstance(qa_item, dict) and 'question' in qa_item and 'answer' in qa_item:
                enhanced_data.append({
                    '问题类型': qa_item.get('difficulty', 'simple'),
                    '问题': qa_item.get('question', ''),
                    '答案': qa_item.get('answer', '')
                })

        # 追加到输出CSV
        if enhanced_data:
            batch_enhanced_count = len(enhanced_data)
            total_enhanced_count += batch_enhanced_count
            append_to_csv(enhanced_data, output_csv)
            print(f"已增强 {batch_enhanced_count} 条数据")

    # 打印统计信息
    print("\n数据增强完成!")
    print(f"原始数据: {len(df_original)}条")
    print(f"增强后新增: {total_enhanced_count}条")
    print(f"总数据量: {len(df_original) + total_enhanced_count}条")
    print(f"API调用次数: {api_call_count}")
    print(f"缓存命中次数: {cache_hit_count}")


if __name__ == "__main__":
    # 配置参数
    API_KEY = ""
    MODEL = "ep-20250422013101-xjhks"  # Doubao-1.5-pro-256k

    # 输入输出路径
    INPUT_CSV = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\SFT_Data\raw\all_raw.csv"
    OUTPUT_CSV = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\SFT_Data\all_raw_enhanced.csv"

    # 执行数据增强
    data_enhance(API_KEY, MODEL, INPUT_CSV, OUTPUT_CSV, batch_size=10)
