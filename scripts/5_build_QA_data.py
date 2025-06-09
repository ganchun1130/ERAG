# coding:utf-8
# @File  : 5_build_QA_data.py
# @Author: ganchun
# @Date  :  2025/04/20
# @Description: 生成SFT问答数据集

import os
import json
import time
from tqdm import tqdm
from use_doubao_api import use_doubao_api_custom
from Prompt.prompt_templates import build_SFT_data_prompt_1
from RAG.ingest.semantic_segment import return_csv_file_list, read_csv_rows
from other_functions import get_cache_key, append_to_csv

def process_json_qa(response_text, max_retries=3, retry_callback=None):
    """
    处理API返回的JSON结果，提取问答对

    Args:
        response_text: API返回的响应文本
        max_retries: 最大重试次数
        retry_callback: 重试回调函数

    Returns:
        list: 解析后的问答对列表
    """
    retry_count = 0

    while retry_count <= max_retries:
        try:
            # 处理可能的错误格式
            response_text = response_text.replace('""', '"').strip()

            # 尝试直接解析整个响应
            try:
                data = json.loads(response_text)
                if isinstance(data, list):
                    return data  # 已经是符合格式的问答列表
                if isinstance(data, dict) and 'question' in data and 'answer' in data:
                    return [data]  # 单个问答对象，包装为列表
            except json.JSONDecodeError:
                pass

            # 查找JSON数组部分
            if '[' in response_text and ']' in response_text:
                start = response_text.find('[')
                end = response_text.rfind(']') + 1
                if start != -1 and end > start:
                    json_str = response_text[start:end]
                    try:
                        data = json.loads(json_str)
                        if isinstance(data, list):
                            return data
                    except:
                        pass

            # 查找JSON对象部分
            if '{' in response_text and '}' in response_text:
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start != -1 and end > start:
                    json_str = response_text[start:end]
                    try:
                        data = json.loads(json_str)
                        # 如果是单个问答对象
                        if isinstance(data, dict) and 'question' in data and 'answer' in data:
                            return [data]
                    except:
                        pass

            # 所有解析尝试失败，如果有回调函数则重试
            if retry_callback and retry_count < max_retries:
                print(f"JSON解析失败，尝试重新请求 (尝试 {retry_count + 1}/{max_retries})")
                retry_count += 1
                response_text = retry_callback()  # 获取新的响应
                continue
            else:
                raise ValueError("无法解析为有效的问答格式")

        except Exception as e:
            if retry_callback and retry_count < max_retries:
                print(f"处理失败({e})，尝试重新请求 (尝试 {retry_count + 1}/{max_retries})")
                retry_count += 1
                response_text = retry_callback()  # 获取新的响应
            else:
                print(f"JSON解析最终失败: {e}")
                print(f"原始响应: {response_text[:200]}...")
                return []  # 所有尝试都失败，返回空列表

    return []  # 超过最大重试次数


def build_SFT_data(api_key, model, input_dir, output_dir):
    """构建SFT训练数据"""
    os.makedirs(output_dir, exist_ok=True)

    # 初始化缓存字典
    cache = {}

    # 获取所有CSV文件列表
    csv_files = return_csv_file_list(input_dir)
    print(f"找到 {len(csv_files)} 个CSV文件")

    # 统计API调用计数
    api_call_count = 0
    cache_hit_count = 0

    # 处理每个CSV文件
    for csv_file in csv_files:
        file_name = os.path.basename(csv_file)
        kb_name = os.path.splitext(file_name)[0]
        print(f"\n处理文件: {file_name}")

        # 准备输出文件
        output_file = os.path.join(output_dir, f"{kb_name}_QA.csv")

        # 读取CSV文件内容
        rows = read_csv_rows(csv_file)
        print(f"读取到 {len(rows)} 行数据")

        # 准备输出数据
        qa_data = []

        # 处理每一行内容
        for row in tqdm(rows, desc=f"处理 {kb_name} 的内容"):
            try:
                # 检查CSV行结构
                if not row or not isinstance(row, list):
                    continue

                # 假设CSV有标题行，且内容在某个固定位置
                # 这里需要根据你的CSV结构调整列索引
                content_index = 2  # 假设内容在第一列
                title_index = 0  # 假设章节标题在第二列
                subtitle_index = 1  # 假设子标题在第三列

                # 获取内容字段
                content = row[content_index] if len(row) > content_index else ''
                if not content:
                    continue

                # 构建输入文本
                chapter_title = row[title_index] if len(row) > title_index else ''
                sub_title = row[subtitle_index] if len(row) > subtitle_index else ''
                context = f"{chapter_title}\n"
                context += f"{sub_title}\n"
                context += f"{content}"
                print(context)

                # 构建提示词
                prompt = build_SFT_data_prompt_1.format(input_text=context)
                print("pr:",prompt)

                # 检查缓存
                cache_key = get_cache_key(prompt)
                if cache_key in cache:
                    print("使用缓存结果")
                    cache_hit_count += 1
                    qa_list = cache[cache_key]
                else:
                    # 定义重试回调函数
                    def retry_api_call():
                        print("重新调用API...")
                        time.sleep(2)  # 添加延迟避免频繁调用
                        return use_doubao_api_custom(api_key=api_key, prompt=prompt, model=model)

                    # API调用
                    api_call_count += 1
                    if api_call_count % 10 == 0:
                        print("已完成10次API调用，暂停2秒...")
                        time.sleep(2)

                    response = use_doubao_api_custom(api_key=api_key, prompt=prompt, model=model)

                    # 使用优化后的JSON处理函数解析返回结果
                    qa_list = process_json_qa(
                        response_text=response,
                        max_retries=3,
                        retry_callback=retry_api_call
                    )

                    # 保存到缓存
                    if qa_list:
                        cache[cache_key] = qa_list

                # 提取问答对数据
                for qa_item in qa_list:
                    if isinstance(qa_item, dict) and 'question' in qa_item and 'answer' in qa_item:
                        qa_data.append({
                            '问题类型': qa_item.get('difficulty', ''),
                            '问题': qa_item.get('question', ''),
                            '答案': qa_item.get('answer', '')
                        })

                # 添加延迟避免API调用过于频繁
                time.sleep(0.5)

            except Exception as e:
                print(f"处理行时出错: {e}")
                continue

        # 将结果保存为CSV
        if qa_data:
            if append_to_csv(qa_data, output_file):
                print(f"已生成并保存 {len(qa_data)} 条问答对至 {output_file}")
            else:
                print(f"警告: {kb_name} 未生成任何问答对")
        else:
            print(f"警告: {kb_name} 未生成任何问答对")

    # 打印统计信息
    print("\n数据构建完成!")
    print(f"API调用次数: {api_call_count}")
    print(f"缓存命中次数: {cache_hit_count}")


if __name__ == "__main__":
    # 配置参数
    INPUT_CSV_DIR = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\cache"
    OUTPUT_CSV_DIR = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\SFT_Data\temp"
    API_KEY = "ce6fac43-3d59-4dfe-8949-ea1029f42a32"
    model = "ep-20250422003842-v9gn4"

    build_SFT_data(api_key=API_KEY,
                   input_dir=INPUT_CSV_DIR,
                   output_dir=OUTPUT_CSV_DIR,
                   model=model)