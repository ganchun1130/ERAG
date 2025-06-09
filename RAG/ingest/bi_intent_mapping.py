# coding:utf-8
# @File  : bi_intent_mapping.py
# @Author: ganchun
# @Date  :  2025/04/18
# @Description: 为知识库文档生成摘要和标签

import json
import os
import time
import csv
import hashlib
from datetime import datetime

from Prompt.prompt_templates import bi_intent_mapping_prompt
from semantic_segment import (use_doubao_api, return_csv_file_list, read_csv_rows)


def process_json(response_text):
    """处理API返回的JSON结果，提取总结和标签"""
    try:
        # 处理可能的错误格式
        response_text = response_text.replace('""', '"')
        # 查找JSON部分（如果API返回包含额外文本）
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start != -1 and end > start:
            json_str = response_text[start:end]
            result = json.loads(json_str)

            # 从API响应中提取总结和标签
            summary = result.get('总结', '')
            tags = result.get('标签', [])

            # 确保标签是列表格式
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except:
                    tags = [tags]

            return summary, tags
        else:
            raise ValueError("无法在响应中找到有效的JSON内容")

    except Exception as e:
        print(f"解析响应失败: {e}")
        print(f"原始响应: {response_text}")
        # 添加空的总结和标签
        return '', []


def list2json(list_data):
    """安全地将列表转换为所需的JSON格式"""
    json_data = {
        '章节标题': '',
        '副标题': '',
        '内容': ''
    }

    # 安全获取列表元素
    if len(list_data) > 0:
        json_data['章节标题'] = list_data[0]

    if len(list_data) > 2:
        json_data['副标题'] = list_data[1]
        json_data['内容'] = list_data[2]
    elif len(list_data) > 1:
        json_data['内容'] = list_data[1]

    return json_data


def get_cache_key(text):
    """生成缓存键值"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def bi_intent_mapping(api_key, input_csv_path, output_csv_path):
    """为知识库文档生成摘要和标签"""
    start_time = datetime.now()
    print(f"开始处理: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 初始化缓存字典
    cache = {}

    input_csv_file_list = return_csv_file_list(input_csv_path)
    print(f"找到 {len(input_csv_file_list)} 个CSV文件待处理")

    # 添加API调用计数器
    api_call_count = 0
    total_processed = 0
    total_rows = 0

    # 首先统计总行数
    for csv_file in input_csv_file_list:
        rows = read_csv_rows(csv_file)
        total_rows += len(rows)

    print(f"总共需要处理 {total_rows} 行数据")

    for file_index, csv_file in enumerate(input_csv_file_list):
        file_name = os.path.basename(csv_file)
        output_file = os.path.join(output_csv_path, file_name)
        print(f"\n[{file_index + 1}/{len(input_csv_file_list)}] 处理文件：{file_name}")

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        rows = read_csv_rows(csv_file)
        print(f"读取到 {len(rows)} 行数据")

        # 创建新表头
        new_header = ['章节标题', '子标题', '内容', '总结', '标签']

        new_rows = []
        for row_index, row in enumerate(rows):
            total_processed += 1
            progress = (total_processed / total_rows) * 100

            # 显示进度
            print(f"进度: {total_processed}/{total_rows} ({progress:.1f}%) - 当前文件进度: {row_index + 1}/{len(rows)}")

            # 确保行数据格式正确
            if not row:
                print("警告: 跳过空行")
                continue

            # 转换为JSON格式
            json_data = list2json(row)
            prompt = bi_intent_mapping_prompt.format(input_text=json.dumps(json_data, ensure_ascii=False))

            # 检查缓存
            cache_key = get_cache_key(prompt)
            if cache_key in cache:
                print("使用缓存结果")
                summary, tags = cache[cache_key]
            else:
                # API调用频率限制：每10次调用暂停2秒
                api_call_count += 1
                if api_call_count % 10 == 0:
                    print("已完成10次API调用，暂停2秒...")
                    time.sleep(2)

                # 添加重试机制
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        print(f"调用API中... (尝试 {retry + 1}/{max_retries})")
                        response = use_doubao_api(api_key, prompt)
                        summary, tags = process_json(response)

                        # 检查结果有效性
                        if summary or tags:
                            # 存入缓存
                            cache[cache_key] = (summary, tags)
                            break
                        else:
                            print("API返回结果为空，重试中...")
                            time.sleep(1)
                    except Exception as e:
                        print(f"API调用失败: {e}")
                        if retry == max_retries - 1:
                            summary, tags = '', []
                        time.sleep(2)

            # 构建新行，安全处理
            new_row = []
            for i in range(min(3, len(row))):
                new_row.append(row[i])

            # 确保行有3个元素
            while len(new_row) < 3:
                new_row.append("")

            # 添加总结和标签
            new_row.extend([summary, tags])
            new_rows.append(new_row)

        # 写入新CSV文件
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(new_header)
            writer.writerows(new_rows)

        print(f"完成文件处理，已写入新文件: {output_file}")

    end_time = datetime.now()
    time_used = end_time - start_time
    print(f"\n处理完成: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {time_used}")
    print(f"API总调用次数: {api_call_count}")
    print(f"缓存命中次数: {total_processed - api_call_count}")


if __name__ == "__main__":
    root_path = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\Knowledge_Docs"
    csv_root = os.path.join(root_path, "processed")
    output_csv_path = os.path.join(root_path, "ERAG")
    api_key = "ce6fac43-3d59-4dfe-8949-ea1029f42a32"

    bi_intent_mapping(
        api_key,
        csv_root,
        output_csv_path
    )