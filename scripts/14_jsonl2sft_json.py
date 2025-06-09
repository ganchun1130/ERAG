# coding:utf-8
# @File  : 14_jsonl2sft_json.py
# @Author: ganchun
# @Date  :  2025/04/23
# @Description:

import os
import json
import argparse
from pathlib import Path


def jsonl_to_sft_json(input_file, output_file):
    """
    将JSONL文件转换为适用于微调的JSON格式

    Args:
        input_file (str): 输入JSONL文件路径
        output_file (str): 输出JSON文件路径
    """
    sft_data = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)

                # 提取必要字段
                instruction = data.get('instruction', '')
                input_text = data.get('input', '')
                output = data.get('output', '')

                # 构建用户内容 - 合并instruction和input
                if input_text:
                    user_content = f"{instruction}\n\n{input_text}"
                else:
                    user_content = instruction

                # 创建会话结构
                conversation = {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": output}
                    ]
                }

                sft_data.append(conversation)
            except json.JSONDecodeError:
                print(f"警告: 无法解析行: {line}")
                continue

    # 写入JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    return len(sft_data)


def process_directory(input_dir, output_dir=None):
    """
    处理目录下所有的JSONL文件，并转换为JSON格式

    Args:
        input_dir (str): 输入目录路径
        output_dir (str, optional): 输出目录路径，如不指定则使用输入目录
    """
    # 如果未指定输出目录，则使用输入目录
    if output_dir is None:
        output_dir = input_dir

    # 创建输出目录（如果不存在）
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 获取所有的JSONL文件
    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]

    if not jsonl_files:
        print(f"警告: 在 {input_dir} 中未找到JSONL文件")
        return

    total_conversations = 0

    # 处理每个JSONL文件
    for jsonl_file in jsonl_files:
        input_path = os.path.join(input_dir, jsonl_file)
        output_path = os.path.join(output_dir, jsonl_file.replace('.jsonl', '.json'))

        print(f"处理文件: {input_path}")
        conversations_count = jsonl_to_sft_json(input_path, output_path)
        total_conversations += conversations_count
        print(f"已生成 {conversations_count} 条对话，保存至: {output_path}")

    print(f"转换完成! 总共生成 {total_conversations} 条对话")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将JSONL文件转换为JSON文件')
    parser.add_argument('-i',
                        '--input_dir',
                        default=r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\SFT_Data\raw",
                        help='包含JSONL文件的目录路径')
    parser.add_argument('-o',
                        '--output_dir',
                        default=r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\SFT_Data\final",
                        help='输出JSON文件的目录路径（默认与输入目录相同）')

    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir)