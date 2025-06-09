# coding:utf-8
# @File  : 7_build_ms_swift_data.py
# @Author: ganchun
# @Date  :  2025/04/20
# @Description:将SFT数据(CSV)转换为MS-Swift训练数据(JSON)格式

import os
import csv
import json
from tqdm import tqdm


def csv_to_msswift(csv_file, output_dir=None):
    """将单个CSV文件转换为MS-Swift格式JSON"""
    try:
        # 准备输出路径
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(csv_file).replace('.csv', '')
            output_file = os.path.join(output_dir, f"{base_name}.json")
        else:
            output_file = csv_file.replace('.csv', '.json')

        # 读取CSV文件
        with open(csv_file, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            headers = next(csv_reader)  # 获取表头

            # 查找必要列的索引
            try:
                q_type_idx = headers.index('问题类型')
                question_idx = headers.index('问题')
                answer_idx = headers.index('答案')
            except ValueError as e:
                print(f"警告：CSV文件 {csv_file} 中缺少必要列: {e}")
                return None

            # 处理数据并转换为MS-Swift格式
            ms_swift_data = []

            for row in csv_reader:
                if len(row) <= max(q_type_idx, question_idx, answer_idx):
                    continue  # 跳过不完整的行

                question = row[question_idx].strip()
                answer = row[answer_idx].strip()
                q_type = row[q_type_idx].strip() if q_type_idx < len(row) else ""

                if not question or not answer:
                    continue  # 跳过空问题或答案

                # 创建单个对话样本
                system_content = "你是个有用无害的助手"
                if q_type:
                    system_content += f"，专长于解答{q_type}问题"

                conversation = {
                    "messages": [
                        # {
                        #     "role": "system",
                        #     "content": system_content
                        # },
                        {
                            "role": "user",
                            "content": question
                        },
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    ]
                }

                ms_swift_data.append(conversation)

        # 保存为JSON文件
        with open(output_file, 'w', encoding='utf-8') as out_f:
            json.dump(ms_swift_data, out_f, ensure_ascii=False, indent=2)

        print(f"已转换: {csv_file} → {output_file} (共 {len(ms_swift_data)} 条对话数据)")
        return output_file

    except Exception as e:
        print(f"处理文件 {csv_file} 时出错: {e}")
        return None


def process_path(input_path, output_dir=None):
    """处理输入路径（文件或目录）"""
    if os.path.isfile(input_path) and input_path.endswith('.csv'):
        # 处理单个文件
        csv_to_msswift(input_path, output_dir)

    elif os.path.isdir(input_path):
        # 处理目录中的所有CSV文件
        csv_files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                     if f.endswith('.csv') and os.path.isfile(os.path.join(input_path, f))]

        print(f"找到 {len(csv_files)} 个CSV文件")

        for csv_file in tqdm(csv_files, desc="处理CSV文件"):
            csv_to_msswift(csv_file, output_dir)

    else:
        print(f"错误: {input_path} 不是有效的CSV文件或目录")


if __name__ == "__main__":
    # 输入和输出路径配置
    input_path = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\SFT_Data\raw\all_raw.csv"
    output_dir = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\SFT_Data\final"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 处理文件
    process_path(input_path, output_dir)
    print("转换完成!")