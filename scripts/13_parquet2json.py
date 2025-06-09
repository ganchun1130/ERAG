# coding:utf-8
# @File  : 13_parquet2json.py
# @Author: ganchun
# @Date  :  2025/04/22
# @Description:
import os
import json
import pandas as pd
import glob
from tqdm import tqdm
import argparse
from pathlib import Path

import numpy as np


def convert_numpy_to_python(obj):
    """递归地将NumPy数据类型转换为Python原生类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.number):
        return obj.item()
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    else:
        return obj


def parquet_to_json(input_dir, output_dir):
    """
    将目录中的所有parquet文件中的conversations列转换为JSON格式

    参数:
    input_dir (str): 包含parquet文件的输入目录
    output_dir (str): 保存JSON文件的输出目录
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 获取目录中的所有parquet文件
    parquet_files = glob.glob(os.path.join(input_dir, "*.parquet"))

    print(f"找到 {len(parquet_files)} 个parquet文件")

    # 处理每个parquet文件
    for parquet_file in tqdm(parquet_files, desc="处理文件"):
        # 提取文件名（不带扩展名）
        base_name = os.path.basename(parquet_file)
        file_name = os.path.splitext(base_name)[0]

        try:
            # 读取parquet文件
            df = pd.read_parquet(parquet_file)

            # 检查是否存在conversations列
            if 'conversations' not in df.columns:
                print(f"警告: {base_name} 中没有找到'conversations'列，跳过该文件")
                continue

            # 提取conversations列数据
            conversations_data = df['conversations'].tolist()

            # 转换NumPy类型为Python原生类型
            conversations_data = convert_numpy_to_python(conversations_data)

            # 保存为JSON文件
            output_file = os.path.join(output_dir, f"{file_name}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(conversations_data, f, ensure_ascii=False, indent=4)

            print(f"已将 {base_name} 转换为 {file_name}.json")

        except Exception as e:
            print(f"处理 {base_name} 时出错: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="将parquet文件中的conversations列转换为JSON格式")
    parser.add_argument("--input_dir",
                        type=str,
                        help="包含parquet文件的输入目录",
                        default=r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\SFT_Data\raw")
    parser.add_argument("--output_dir",
                        type=str,
                        help="保存JSON文件的输出目录",
                        default=r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\SFT_Data\final")

    args = parser.parse_args()

    parquet_to_json(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()