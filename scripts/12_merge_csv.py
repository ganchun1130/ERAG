# coding:utf-8
# @File  : 12_merge_csv.py
# @Author: ganchun
# @Date  :  2025/04/22
# @Description:合并任意两个CSV文件，将B文件的内容追加到A文件的末尾

import pandas as pd
import os

def merge_csv_files(file_a, file_b):
    """
    将文件B的内容追加到文件A的末尾
    :param file_a: 文件A的路径
    :param file_b: 文件B的路径
    """
    # 读取两个CSV文件
    df_a = pd.read_csv(file_a, encoding='utf-8')
    df_b = pd.read_csv(file_b, encoding='utf-8')

    # 检查列名是否一致
    if list(df_a.columns) != list(df_b.columns):
        print(f"警告: {file_a} 和 {file_b} 的列名不一致，但仍尝试合并")

    # 合并数据
    df_combined = pd.concat([df_a, df_b], ignore_index=True)

    # 将合并后的数据写回A文件
    df_combined.to_csv(file_a, index=False, encoding='utf-8')
    print(f"已合并: {file_a} 和 {file_b} (A行数:{len(df_a)}, B行数:{len(df_b)}, 合并后:{len(df_combined)})")

if __name__ == "__main__":
    root_path = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\SFT_Data"
    file_a = os.path.join(root_path, "raw", "all_raw.csv")  # 替换为文件A的路径
    file_b_names = ["Qwen2_5_1M_Technical_Report_QA.csv",
                    "Qwen2_5_Technical_Report_QA.csv",
                    "Qwen2_5_Coder_Technical_Report_QA.csv",
                    "Qwen2_5_Math_Technical_Report_QA.csv",
                    "Qwen2_Technical_Report_QA.csv",
                    "Qwen_Technical_Report_QA.csv"]  # 替换为文件B的路径

    for file_b_name in file_b_names:
        file_b = os.path.join(root_path, "temp", file_b_name)
        merge_csv_files(file_a, file_b)

    for file_b_name in file_b_names:
        file_b = os.path.join(root_path, file_b_name)
        merge_csv_files(file_a, file_b)