# coding:utf-8
# @File  : 10_concat_same_name_csv.py
# @Author: ganchun
# @Date  :  2025/04/21
# @Description: 合并两个目录中同名CSV文件的内容

import os
import pandas as pd
from tqdm import tqdm


def get_filename_map(directory):
    """
    获取目录中所有CSV文件的映射表
    返回：{文件名: 完整路径}
    """
    file_map = {}
    if os.path.exists(directory) and os.path.isdir(directory):
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                file_map[filename] = os.path.join(directory, filename)
    return file_map


def append_csv_files(dir_a, dir_b):
    """
    将目录B中的CSV文件内容追加到目录A中同名文件的末尾
    """
    # 获取两个目录中的文件映射
    files_a = get_filename_map(dir_a)
    files_b = get_filename_map(dir_b)

    # 找出共有的文件名
    common_filenames = set(files_a.keys()) & set(files_b.keys())
    print(f"在两个目录中找到 {len(common_filenames)} 个同名CSV文件")

    if not common_filenames:
        print("没有找到同名文件，无需合并")
        return

    # 处理每个共有文件
    for filename in tqdm(common_filenames, desc="合并文件"):
        file_a_path = files_a[filename]
        file_b_path = files_b[filename]

        try:
            # 读取两个CSV文件
            df_a = pd.read_csv(file_a_path, encoding='utf-8')
            df_b = pd.read_csv(file_b_path, encoding='utf-8')

            # 检查列名是否一致
            if list(df_a.columns) != list(df_b.columns):
                print(f"警告: {filename} 的列名不一致，但仍尝试合并")

            # 合并数据
            df_combined = pd.concat([df_a, df_b], ignore_index=True)

            # 将合并后的数据写回A目录中的文件
            df_combined.to_csv(file_a_path, index=False, encoding='utf-8')
            print(f"已合并: {filename} (A行数:{len(df_a)}, B行数:{len(df_b)}, 合并后:{len(df_combined)})")

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")


if __name__ == "__main__":
    # 配置目录路径
    directory_a = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\SFT_Data\raw"
    directory_b = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\SFT_Data"

    # 执行合并操作
    append_csv_files(directory_a, directory_b)
    print("合并操作完成!")