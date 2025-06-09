# coding:utf-8
# @File  : 6_csv2txt.py
# @Author: ganchun
# @Date  :  2025/04/20
# @Description: 将 CSV 文件转换为 TXT 格式文本

import os
import csv
from tqdm import tqdm

def csv_to_txt(csv_file, output_dir=None):
    """将单个CSV文件转换为TXT格式"""
    try:
        # 准备输出路径
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(csv_file).replace('.csv', '')
            output_file = os.path.join(output_dir, f"{base_name}.txt")
        else:
            output_file = csv_file.replace('.csv', '.txt')

        # 读取CSV文件
        with open(csv_file, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            headers = next(csv_reader)  # 获取表头

            # 查找必要列的索引
            title_idx = headers.index('章节标题') if '章节标题' in headers else None
            subtitle_idx = headers.index('子标题') if '子标题' in headers else None
            content_idx = headers.index('内容') if '内容' in headers else None

            if content_idx is None:
                print(f"警告：CSV文件 {csv_file} 中缺少'内容'列")
                return None

            # 跟踪已写入的章节标题
            last_written_title = None

            # 按顺序处理每一行并写入TXT
            with open(output_file, 'w', encoding='utf-8') as out_f:
                for row in csv_reader:
                    # 获取各字段的值
                    chapter_title = row[title_idx] if title_idx is not None and title_idx < len(row) else ''
                    subtitle = row[subtitle_idx] if subtitle_idx is not None and subtitle_idx < len(row) else ''
                    content = row[content_idx] if content_idx is not None and content_idx < len(row) else ''

                    # 只有当章节标题不同于上一个已写入的标题时才写入
                    if chapter_title and chapter_title != last_written_title:
                        out_f.write(f"{chapter_title}\n")
                        last_written_title = chapter_title

                    # 写入子标题（如果存在且与章节标题不同）
                    if subtitle and subtitle != chapter_title:
                        out_f.write(f"{subtitle}\n")

                    # 写入内容（如果存在）
                    if content:
                        out_f.write(f"{content}\n")

        print(f"已转换: {csv_file} → {output_file}")
        return output_file

    except Exception as e:
        print(f"处理文件 {csv_file} 时出错: {e}")
        return None


def process_path(input_path, output_dir=None):
    """处理输入路径（文件或目录）"""
    if os.path.isfile(input_path) and input_path.endswith('.csv'):
        # 处理单个文件
        csv_to_txt(input_path, output_dir)

    elif os.path.isdir(input_path):
        # 处理目录中的所有CSV文件
        csv_files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                    if f.endswith('.csv') and os.path.isfile(os.path.join(input_path, f))]

        print(f"找到 {len(csv_files)} 个CSV文件")

        for csv_file in tqdm(csv_files, desc="处理CSV文件"):
            csv_to_txt(csv_file, output_dir)

    else:
        print(f"错误: {input_path} 不是有效的CSV文件或目录")


if __name__ == "__main__":
    csv_root_path = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\Knowledge_Docs\raw"
    output_dir = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\Knowledge_Docs\raw"

    process_path(csv_root_path, output_dir)
    print("转换完成!")