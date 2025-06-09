# coding:utf-8
# @File  : 3_txt2csv.py
# @Author: ganchun
# @Date  :  2025/04/15
# @Description: 使用OCR模型将PDF文档转成txt，然后将txt转成csv

import csv
import re
import os

def parse_technical_report(text):
    # 定义正则表达式匹配章节标题和子标题
    chapter_pattern = re.compile(r'^第[一二三四五六七八九十]+章\s+.*$|^摘要$|^附录章节总结$')
    subtitle_pattern = re.compile(r'^(?:\d+\.\d+|[A-Za-z]\.\d+)\s+.*$')

    # 解析数据
    data = []
    chapter = None
    subtitle = None
    content = []

    for line in text.split('\n'):
        line = line.strip()

        # 检查是否是章节标题
        if chapter_pattern.match(line):
            # 保存之前的内容
            if chapter is not None and content:
                data.append([chapter, subtitle or "", "\\n".join(content)])

            # 开始新章节
            chapter = line
            subtitle = None
            content = []

        # 检查是否是子标题
        elif subtitle_pattern.match(line):
            # 保存之前的内容
            if chapter is not None and content:
                data.append([chapter, subtitle or "", "\\n".join(content)])

            # 开始新子标题
            subtitle = line
            content = []

        # 普通内容行
        else:
            if line or content:  # 避免开头添加空行
                content.append(line)

    # 保存最后一部分内容
    if chapter is not None and content:
        data.append([chapter, subtitle or "", "\\n".join(content)])

    return data

def create_knowledge_base_csv(input_txt_path, output_csv_path):
    # 读取输入文件
    with open(input_txt_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 解析内容
    parsed_data = parse_technical_report(content)

    # 写入CSV文件
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 写入表头
        csv_writer.writerow(['章节标题', '子标题', '内容'])
        # 写入数据
        csv_writer.writerows(parsed_data)

    print(f"CSV文件已成功创建：{output_csv_path}")

if __name__ == "__main__":
    # 使用示例
    root_path = r'F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\Knowledge_Docs'
    input_file = os.path.join(root_path, 'Qwen_Technical_Report.txt')
    output_file = os.path.join(root_path, 'Qwen_Technical_Report.csv')
    create_knowledge_base_csv(input_file, output_file)#