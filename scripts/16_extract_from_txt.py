import re


def merge_chapters_to_lines(input_file, output_file):
    """
    将文本文件中的每一章内容合并为一行
    1. 识别"第X 章"格式的章节标题
    2. 将每章所有内容合并为一行
    3. 当遇到新的章节标题时，开始新的一章
    """
    # 读取文件内容
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 初始化变量
    chapters = {}  # 用于存储每个章节的内容
    current_chapter = None
    chapter_pattern = re.compile(r'^第(\d+)\s+章')  # 匹配"第X 章"格式

    # 处理每一行
    for line in lines:
        line = line.strip()
        if not line:  # 跳过空行
            continue

        # 检查是否是章节标题
        chapter_match = chapter_pattern.search(line)
        if chapter_match:
            chapter_num = chapter_match.group(1)
            current_chapter = f"第{chapter_num} 章"

            # 创建新章节
            if current_chapter not in chapters:
                chapters[current_chapter] = line
            continue

        # 将内容添加到当前章节
        if current_chapter:
            chapters[current_chapter] += line

    # 将处理后的内容写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for chapter, content in sorted(chapters.items(), key=lambda x: int(re.search(r'第(\d+)', x[0]).group(1))):
            f.write(f"{content}\n")

    return len(chapters)  # 返回处理的章节数


if __name__ == "__main__":
    # 文件路径
    input_file = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\Knowledge_Docs\final\result.txt"
    output_file = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\Knowledge_Docs\final\chapter_merged.txt"

    # 处理文件
    chapters_count = merge_chapters_to_lines(input_file, output_file)
    print(f"处理完成，共处理{chapters_count}个章节，结果已保存到 {output_file}")