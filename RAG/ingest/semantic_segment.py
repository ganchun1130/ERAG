# coding:utf-8
# @File  : semantic_segment.py
# @Author: ganchun
# @Date  :  2025/04/15
# @Description:
# 输入是一个csv文件，调用LLM的API服务针对较多的内容进行语义分块，输出也是一个csv文件
import csv
import json
import os
import time

from openai import OpenAI

from Prompt.prompt_templates import semantic_segment_prompt

def use_doubao_api(api_key, prompt):
    # 请确保您已将 API Key 存储在环境变量 ARK_API_KEY 中
    client = OpenAI(
        # 此为默认路径，您可根据业务所在地域进行配置
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=api_key,
    )

    # Non-streaming:
    print("----- standard request -----")
    completion = client.chat.completions.create(
        # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
        model="ep-20250306115053-vzsng",
        messages=[
            {"role": "system", "content": "你是十分强大的人工智能助手"},
            {"role": "user", "content": prompt},
        ],
    )
    result = completion.choices[0].message.content
    # print(result)
    return result

def process_json(response_text):
    """处理API返回的JSON字符串，提value值作为列表返回"""
    try:
        # 尝试直接解析JSON
        response_text = response_text.replace('""', '"')
        data = json.loads(response_text)
        # 将字典的值提取为列表
        return list(data.values())
    except json.JSONDecodeError:
        # 如果直接解析失败，尝试在文本中查找JSON部分
        # 查找第一个 { 和最后一个 } 之间的内容
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start != -1 and end != 0:
            try:
                json_str = response_text[start:end]
                data = json.loads(json_str)
                return list(data.values())
            except:
                pass
        # 如果仍然失败，输出错误信息
        print(f"无法解析JSON: {response_text}")
        return [response_text]  # 返回原始文本作为单个块

def read_csv_rows(file_path, encoding='utf-8'):
    rows = []
    try:
        with open(file_path, 'r', encoding=encoding) as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # 直接跳过首行
            # next(csv_reader)  # 直接跳过第二行
            for row in csv_reader:
                rows.append(row)
        return rows
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
        return []
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
        return []

def return_csv_file_list(input_csv_path):
    # 返回目录中所有csv文件的路径
    csv_file_list = []
    for root, dirs, files in os.walk(input_csv_path):
        for file in files:
            if file.endswith('.csv'):
                csv_file_list.append(os.path.join(root, file))
    return csv_file_list


def semantic_segment(api_key, input_csv_path, output_csv_path):

    # 读取CSV文件
    csv_file_list = return_csv_file_list(input_csv_path)
    print("一共有{}份csv文件".format(len(csv_file_list)))

    # 添加API调用计数器
    api_call_count = 0

    for csv_file in csv_file_list:
        file_name = os.path.basename(csv_file)
        output_file = os.path.join(output_csv_path, file_name)
        print("正在处理文件：{}".format(file_name))

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        rows = read_csv_rows(csv_file)
        print("读取到{}行数据".format(len(rows)))

        # 存储新行数据
        new_rows = []

        for row in rows:
            if row[0] == "摘要":
                new_rows.append(row)
            if len(row[-1]) >= 500:
                print("当前行的内容长度为{}，大于500，可进行语义分块".format(len(row[-1])))

                # API调用频率限制：每10次调用暂停2秒
                api_call_count += 1
                if api_call_count % 10 == 0:
                    print("已完成10次API调用，暂停2秒...")
                    time.sleep(2)

                # 调用API进行语义分块
                prompt = semantic_segment_prompt.format(input_text=row[-1])
                print(prompt)
                response = use_doubao_api(api_key, prompt)

                # 处理API返回的JSON数据
                result = process_json(response)

                for segment in result:
                    segment = segment.strip()
                    new_rows.append([row[0], row[1], segment])

            else:
                print("当前行的内容长度为{}，小于500，不进行语义分块".format(len(row[-1])))
                new_rows.append(row)

        # 写入新的CSV文件
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            # 写入标题行
            csv_writer.writerow(['章节标题', '子标题', '内容'])
            # 写入数据行
            csv_writer.writerows(new_rows)

        print(f"完成文件处理，已写入新文件: {output_file}")


if __name__ == "__main__":

    root_path = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\Knowledge_Docs"
    csv_root = os.path.join(root_path, "raw")
    output_csv_path = os.path.join(root_path, "processed")
    api_key = ""

    semantic_segment(
        api_key=api_key,
        input_csv_path=csv_root,
        output_csv_path=output_csv_path
    )
