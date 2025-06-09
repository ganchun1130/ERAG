# coding:utf-8
# @File  : other_functions.py
# @Author: ganchun
# @Date  :  2025/04/18
# @Description:

import hashlib
import json
import os
import pandas as pd


# 1.如果在Windows环境下，首先运行 ERAG\API\start_xinference.py
# 2.然后运行 ERAG\API\start_qwen2_5_api.py，这个文件不仅能启动qwen的API服务，还可进行api测试

# 1.如果在Linux环境下，首先运行bash文件 ERAG\API\1_start_xinference_serve.sh
# 2.启动xinference服务后，开始launch model，运行 ERAG\API\2_launch_qwen_model.sh
# 3.最后运行 ERAG\API\api_request.py，检验是否可以成功调用API

def process_json_qa(response_text, max_retries=3, retry_callback=None):
    """
    处理API返回的JSON结果，提取问答对

    Args:
        response_text: API返回的响应文本
        max_retries: 最大重试次数
        retry_callback: 重试回调函数

    Returns:
        list: 解析后的问答对列表
    """
    retry_count = 0

    while retry_count <= max_retries:
        try:
            # 处理可能的错误格式
            response_text = response_text.replace('""', '"').strip()

            # 尝试直接解析整个响应
            try:
                data = json.loads(response_text)
                if isinstance(data, list):
                    return data  # 已经是符合格式的问答列表
                if isinstance(data, dict) and 'question' in data and 'answer' in data:
                    return [data]  # 单个问答对象，包装为列表
            except json.JSONDecodeError:
                pass

            # 查找JSON数组部分
            if '[' in response_text and ']' in response_text:
                start = response_text.find('[')
                end = response_text.rfind(']') + 1
                if start != -1 and end > start:
                    json_str = response_text[start:end]
                    try:
                        data = json.loads(json_str)
                        if isinstance(data, list):
                            return data
                    except:
                        pass

            # 查找JSON对象部分
            if '{' in response_text and '}' in response_text:
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start != -1 and end > start:
                    json_str = response_text[start:end]
                    try:
                        data = json.loads(json_str)
                        # 如果是单个问答对象
                        if isinstance(data, dict) and 'question' in data and 'answer' in data:
                            return [data]
                    except:
                        pass

            # 所有解析尝试失败，如果有回调函数则重试
            if retry_callback and retry_count < max_retries:
                print(f"JSON解析失败，尝试重新请求 (尝试 {retry_count + 1}/{max_retries})")
                retry_count += 1
                response_text = retry_callback()  # 获取新的响应
                continue
            else:
                raise ValueError("无法解析为有效的问答格式")

        except Exception as e:
            if retry_callback and retry_count < max_retries:
                print(f"处理失败({e})，尝试重新请求 (尝试 {retry_count + 1}/{max_retries})")
                retry_count += 1
                response_text = retry_callback()  # 获取新的响应
            else:
                print(f"JSON解析最终失败: {e}")
                print(f"原始响应: {response_text[:200]}...")
                return []  # 所有尝试都失败，返回空列表

    return []  # 超过最大重试次数

def get_cache_key(text):
    """生成缓存键值"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def append_to_csv(qa_data, output_file):
    """将数据追加到CSV文件中，如文件不存在则创建"""
    if not qa_data:
        return False

    file_exists = os.path.isfile(output_file)

    df = pd.DataFrame(qa_data)
    df.to_csv(output_file, mode='a', header=not file_exists, index=False, encoding='utf-8')

    return True