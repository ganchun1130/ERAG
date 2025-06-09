# coding:utf-8
# @File  : 9_build_QA_data_all.py
# @Author: ganchun
# @Date  :  2025/04/21
# @Description: 根据TXT文件生成SFT问答数据集

import os
import time
import pandas as pd
from Prompt.prompt_templates import build_SFT_data_prompt_2
from use_doubao_api import use_doubao_api_custom
from other_functions import process_json_qa, get_cache_key


def append_to_csv(qa_data, output_file):
    """将数据追加到CSV文件中，如文件不存在则创建"""
    if not qa_data:
        return False

    file_exists = os.path.isfile(output_file)

    df = pd.DataFrame(qa_data)
    df.to_csv(output_file, mode='a', header=not file_exists, index=False, encoding='utf-8')

    return True


def process_single_file(api_key, model, txt_file, output_file, cache=None, api_call_count=0):
    """处理单个TXT文件生成问答数据并保存"""
    file_name = os.path.basename(txt_file)
    cache_hit = False

    try:
        # 读取TXT文件内容
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()

        if not content:
            print(f"警告: {file_name} 内容为空")
            return False, api_call_count, False

        print(f"成功读取文件，内容长度: {len(content)} 字符")

        # 构建提示词
        prompt = build_SFT_data_prompt_2.format(input_text=content)

        # 检查缓存
        qa_list = None
        if cache is not None:
            cache_key = get_cache_key(prompt)
            if cache_key in cache:
                print("使用缓存结果")
                qa_list = cache[cache_key]
                cache_hit = True

        if qa_list is None:
            # API调用
            api_call_count += 1
            if api_call_count % 5 == 0:
                print(f"已完成{api_call_count}次API调用，暂停1秒...")
                time.sleep(1)

            response = use_doubao_api_custom(api_key=api_key, model=model, prompt=prompt)

            # 定义重试回调函数
            def retry_api_call():
                print("重新调用API...")
                time.sleep(2)
                return use_doubao_api_custom(api_key=api_key, model=model, prompt=prompt)

            # 解析返回结果
            qa_list = process_json_qa(
                response_text=response,
                max_retries=3,
                retry_callback=retry_api_call
            )

            # 保存到缓存
            if qa_list and cache is not None:
                cache_key = get_cache_key(prompt)
                cache[cache_key] = qa_list

        # 提取问答对数据
        qa_data = []
        for qa_item in qa_list:
            if isinstance(qa_item, dict) and 'question' in qa_item and 'answer' in qa_item:
                qa_data.append({
                    '问题类型': qa_item.get('difficulty', ''),
                    '问题': qa_item.get('question', ''),
                    '答案': qa_item.get('answer', '')
                })

        # 追加或创建CSV文件
        if append_to_csv(qa_data, output_file):
            print(f"已生成并保存 {len(qa_data)} 条问答对至 {output_file}")
            return True, api_call_count, cache_hit
        else:
            print(f"警告: {file_name} 未生成任何问答对")
            return False, api_call_count, cache_hit

    except Exception as e:
        print(f"处理文件 {txt_file} 时出错: {e}")
        return False, api_call_count, cache_hit


def build_SFT_data(api_key, model, input_dir, output_dir):
    """根据TXT文件构建SFT训练数据"""
    os.makedirs(output_dir, exist_ok=True)

    # 初始化缓存字典和计数器
    cache = {}
    api_call_count = 0
    cache_hit_count = 0
    success_count = 0

    # 获取所有TXT文件列表
    txt_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                if f.endswith('.txt') and os.path.isfile(os.path.join(input_dir, f))]
    print(f"找到 {len(txt_files)} 个TXT文件")

    # 处理每个TXT文件
    for txt_file in txt_files:
        file_name = os.path.basename(txt_file)
        kb_name = os.path.splitext(file_name)[0]
        print(f"\n处理文件: {file_name}")

        # 准备输出文件
        output_file = os.path.join(output_dir, f"{kb_name}_QA.csv")

        # 处理单个文件
        success, api_call_count, is_cached = process_single_file(
            api_key, model, txt_file, output_file, cache, api_call_count
        )

        if success:
            success_count += 1
        if is_cached:
            cache_hit_count += 1

    # 打印统计信息
    print("\n数据构建完成!")
    print(f"成功处理文件数: {success_count}/{len(txt_files)}")
    print(f"API调用次数: {api_call_count}")
    print(f"缓存命中次数: {cache_hit_count}")


def process_path(api_key, model, input_path, output_dir):
    """处理输入路径（文件或目录）"""
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(input_path) and input_path.endswith('.txt'):
        # 处理单个文件
        file_name = os.path.basename(input_path)
        kb_name = os.path.splitext(file_name)[0]
        output_file = os.path.join(output_dir, f"{kb_name}_QA.csv")

        # 调用单文件处理函数
        success, _, _ = process_single_file(api_key, model, input_path, output_file)
        return success

    elif os.path.isdir(input_path):
        # 处理目录
        build_SFT_data(api_key, model, input_path, output_dir)
        return True
    else:
        print(f"错误: {input_path} 不是有效的TXT文件或目录")
        return False


if __name__ == "__main__":
    # 配置参数
    API_KEY = ""
    MODEL = "ep-20250422003842-v9gn4"
    INPUT_PATH = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\Knowledge_Docs\raw"
    OUTPUT_DIR = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\SFT_Data"

    # 处理文件或目录
    process_path(API_KEY, MODEL, INPUT_PATH, OUTPUT_DIR)
    print("转换完成!")
