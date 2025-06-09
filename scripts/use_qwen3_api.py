# coding:utf-8
# @File  : use_qwen3_api.py
# @Author: ganchun
# @Date  :  2025/06/06
# @Description:
from dashscope import api_key
from openai import OpenAI, api_type


def use_qwen3_api(api_key, model, prompt):
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
        # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
        extra_body={"enable_thinking": False},
    )
    response = completion.choices[0].message.content
    print(response)
    return response

if __name__ == '__main__':
    api_key = ""
    model = "qwen3-32b"
    prompt = "大明最后一个皇帝是谁？"
    use_qwen3_api(api_key=api_key, model=model, prompt=prompt)
