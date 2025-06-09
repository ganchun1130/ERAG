# coding:utf-8
# @File  : use_doubao_api.py
# @Author: ganchun
# @Date  :  2025/04/15
# @Description: 如何调用豆包推理API，下面提供了一个例子
from openai import OpenAI

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
            {"role": "system", "content": "你是人工智能助手"},
            {"role": "user", "content": prompt},
        ],
    )
    result = completion.choices[0].message.content
    print(result)
    return result

def use_doubao_api_custom(api_key, model, prompt):
    # 请确保您已将 API Key 存储在环境变量 ARK_API_KEY 中
    client = OpenAI(
        # 此为默认路径，您可根据业务所在地域进行配置
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=api_key,
    )
    print("----- standard request -----")
    completion = client.chat.completions.create(
        # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
        model=model,
        messages=[
            {"role": "system", "content": "你是人工智能助手"},
            {"role": "user", "content": prompt},
        ],
    )
    result = completion.choices[0].message.content
    # print(result)
    return result


if __name__ == "__main__":
    api_key = "ce6fac43-3d59-4dfe-8949-ea1029f42a32"
    prompt = "请给我讲一个笑话"
    test = use_doubao_api(api_key, prompt)

# ep-20250421154507-59xth   Doubao-1.5-pro-256k
# ep-20250306115053-vzsng   Doubao-1.5-pro-32k
# ep-20250422015143-6nx9l   Doubao-1.5-thinking-pro
# ep-20250422003842-v9gn4   DeepSeek-V3
# ep-20250422013101-xjhks   DeepSeek-R1