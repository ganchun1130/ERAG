# coding:utf-8
# @File  : use_deepseek_api.py
# @Author: ganchun
# @Date  :  2025/04/18
# @Description: 
#
# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI


def use_deepseek_api(api_key):
    client = OpenAI(api_key=api_key,
                    base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        # model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "讲一个笑话吧"},
        ],
        stream=False
    )
    result = response.choices[0].message.content
    return result

if __name__ == '__main__':
    api_key = "sk-d40a6d50bdf049e4bbc251af8747db34"
    result = use_deepseek_api(api_key)
    print(result)
