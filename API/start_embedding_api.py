# coding:utf-8
# @File  : start_embedding_api.py
# @Author: ganchun
# @Date  :  2025/04/13
# @Description: 

import openai
from xinference.client import RESTfulClient
client = RESTfulClient("http://localhost:9997")

# The bge-small-en-v1.5 is an embedding model, so the `model_type` needs to be specified.
model_uid = client.launch_model(
        model_name="bge-m3",
        model_type="embedding",
        model_path=r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Model\EM\bge-m3"
)

# 以下使用OpenAI的方法进行验证
openai_client = openai.Client(api_key="not empty", base_url="http://localhost:9997/v1")
result = openai_client.embeddings.create(model=model_uid, input=["What is the capital of China?"])
embedding = result.data[0].embedding
print(embedding)