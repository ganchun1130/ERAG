# coding:utf-8
# @File  : start_reranker_api.py
# @Author: ganchun
# @Date  :  2025/04/13
# @Description:

from xinference.client import Client

client = Client("http://localhost:9997")
# model_uid = client.launch_model(
#     model_name="bge-reranker-v2-m3",
#     model_type="rerank",
#     model_path=r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Model\EM\bge-reranker-v2-m3"
# )
# model = client.get_model(model_uid)

model = client.get_model("bge-reranker-v2-m3")
# 以下使用xinference进行验证
query = "A man is eating pasta."
corpus = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin."
]
print(model.rerank(corpus, query))

