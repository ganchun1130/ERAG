# coding:utf-8
# @File  : model_config.py
# @Author: ganchun
# @Date  :  2025/04/25
# @Description:

from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from config import DOUBAO_MODEL_MAPPING
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_core.prompt_values import StringPromptValue,ChatPromptValue

l_llm_configs = {
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "model": "ep-20250306115053-vzsng",
    "api_key":"ce6fac43-3d59-4dfe-8949-ea1029f42a32"
}

db_llm_config = {
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "model": DOUBAO_MODEL_MAPPING["DeepSeek-V3"],
    "api_key":"ce6fac43-3d59-4dfe-8949-ea1029f42a32"
}

em_config = {
    "base_url": "http://localhost:9997/v1",
    "model_name": "bge-m3",
    "dimensions":4096,
    "api_key":"not empty"
}

db_llm = ChatOpenAI(
    base_url=db_llm_config["base_url"],
    api_key=db_llm_config["api_key"],
    model=db_llm_config["model"],
)

local_em = OpenAIEmbeddings(
    base_url=em_config["base_url"],
    api_key=em_config["api_key"],
    model=em_config["model_name"],
    dimensions=em_config["dimensions"]
)

local_rerank_em = OpenAIEmbeddings(
    base_url=em_config["base_url"],
    api_key=em_config["api_key"],
    model=em_config["model_name"],

)

# 测试LLM：使用langchain调用api
# response = db_llm.invoke("你是谁")
# print(response)  # 测试成功

# 测试LLM：使用ragas的LangchainLLMWrapper调用api，通过字符串调用
# prompt = StringPromptValue(text = "你是谁")
# llm = LangchainLLMWrapper(db_llm)
# result = llm.generate_text(prompt=prompt)
# print(result)  # 测试成功

# 测试LLM：使用ragas的LangchainLLMWrapper调用api，通过构建会话调用
# template = ChatPromptTemplate.from_messages([
#     ("human", "你是谁")
# ])
# messages = template.format_messages()
# llm = LangchainLLMWrapper(db_llm)
# result = llm.generate_text(prompt=ChatPromptValue(messages=messages))
# print(result)  # 测试成功

# 测试EM：使用ragas的LangchainEmbeddingsWrapper调用api，通过字符串调用
# em = LangchainEmbeddingsWrapper(local_em)
# result = em.embed_query(text = "你是谁")
# print(result)  # 测试成功


