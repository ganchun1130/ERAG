# coding:utf-8
# @File  : model_config.py
# @Author: ganchun
# @Date  :  2025/04/25
# @Description: 参数配置

# doubao api
DOUBAO_API_URL = "https://ark.cn-beijing.volces.com/api/v3"
DOUBAO_MODEL_MAPPING = {
    "Doubao-1.5-pro-256k": "ep-20250421154507-59xth",
    "Doubao-1.5-pro-32k": "ep-20250306115053-vzsng",
    "Doubao-1.5-thinking-pro": "ep-20250422015143-6nx9l",
    "DeepSeek-V3": "ep-20250422003842-v9gn4",
    "DeepSeek-R1": "ep-20250422013101-xjhks"
}
DOUBAO_MODEL = DOUBAO_MODEL_MAPPING["Doubao-1.5-pro-32k"]  # 请替换为您选择的模型
DOUBAO_API_KEY = "ce6fac43-3d59-4dfe-8949-ea1029f42a32"  # 请替换为您的API密钥
DOUBAO_BOT_ID = "bot-20250427183733-wzfdm"
DOUBAO_AK = "AKLTZmM5MGNkZmYzM2IzNGI4MmIyODRhNTJmOTA4MThiYmY"
DOUBAO_SK = "WWpFeFpEQmpPRE0yTVRobE5EQTBPV0V5WmpJell6RXhPVEJtWVRaa01XUQ=="
ACCOUNT_ID = "2101057926"

# local api
LLM_API_URL = "http://localhost:9997/v1"
LLM_MODEL_UID = "qwen2.5-instruct"  # 查询重写使用的大模型
EMBEDDING_API_URL = "http://localhost:9997/v1"
EMBEDDING_MODEL_UID = "bge-m3"  # 嵌入向量模型
EMBEDDING_DIMENSION = 1024  # bge-m3的维度
RERANKER_API_URL = "http://localhost:9997"
RERANKER_MODEL_UID = "bge-reranker-v2-m3"

# retrieval
KNOWLEDGE_BASE_ROOT = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\Knowledge_Base"
SUB_QUERY_TOP_K = 5  # 子查询返回的结果数量
SUB_QUERY_TOP_P = 0.85  # 子查询保留的得分比例
KEY_QUERY_TOP_K = 5  # 关键查询返回的结果数量
KEY_QUERY_TOP_P = 0.8  # 关键查询保留的得分比例
FINAL_DOCS_TOP_K = 15 # 最终返回的结果数量

# fine-tune
MODEL_PATH = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Model\Qwen2.5-Chat"
DATA_PATH = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\Fine_Tune_Data"
LEARNING_RATE = 1e-5
BATCH_SIZE = 1
EPOCHS = 1

# chat history
CHAT_HISTORY_DIR = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\Conversation"
MAX_HISTORY_TURNS = 5  # 最大对话历史轮次
