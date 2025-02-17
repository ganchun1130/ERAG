from xinference.client import RESTfulClient
from api_request import xinference_client, openai_client

client = RESTfulClient("http://0.0.0.0:9997")

model_uid = client.launch_model(
    model_engine="transformers",
    model_name="qwen2.5-instruct",
    model_path="～/Qwen/Qwen2.5-0.5B-Instruct",
)
print('Model uid: ' + model_uid)
# 以上是启动qwen的API服务

# 以下是测试API
xinference_client(model_uid)
openai_client(model_uid)
