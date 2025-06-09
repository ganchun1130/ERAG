# xinference：本地部署 LLM 和 embedding model 的框架 🚀

xinference 是一个用于在本地部署 LLM（大语言模型）和 embedding model（嵌入模型）的框架。通过它，你可以便捷地在本地环境中利用强大的语言模型和嵌入模型能力。点击查看使用文档：[https://inference.readthedocs.io/zh-cn/latest/getting\_started/using\_xinference.html#run-xinference-l](https://inference.readthedocs.io/zh-cn/latest/getting_started/using_xinference.html#run-xinference-locally)[o](https://inference.readthedocs.io/zh-cn/latest/getting_started/using_xinference.html#run-xinference-locally)[c](https://inference.readthedocs.io/zh-cn/latest/getting_started/using_xinference.html#run-xinference-locally)[a](https://inference.readthedocs.io/zh-cn/latest/getting_started/using_xinference.html#run-xinference-locally)[l](https://inference.readthedocs.io/zh-cn/latest/getting_started/using_xinference.html#run-xinference-locally)[l](https://inference.readthedocs.io/zh-cn/latest/getting_started/using_xinference.html#run-xinference-locally)[y](https://inference.readthedocs.io/zh-cn/latest/getting_started/using_xinference.html#run-xinference-locally) 🔗

📣请注意，以下命令均在Linux环境下执行，如果是Windows环境，则按顺序运行以下文件，即可启动xinference服务和LLM以及EM：

1. `start_xinference.py`：使用代码启动xinference服务。
2. `start_qwen2_5_api.py`：启动qwen2.5模型。
3. `start_embedding_api.py` 
4. `start_reranker_api.py` 


## 安装 🛠️

使用以下命令进行安装：



```Python
pip install "xinference[all]"

# 有点慢，因为要安装很多东西
```

这里安装的 `xinference[all]` 包含了运行时所需的所有相关依赖，由于依赖较多，安装过程可能会耗费一些时间，请耐心等待 ⏳

## 启动服务 🏃‍♂️

### 启动网页服务 🌐

指定缓存路径并启动网页服务，然后打开网页：[http:](http://10.119.20.233:9997/ui)[//10.](http://10.119.20.233:9997/ui)[119.2](http://10.119.20.233:9997/ui)[0.233](http://10.119.20.233:9997/ui)[:9997](http://10.119.20.233:9997/ui)[/ui](http://10.119.20.233:9997/ui)



```
XINFERENCE_HOME=/mnt/general/ganchun/code/xinference/cache xinference-local --host 0.0.0.0 --port 9997
```

上述命令中，`XINFERENCE_HOME` 指定了缓存路径，`xinference-local` 为启动命令，`--host` 和 `--port` 分别指定了服务监听的主机和端口。启动成功后，在浏览器中访问上述链接，即可打开服务的用户界面。

### 启动 qwen 的 API 服务 📡

使用命令行或者 python 代码启动 qwen 的 API 服务。

注意：如果需要两个相同模型，则需要设置额外参数，具体参考：[h](https://github.com/xorbitsai/inference/issues/2773)[ttps:](https://github.com/xorbitsai/inference/issues/2773)[//git](https://github.com/xorbitsai/inference/issues/2773)[hub.c](https://github.com/xorbitsai/inference/issues/2773)[om/xo](https://github.com/xorbitsai/inference/issues/2773)[rbits](https://github.com/xorbitsai/inference/issues/2773)[ai/in](https://github.com/xorbitsai/inference/issues/2773)[feren](https://github.com/xorbitsai/inference/issues/2773)[ce/is](https://github.com/xorbitsai/inference/issues/2773)[sues/](https://github.com/xorbitsai/inference/issues/2773)[2773](https://github.com/xorbitsai/inference/issues/2773)



```
xinference launch --model_path /mnt/general/ganchun/model/Qwen2.5-0.5B-Instruct --model-engine Transformers -n qwen2.5-instruct
```

该命令用于启动 qwen 模型的 API 服务，`--model_path` 指定模型文件所在路径，`--model-engine` 指定模型引擎，`-n` 为模型实例命名。通过此命令，你可以将 qwen 模型部署为可供调用的 API 服务。

### 在 python 代码中验证 API 服务 🐍



```
python3 api_request.py
```

运行上述命令，即可使用 `api_request.py` 脚本来验证刚刚启动的 qwen API 服务是否正常工作。脚本会向 API 发送请求并处理返回结果，以判断 API 服务是否运行良好。

## 使用纯代码 launch model 💻

同样需要先使用命令行启动 xinference 的服务：

### 启动网页服务 🌐

指定缓存路径并启动网页服务，然后打开网页：[http://10.119.20.233:9997/ui](http://10.119.20.233:9997/ui)



```bash
XINFERENCE_HOME=/mnt/general/ganchun/code/xinference/cache xinference-local --host 0.0.0.0 --port 9997
```

这一步与前面启动网页服务一致，是后续使用纯代码启动模型的前置条件，确保服务端正常运行。

### 使用 Python 代码运行 🐍



```bash
python api_request.py
```

在启动服务后，运行此命令可通过 `api_request.py` 脚本以纯代码的方式启动xinference。该脚本会与之前启动的 xinference 服务进行交互，完成模型的加载和启动过程。

**注意**：无论采用何种方式，都需要先启动 xinference 的服务，再进行其他操作。这是确保整个流程正确运行的关键，若未先启动服务，后续操作可能会因无法连接到服务端而失败。