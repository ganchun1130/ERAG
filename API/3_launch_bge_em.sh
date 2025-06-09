#!/bin/bash
# 要么运行这个文件，要么就运行F:\StrivingRendersMeCozy\DeepLearning\ERAG\API\start_embedding_api.py
# 启动 xinference 服务
xinference launch --model_path path \
                  --model-name bce-embedding-base_v1 \
                  --model-type embedding