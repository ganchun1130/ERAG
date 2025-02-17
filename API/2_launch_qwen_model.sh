#!/bin/bash
# 要么运行这个文件，要么就运行～/XiaoYa/API/start_qwen2_5_api.py
# 启动 xinference 服务
xinference launch --model_path ～/Qwen/Qwen2.5-0.5B-Instruct \
                  --model-engine Transformers \
                  -n qwen2.5-instruct
