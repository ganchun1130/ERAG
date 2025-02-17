#!/bin/bash
# 要么运行这个文件，要么就运行/mnt/general/ganchun/code/XiaoYa/API/start_qwen2_5_api.py
# 启动 xinference 服务
xinference launch --model_path /mnt/general/share/model/Qwen/Qwen2.5-0.5B-Instruct \
                  --model-engine Transformers \
                  -n qwen2.5-instruct