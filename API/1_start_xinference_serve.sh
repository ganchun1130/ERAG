#!/bin/bash

# 设置环境变量
export XINFERENCE_HOME=/ganchun/code/xinference/cache

# 启动 xinference-local 服务
xinference-local --host 0.0.0.0 --port 9997
