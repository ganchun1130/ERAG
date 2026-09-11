#!/usr/bin/env bash
set -euo pipefail
python API/launch_local_models.py --model-path "${1:?Usage: $0 /path/to/Qwen3-Instruct}"
