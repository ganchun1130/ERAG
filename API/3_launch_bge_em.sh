#!/usr/bin/env bash
set -euo pipefail
python API/launch_local_models.py --skip-llm --embedding-path "${1:?Usage: $0 /path/to/Qwen3-Embedding-0.6B}"
