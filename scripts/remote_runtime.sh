#!/usr/bin/env bash
# Run in a selected Python environment without changing shell startup files.
set -euo pipefail
ERAG_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ERAG_PYTHON_PREFIX="${ERAG_PYTHON_PREFIX:-${CONDA_PREFIX:-$ERAG_PROJECT_ROOT/.conda/envs/erag}}"
if [[ ! -x "$ERAG_PYTHON_PREFIX/bin/python" ]]; then
    echo "Missing environment: $ERAG_PYTHON_PREFIX (see docs/deployment.md)" >&2
    exit 1
fi
export PATH="$ERAG_PYTHON_PREFIX/bin:$PATH"
export CONDA_ENVS_PATH="$ERAG_PROJECT_ROOT/.conda/envs"
export CONDA_PKGS_DIRS="$ERAG_PROJECT_ROOT/.cache/conda"
export PIP_CACHE_DIR="$ERAG_PROJECT_ROOT/.cache/pip"
export UV_CACHE_DIR="$ERAG_PROJECT_ROOT/.cache/uv"
export HF_HOME="$ERAG_PROJECT_ROOT/.cache/huggingface"
export XINFERENCE_HOME="$ERAG_PROJECT_ROOT/.cache/xinference"
export XINFERENCE_ENABLE_VIRTUAL_ENV=0
export XDG_CACHE_HOME="$ERAG_PROJECT_ROOT/.cache"
export TORCH_HOME="$ERAG_PROJECT_ROOT/.cache/torch"
export TRITON_CACHE_DIR="$ERAG_PROJECT_ROOT/.cache/triton"
export GRADIO_TEMP_DIR="$ERAG_PROJECT_ROOT/.cache/gradio"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd "$ERAG_PROJECT_ROOT"
exec "$@"
