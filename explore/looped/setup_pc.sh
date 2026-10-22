#!/usr/bin/env bash
# Run only inside the task-owned, newly created ~/grader-looped directory.
set -euo pipefail
cd "$HOME/grader-looped"
export HF_HOME="$PWD/hf-cache"
export UV_CACHE_DIR="$PWD/uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/python"
export UV_PYTHON_BIN_DIR="$PWD/bin"
export TMPDIR="$PWD/tmp"
mkdir -p "$TMPDIR" bin
curl -fLsS https://github.com/astral-sh/uv/releases/download/0.9.4/uv-x86_64-unknown-linux-gnu.tar.gz -o uv.tar.gz
tar xzf uv.tar.gz --strip-components=1 -C bin uv-x86_64-unknown-linux-gnu/uv
bin/uv python install 3.12
bin/uv venv --clear --python 3.12 .venv
bin/uv pip install --python .venv/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
bin/uv pip install --python .venv/bin/python transformers==4.57.6 accelerate==1.10.1 sentencepiece==0.2.1
.venv/bin/python -c 'import torch,transformers; print(torch.__version__,transformers.__version__,torch.cuda.is_available())'
bin/uv pip freeze --python .venv/bin/python > requirements-resolved.txt
