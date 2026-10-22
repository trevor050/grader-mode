#!/usr/bin/env bash
# Optional stronger 2026 candidate, with the author's documented Transformers version.
set -euo pipefail
cd "$HOME/grader-looped"
export UV_CACHE_DIR="$PWD/uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/python"
export UV_PYTHON_BIN_DIR="$PWD/bin"
export HF_HOME="$PWD/hf-cache"
export HF_MODULES_CACHE="$PWD/hf-modules"
export TMPDIR="$PWD/tmp"
bin/uv venv --python 3.12 .venv-nanbeige
bin/uv pip install --python .venv-nanbeige/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
bin/uv pip install --python .venv-nanbeige/bin/python transformers==4.45.1 accelerate==1.10.1 sentencepiece==0.2.1 protobuf==5.29.3
bin/uv pip freeze --python .venv-nanbeige/bin/python > requirements-nanbeige.txt
.venv-nanbeige/bin/python - <<'PY'
from transformers import AutoConfig, AutoTokenizer
name = "Nanbeige/Nanbeige4.2-3B"
revision = "b82e54bd609793562a75cbf9337970a93369eab5"
config = AutoConfig.from_pretrained(name, revision=revision, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=True, use_fast=False)
print("Nanbeige config/tokenizer ready; loops=", config.num_loops, flush=True)
PY
