#!/bin/bash
set -euo pipefail
cd /workspace/pulse
trap 'printf "%s\n" "{\"type\":\"setup_or_runner_failure\"}" > /workspace/pulse/out/setup.error.json' ERR
exec > >(tee -a setup.log) 2>&1
echo "setup start $(date -u +%FT%TZ)"
touch heartbeat
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/workspace/pulse/bin sh; export PATH=/workspace/pulse/bin:$PATH; fi
uv venv -p 3.12 /workspace/pulse/venv
uv pip install --python /workspace/pulse/venv/bin/python 'vllm==0.32.0' 'transformers==5.19.0' huggingface-hub scipy
/workspace/pulse/venv/bin/python - <<'PY'
import vllm,transformers,torch,json
print(json.dumps({'vllm':vllm.__version__,'transformers':transformers.__version__,'torch':torch.__version__}),flush=True)
PY
echo "install complete $(date -u +%FT%TZ)"
touch heartbeat
export HF_HOME=/workspace/pulse/hf
/workspace/pulse/venv/bin/python -u - <<'PY'
from huggingface_hub import snapshot_download
import time
start=time.time()
p=snapshot_download('Qwen/Qwen3.8-27B',revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','*.model','*.tiktoken'],max_workers=8)
from pathlib import Path
import shutil
shutil.copyfile(Path(p)/'generation_config.json','/workspace/pulse/generation_config.json')
print('download complete',p,'seconds',time.time()-start,flush=True)
PY
echo "download finished $(date -u +%FT%TZ)"
touch heartbeat
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
/workspace/pulse/venv/bin/python -u run_round3.py > run.log 2>&1
