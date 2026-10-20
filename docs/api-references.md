# Verified API references (2026-10-20)

- [Modal vLLM example](https://modal.com/docs/examples/vllm_inference): CUDA 12.9 devel + Python 3.12, `uv_pip_install("vllm==0.21.0")`, persistent HF Volume. This project uses offline Functions rather than the example's deployed HTTP server.
- [Modal Function invocation](https://modal.com/docs/reference/modal.Function): `spawn` returns a FunctionCall; submit up to five models before `get`. Installed SDK 1.6.2 signatures verified for explicit timeout, `single_use_containers=True`, FunctionCall.cancel(terminate_containers=True).
- [Modal Volumes](https://modal.com/docs/reference/modal.Volume): `commit` before pulling with the local `read_file` iterator.
- [vLLM LLM.generate](https://docs.vllm.ai/en/stable/api/vllm/entrypoints/llm/): sequence of SamplingParams paired by position; outputs preserve input order.
- [vLLM SamplingParams](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/): explicit seed per request. Reproducibility is scoped to pinned model/software/hardware/batch configuration, not guaranteed bit-identical across GPU architectures.
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B): `apply_chat_template(..., enable_thinking=False)`. Sampling follows DESIGN.md rather than model-card recommendations.
- All six original dense model IDs and revisions in `src/models.py` were confirmed through the Hugging Face API, as was the GSM8K dataset revision. These original post-trained models support the requested non-thinking switch; no base or later thinking-only variant is substituted.

- [Modal GPU strings](https://modal.com/docs/guide/gpu): the exact 96GB RTX PRO 6000 selector is `RTX-PRO-6000`. User steering selects this for 32B; `config/gpus.json` is the per-model mapping. GPU rate assumptions are the supplied hourly rates, not a billing receipt.

- [Volume concurrency](https://modal.com/docs/guide/volumes): cap concurrent model invocations at five so both shared caches and result Volumes stay within the recommended number of concurrent writers.

- [Qwen3 quickstart](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/quickstart.md): reasoning output can lack an opening `<think>` tag; split on the closing tag and retain the reasoning separately. Phase 1b explicitly sets the chat-template thinking switch per request.
