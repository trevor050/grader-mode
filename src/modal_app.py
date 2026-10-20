"""Ephemeral offline vLLM experiment, invoked with modal run -m src.modal_app."""
import json
from pathlib import Path
import modal
from src.models import MODELS, gpu_for, select_models, load_gpu_config

app = modal.App('grader-mode-phase1')
hf_cache = modal.Volume.from_name('grader-mode-hf-cache', create_if_missing=True)
raw_volume = modal.Volume.from_name('grader-mode-results', create_if_missing=True)
image = (
    modal.Image.from_registry('nvidia/cuda:12.9.0-devel-ubuntu22.04', add_python='3.12')
    .entrypoint([])
    .uv_pip_install('vllm==0.21.0')
    .env({'HF_HOME': '/hf', 'HF_XET_HIGH_PERFORMANCE': '1',
          'DO_NOT_TRACK': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn'})
    .add_local_python_source('src')
)


def package_versions():
    import importlib.metadata
    versions = {}
    for package in ('vllm', 'torch', 'transformers', 'modal'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            # Modal injects its runtime source without a dist-info installation.
            versions[package] = getattr(modal, '__version__', 'runtime-injected') if package == 'modal' else 'unavailable'
    return versions


def render_prompt(tokenizer, request):
    """Use native template, removing auto-injected system only for OLMo ceiling."""
    olmo = request.get('phase', '').startswith('phase1c')
    kwargs = {} if olmo else {'enable_thinking': request.get('thinking', False)}
    prompt = tokenizer.apply_chat_template(request['messages'], tokenize=False,
                                           add_generation_prompt=True, **kwargs)
    if olmo and request['arm'] == 'ceiling':
        if any(m['role'] == 'system' for m in request['messages']):
            raise ValueError('Ceiling requests must not contain a system message')
        prefix = '<|im_start|>system\n'
        if prompt.startswith(prefix):
            prompt = prompt.split('<|im_end|>\n', 1)[1]
        if not prompt.startswith('<|im_start|>user\n'):
            raise ValueError('Unexpected OLMo ceiling template')
    return prompt


def generation_limits(model, model_config, thinking):
    from src.models_1c import MODELS_1C
    max_tokens = model_config.get('max_tokens_think', 3072) if thinking else 768
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError('max_tokens_think must be a positive integer')
    context = (max(8192, max_tokens + 4096) if model in MODELS_1C
               else (4096 if thinking else 2048))
    return max_tokens, context


def generate(model, requests, run_id, manifest, model_config):
    """Shared implementation; only the two GPU-decorated wrappers execute remotely."""
    import hashlib
    import random
    import time
    started = time.monotonic()
    slug = model.split('/')[-1]
    directory = Path('/raw') / run_id
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / f'{slug}.jsonl'
    if raw_path.exists():
        raise FileExistsError(f'Refusing to overwrite {raw_path}')
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    from src.models_1c import MODELS_1C
    registry = {**MODELS, **{m: s['revision'] for m, s in MODELS_1C.items()}}
    if model not in registry or not requests:
        raise ValueError('Expected a registered model and nonempty requests')
    revision = registry[model]
    snapshot_download(model, revision=revision, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    hf_cache.commit()
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    thinking_tokens, max_model_len = generation_limits(model, model_config, any(r.get('thinking', False) for r in requests))
    # Explicitly bypass model generation_config defaults (notably Qwen top-k).
    llm = LLM(model=model, revision=revision, tokenizer_revision=revision,
              dtype='bfloat16', quantization=None, tensor_parallel_size=1,
              max_model_len=max_model_len, max_num_seqs=model_config['batch_size'], gpu_memory_utilization=0.90,
              generation_config='vllm', seed=0)
    loaded = time.monotonic()
    ordered = list(requests)
    random.Random(0).shuffle(ordered)
    batch_size = model_config['batch_size']
    # vLLM batches continuously within a generate() call; chunks exist only for progress logs and crash safety.
    chunk_size = batch_size * 8
    total_tokens = 0
    generation_seconds = 0.0
    prompt_tokens = 0
    completed = 0
    by_thinking = {str(t).lower(): {'requests': 0, 'output_tokens': 0, 'length_finishes': 0} for t in (False, True)}
    with raw_path.open('x') as out:
        for offset in range(0, len(ordered), chunk_size):
            batch = ordered[offset:offset + chunk_size]
            prompts = [render_prompt(tokenizer, r) for r in batch]
            if any(len(tokenizer.encode(p, add_special_tokens=False)) + (thinking_tokens if r.get('thinking', False) else 768)
                   > max_model_len for p, r in zip(prompts, batch)):
                raise ValueError('Prompt plus max_tokens exceeds pinned context length')
            params = [SamplingParams(temperature=0.7, top_p=0.95, top_k=-1,
                      max_tokens=thinking_tokens if r.get('thinking', False) else 768, seed=r['seed']) for r in batch]
            tick = time.monotonic()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            generation_seconds += time.monotonic() - tick
            if len(outputs) != len(batch):
                raise RuntimeError('vLLM returned an incomplete batch')
            for request, output, prompt in zip(batch, outputs, prompts):
                completion = output.outputs[0]
                n = len(completion.token_ids)
                total_tokens += n
                prompt_tokens += len(output.prompt_token_ids)
                group = by_thinking[str(request.get('thinking', False)).lower()]
                group['requests'] += 1
                group['output_tokens'] += n
                group['length_finishes'] += int(completion.finish_reason == 'length')
                row = {**request, 'model': model, 'model_revision': revision, 'run_id': run_id,
                       'response': completion.text, 'output_tokens': n,
                       'prompt_tokens': len(output.prompt_token_ids), 'finish_reason': completion.finish_reason,
                       'rendered_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
            out.flush()
            completed += len(batch)
            elapsed = time.monotonic() - loaded
            eta = elapsed / completed * (len(ordered) - completed)
            if model_config.get('progress') is not None:
                model_config['progress'][model] = {
                    'completed':completed, 'requests':len(ordered), 'output_tokens':total_tokens,
                    'generation_seconds':generation_seconds, 'setup_seconds':loaded-started}
            print(f'{model}: {completed}/{len(ordered)} elapsed={elapsed:.1f}s ETA={eta:.1f}s '
                  f'output_tokens/s={total_tokens/generation_seconds:.1f}', flush=True)
    # Preserve completed generations before any optional metadata/reporting work.
    raw_volume.commit()
    elapsed = time.monotonic() - started
    stats = {'model': model, 'model_revision': revision, 'gpu': model_config['gpu'],
             'requests': completed, 'output_tokens': total_tokens, 'prompt_tokens': prompt_tokens,
             'generation_seconds': generation_seconds, 'setup_seconds': loaded - started,
             'function_seconds': elapsed, 'output_tokens_per_second': total_tokens / generation_seconds,
             'requests_per_second': completed / generation_seconds,
             'by_thinking': by_thinking,
             'sampling': {'temperature': 0.7, 'top_p': 0.95, 'top_k': -1, 'max_tokens': 768,
                          'enable_thinking': False, 'dtype': 'bfloat16', 'quantization': None,
                          'thinking_on': {'enable_thinking': True, 'max_tokens': 3072}},
             'engine': {'batch_size': batch_size, 'enforce_eager': False, 'chunk_size': chunk_size, 'max_model_len': max_model_len},
             'versions': package_versions(),
             'manifest': manifest}
    if model in MODELS_1C:
        stats['phase'] = requests[0]['phase']
        stats['model_spec'] = MODELS_1C[model]
        stats['sampling']['max_tokens'] = thinking_tokens if MODELS_1C[model]['ladder']=='think' else 768
        stats['model_spec'] = {**MODELS_1C[model], 'max_tokens':stats['sampling']['max_tokens'], 'max_model_len':max_model_len}
        stats['sampling']['thinking_mode'] = 'native-checkpoint-template'
        stats['sampling'].pop('enable_thinking', None)
        stats['sampling'].pop('thinking_on', None)
    stats_path = directory / f'{slug}.metrics.json'
    stats_path.write_text(json.dumps(stats, indent=2) + '\n')
    raw_volume.commit()
    return {'raw': f'{run_id}/{raw_path.name}', 'metrics': f'{run_id}/{stats_path.name}'}


@app.function(image=image, gpu='L40S', volumes={'/hf': hf_cache, '/raw': raw_volume},
              timeout=5400, startup_timeout=1200, single_use_containers=True,
              max_containers=6, cpu=4, memory=32768)
def run_model(model: str, requests: list, run_id: str, manifest: dict, model_config: dict):
    return generate(model, requests, run_id, manifest, model_config)


def pull_file(remote, destination):
    """Pull committed Volume bytes, atomically replacing a temporary local file."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.part')
    with temporary.open('wb') as out:
        for chunk in raw_volume.read_file(remote):
            out.write(chunk)
    temporary.replace(target)


@app.local_entrypoint()
def main(models: str = '0.6B', limit_problems: int = 200, run_id: str = '', gpu_config: str = 'config/gpus.json',
         phase: str = 'phase1'):
    from datetime import datetime, timezone
    from src.build_prompts import build
    if phase not in ('phase1', 'phase1b'):
        raise ValueError('Unknown phase')
    selected = select_models('4B,8B,14B,32B' if phase == 'phase1b' and models == 'all' else models)
    if phase == 'phase1b' and any(m.endswith(('-0.6B', '-1.7B')) for m in selected):
        raise ValueError('Phase 1b models are 4B, 8B, 14B, 32B')
    config = load_gpu_config(gpu_config)
    if not run_id:
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if not run_id.replace('-', '').replace('_', '').isalnum():
        raise ValueError('run-id must contain only letters, digits, - or _')
    local = Path('results/raw') / run_id
    if local.exists():
        raise FileExistsError(f'Refusing to mix runs in {local}')
    requests, manifest = build(local, limit_problems, phase) if phase == 'phase1b' else build(local, limit_problems)
    print(f'Run {run_id}: {len(requests)} requests/model; models={selected}', flush=True)
    calls = []
    def collect(call):
        files = call.get()
        for remote in files.values():
            pull_file(remote, local / Path(remote).name)
            print(f'Pulled {remote} to {local}', flush=True)

    try:
        for model in selected:
            if len(calls) >= config.get('max_parallel_models', 5):
                collect(calls[0])
                calls.pop(0)
            spec = config['models'][model]
            fn = run_model.with_options(gpu=spec['gpu'])
            calls.append(fn.spawn(model, requests, run_id, manifest, spec))
        # Bound shared-Volume writers to five while keeping independent models parallel.
        for call in calls:
            collect(call)
    finally:
        # Abort siblings on exceptions/interrupt; no detached calls survive this entrypoint.
        for call in calls:
            call.cancel(terminate_containers=True)
    print(f'Raw run complete: {local}', flush=True)
