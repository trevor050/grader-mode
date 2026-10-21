"""Phase 6 standalone gated vLLM worker. No scoring or provisioning.

Build locally: python -m src.vast_generate_6 --build /fresh/requests.json
Run on authorized host: python -m src.vast_generate_6 --requests requests.json
  --model 3.8 --output-dir /phase6/output --deadline-unix UNIX_SECONDS

The CLI supervises one persistent engine process, killing its process group at
the absolute deadline, including while loading, generating, or waiting for a
permit. The caller/controller owns rental destruction and accounting.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from src.build_prompts import DATASET_REVISION, ROOT, sample_problems
from src.models_6 import (CHUNK_SIZE, EOS_TOKEN_IDS, MAX_MODEL_LEN, MODELS_6,
                          REGISTRATION_COMMIT, SAMPLE_SHA256, SAMPLING,
                          SOURCE_SHA256, THINK_CLOSE, THINK_OPEN, VLLM_VERSION,
                          engine_kwargs, model_config)
from src.vast_generate_4 import expand_requests as phase4_requests

DEFAULT_PHASE4_MANIFEST = Path('/Users/user/codeprojects/grader-mode/results/raw/phase4-main/manifest.json')
CELL_ORDER = ('main/off/plain', 'main/off/explicit', 'main/on/plain',
              'main/on/explicit', 'prefill/reminder', 'prefill/neutral',
              'prefill/irrelevant')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_sources():
    from src.phase6_registration import registered_source_matches
    for filename, expected in SOURCE_SHA256.items():
        if not registered_source_matches(filename, file_hash(ROOT / filename), expected):
            raise ValueError(f'immutable Phase 6 source hash mismatch: {filename}')


def recipe():
    return (json.loads((ROOT / 'prompts/templates_4.json').read_text()),
            json.loads((ROOT / 'prompts/templates.json').read_text()),
            (ROOT / 'prompts/system_tutor.txt').read_text())


def expand_requests(problems, model):
    cfg = model_config(model)
    rows = phase4_requests(problems, *recipe(), prefill=model == '3.8')
    result = []
    for r in rows:
        if r['request_type'] not in ('plain', 'explicit'):
            continue
        if r['arm'] == 'prefill' and (r['request_type'] != 'explicit' or r['prefill'] == 'empty'):
            continue
        result.append({**r, 'phase': 'phase6', 'model': cfg['model'],
                       'source_request_id': r['id'],
                       'id': r['id'].replace('phase4:', 'phase6:', 1)})
    return result


def cell_id(row):
    if row['arm'] == 'prefill':
        return f"prefill/{row['prefill']}"
    return f"main/{'on' if row['thinking'] else 'off'}/{row['request_type']}"


def load_sample(path=None):
    path = Path(path) if path is not None else DEFAULT_PHASE4_MANIFEST
    if path.is_file():
        old = json.loads(path.read_text())
        if old.get('dataset_revision') != DATASET_REVISION or old.get('sample_seed') != 0:
            raise ValueError('Phase 4 sample provenance mismatch')
        for name, expected in SOURCE_SHA256.items():
            if name.startswith('prompts/') and old.get('source_sha256', {}).get(name) != expected:
                raise ValueError(f'Phase 4 recipe hash mismatch: {name}')
        sample = old['sample']
        origin = {'kind': 'phase4_manifest', 'sha256': file_hash(path)}
    elif path != DEFAULT_PHASE4_MANIFEST:
        raise FileNotFoundError(path)
    else:
        from datasets import load_dataset
        sample = sample_problems(load_dataset('openai/gsm8k', 'main', split='test',
                                             revision=DATASET_REVISION))
        origin = {'kind': 'pinned_dataset_reconstruction'}
    if len(sample) != 200 or digest(sample) != SAMPLE_SHA256:
        raise ValueError('original fixed200 sample hash mismatch')
    return sample, origin


def build(output, phase4_manifest=None):
    check_sources()
    sample, origin = load_sample(phase4_manifest)
    manifest = dict(phase='phase6', registration_commit=REGISTRATION_COMMIT,
                    dataset='openai/gsm8k', dataset_revision=DATASET_REVISION,
                    sample_seed=0, sample_size=200, sample=sample, sample_sha256=SAMPLE_SHA256,
                    sample_origin=origin, source_sha256=SOURCE_SHA256,
                    models=MODELS_6, sampling=SAMPLING,
                    seed_pairing='SHA256 phase4:problem:template, first 4 bytes',
                    empty_prefill_reuse='main/on/explicit', pilot_rows_in_main=False)
    spec = dict(schema_version=1, manifest=manifest, models={
        m: {'model': MODELS_6[m]['model'], 'requests': expand_requests(sample, m)}
        for m in MODELS_6})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, spec, immutable=True)
    atomic_write(output.with_suffix('.README.md'), INTERFACE_README, immutable=True)
    return spec


def validate_spec(spec, model):
    check_sources()
    if spec.get('schema_version') != 1:
        raise ValueError('unsupported requests schema')
    manifest = spec['manifest']
    expected = {'phase': 'phase6', 'registration_commit': REGISTRATION_COMMIT,
                'dataset_revision': DATASET_REVISION, 'sample_seed': 0, 'sample_size': 200,
                'sample_sha256': SAMPLE_SHA256, 'source_sha256': SOURCE_SHA256,
                'models': MODELS_6, 'sampling': SAMPLING,
                'empty_prefill_reuse': 'main/on/explicit', 'pilot_rows_in_main': False}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('requests manifest/pins/settings mismatch')
    sample = manifest['sample']
    if len(sample) != 200 or digest(sample) != SAMPLE_SHA256:
        raise ValueError('original fixed200 sample hash mismatch')
    if set(spec['models']) != set(MODELS_6):
        raise ValueError('unexpected model grid')
    for version, cfg in spec['models'].items():
        if cfg != {'model': MODELS_6[version]['model'], 'requests': expand_requests(sample, version)}:
            raise ValueError(f'mixed or incomplete request grid: {version}')
    return spec['models'][model]['requests'], manifest


def pilot_requests(rows, model, stage):
    if stage not in ('smoke', 'throughput'):
        raise ValueError('unknown pilot stage')
    cells = CELL_ORDER if model == '3.8' else CELL_ORDER[:4]
    grouped = {c: [r for r in rows if cell_id(r) == c] for c in cells}
    if any(not group for group in grouped.values()):
        raise ValueError('pilot requires every registered cell')
    if stage == 'throughput':
        # Fixed round-robin: 37/36 across seven cells, 64 across four.
        # No outcome-dependent selection, including for the final remainder.
        selected = [grouped[cells[i % len(cells)]][(i // len(cells)) % len(grouped[cells[i % len(cells)]])]
                    for i in range(256)]
    elif model == '3.5':
        selected = [next(r for r in grouped[c] if r['template_idx'] == t)
                    for t in (0, 1) for c in cells]
    else:
        selected = [grouped[c][0] for c in cells]
        selected.append(next(r for r in grouped[cells[0]]
                             if r['problem_idx'] != selected[0]['problem_idx']))
    return [{**r, 'id': f'phase6-pilot:{stage}:{i:04d}:{r["id"]}',
             'study_request_id': r['id'], 'excluded_from_main': True}
            for i, r in enumerate(selected)]


def render_prompt(tokenizer, row, model):
    cfg = model_config(model)
    kwargs = dict(tokenize=False, add_generation_prompt=True,
                  enable_thinking=row['thinking'], preserve_thinking=False)
    if cfg['reasoning_effort'] is not None:
        kwargs['reasoning_effort'] = cfg['reasoning_effort']
    native = tokenizer.apply_chat_template(row['messages'], **kwargs)
    suffix = '<|im_start|>assistant\n<think>\n' if row['thinking'] else (
        '<|im_start|>assistant\n<think>\n\n</think>\n\n')
    if not native.endswith(suffix):
        raise ValueError(f'native assistant suffix mismatch: {model}/{row["id"]}')
    if row['arm'] == 'prefill':
        if not row['thinking'] or not row['prefill_text']:
            raise ValueError('prefill must be native thinking on and nonempty')
        return native + row['prefill_text']
    return native


def prepare_requests(tokenizer, rows, model):
    result = []
    for row in rows:
        prompt = render_prompt(tokenizer, row, model)
        ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        if len(ids) + SAMPLING['max_tokens'] > MAX_MODEL_LEN:
            raise ValueError(f'prompt + 4096 exceeds 8192: {row["id"]} ({len(ids)})')
        result.append((row, prompt, ids))
    return result


def token_fields(tokenizer, ids, thinking, prefill_text=''):
    """Save protocol spans, never infer numeric answers or tutor adherence."""
    def decode(tokens):
        return tokenizer.decode(tokens, skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
    ends = [i for i, token in enumerate(ids) if token in EOS_TOKEN_IDS]
    end = ends[0] if ends else len(ids)
    closers = [i for i, token in enumerate(ids) if token == THINK_CLOSE]
    opening = 1 if ids and ids[0] == THINK_OPEN else 0
    close = next((i for i in closers if i < end), None)
    if thinking:
        reasoning_span = [opening, close if close is not None else end]
        final_span = [close + 1, end] if close is not None else [end, end]
    else:
        reasoning_span, final_span = [0, 0], [0, end]
    reasoning = decode(ids[slice(*reasoning_span)])
    specials = set(getattr(tokenizer, 'all_special_ids', ())) | {THINK_OPEN, THINK_CLOSE, *EOS_TOKEN_IDS}
    segments, start = [], 0
    for i, token in enumerate(ids):
        if token in specials:
            if start < i:
                segments.append(dict(start=start, end=i, text=decode(ids[start:i])))
            segments.append(dict(start=i, end=i + 1, text=decode([token])))
            start = i + 1
    if start < len(ids):
        segments.append(dict(start=start, end=len(ids), text=decode(ids[start:])))
    raw_text = decode(ids)
    if ''.join(s['text'] for s in segments) != raw_text:
        raise ValueError('saved token segment decoding disagrees with full raw text')
    return dict(raw_text=raw_text, response=raw_text, reasoning_continuation=reasoning,
                reasoning_with_prefill=prefill_text + reasoning,
                final_continuation=decode(ids[slice(*final_span)]),
                initial_channel='thought' if thinking else 'final',
                special_token_ids=sorted(specials), decoded_token_segments=segments,
                token_spans={'reasoning': reasoning_span, 'final': final_span},
                token_boundaries={'think_open': [i for i, t in enumerate(ids) if t == THINK_OPEN],
                                  'think_close': closers, 'message_end': ends},
                thinking_closed=close is not None if thinking else True)


def checkpoint_metadata(model):
    """Download/verify only five minimal files before any engine/weight load."""
    from huggingface_hub import snapshot_download
    cfg = model_config(model)
    directory = Path(snapshot_download(repo_id=cfg['model'], revision=cfg['revision'],
                                       allow_patterns=list(cfg['file_sha256'])))
    hashes = {name: file_hash(directory / name) for name in cfg['file_sha256']}
    if hashes != cfg['file_sha256']:
        raise ValueError(f'checkpoint metadata hash mismatch: {model}')
    config = json.loads((directory / 'config.json').read_text())
    generation = json.loads((directory / 'generation_config.json').read_text())
    if config.get('quantization_config', {}).get('quant_method') != 'fp8':
        raise ValueError('checkpoint is not publisher FP8')
    if generation.get('eos_token_id') != list(EOS_TOKEN_IDS):
        raise ValueError('checkpoint EOS mismatch')
    return dict(model=cfg['model'], revision=cfg['revision'], file_sha256=hashes,
                checkpoint_quantization=config['quantization_config'],
                generation_config=generation)


def atomic_write(path, text, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    if immutable and path.exists():
        raise FileExistsError(path)
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    if immutable and path.exists():
        temporary.unlink()
        raise FileExistsError(path)
    os.replace(temporary, path)


def atomic_json(path, value, *, immutable=False):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                 immutable=immutable)


def claim_output(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # A shell may open worker.log before exec. It is never rewritten by us and
    # is not generation/resume state. Every other pre-existing entry is refused.
    if any(p.name != 'worker.log' or not p.is_file() for p in directory.iterdir()):
        raise FileExistsError(f'output must be fresh/empty; resume is forbidden: {directory}')
    with (directory / '.worker.lock').open('x') as stream:
        stream.write(str(os.getpid()))
    # The exclusive lock also rejects a racing second worker.
    return directory


def check_deadline(deadline_unix):
    if not math.isfinite(deadline_unix) or time.time() >= deadline_unix:
        raise TimeoutError('Phase 6 absolute deadline reached')


def wait_permit(path, *, deadline_unix, model, requests_sha256):
    path = Path(path)
    while True:
        check_deadline(deadline_unix)
        if path.exists():
            try:
                permit = json.loads(path.read_text())
                if not isinstance(permit, dict) or permit.get('decision') not in ('run', 'drop'):
                    raise ValueError('permit decision must be run or drop')
                for key, expected in [('model', model), ('requests_sha256', requests_sha256)]:
                    if key in permit and permit[key] != expected:
                        raise ValueError(f'permit {key} mismatch')
                return permit['decision']
            except (json.JSONDecodeError, UnicodeError) as error:
                raise ValueError(f'malformed permit: {path}') from error
        time.sleep(min(.25, max(0, deadline_unix - time.time())))


def execute_stages(engine, tokenizer, rows, model, output_dir, manifest, *,
                   settings, requests_sha256, deadline_unix, prepared=None):
    from vllm import SamplingParams
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    prepared = prepared or prepare_requests(tokenizer, rows, model)
    by_id = {r['id']: (prompt, ids) for r, prompt, ids in prepared}
    implementation_hashes = {name: file_hash(ROOT / name) for name in (
        'src/vast_generate_6.py', 'src/models_6.py', 'src/vast_generate_4.py', 'src/build_prompts.py',
        'src/phase6_registration.py')}
    stages = [('smoke', pilot_requests(rows, model, 'smoke')),
              ('throughput', pilot_requests(rows, model, 'throughput')),
              ('main', [r for r in rows if r['arm'] == 'main'])]
    if model == '3.8':
        stages.append(('prefill', [r for r in rows if r['arm'] == 'prefill']))
    for stage, stage_rows in stages:
        common = dict(stage=stage, model=model, model_id=MODELS_6[model]['model'],
                      requests_sha256=requests_sha256,
                      run_id='phase6-' + requests_sha256[:16],
                      registration_commit=REGISTRATION_COMMIT,
                      excluded_from_main=stage in ('smoke', 'throughput'))
        if stage != 'smoke':
            atomic_json(directory / 'progress.json', {**common, 'status': 'waiting_permit'})
            decision = wait_permit(directory / f'{stage}.permit.json',
                                   deadline_unix=deadline_unix, model=model,
                                   requests_sha256=requests_sha256)
            if decision == 'drop':
                state = {**common, 'status': 'dropped', 'requests': 0, 'chunks': [],
                         'output_tokens': 0, 'generation_seconds': 0, 'per_cell': {}}
                atomic_json(directory / f'{stage}.done.json', state, immutable=True)
                atomic_json(directory / 'worker.done.json', state, immutable=True)
                return state
        per_cell = defaultdict(lambda: {'count': 0, 'output_tokens': 0})
        chunks, generation_seconds, total_tokens = [], 0., 0
        for offset in range(0, len(stage_rows), CHUNK_SIZE):
            check_deadline(deadline_unix)
            chunk = stage_rows[offset:offset + CHUNK_SIZE]
            prompt_data = [by_id[r.get('study_request_id', r['id'])] for r in chunk]
            params = [SamplingParams(**SAMPLING, seed=r['seed'], n=1,
                                     skip_special_tokens=False,
                                     stop_token_ids=list(EOS_TOKEN_IDS), ignore_eos=False)
                      for r in chunk]
            started = time.monotonic()
            outputs = engine.generate([{'prompt_token_ids': ids} for _, ids in prompt_data],
                                      params, use_tqdm=True)
            elapsed = time.monotonic() - started
            check_deadline(deadline_unix)
            if len(outputs) != len(chunk):
                raise ValueError('engine output count differs from request count')
            raw_rows = []
            for row, (prompt, prompt_ids), output in zip(chunk, prompt_data, outputs):
                if len(output.outputs) != 1 or list(output.prompt_token_ids) != prompt_ids:
                    raise ValueError('engine prompt/response identity mismatch')
                completion = output.outputs[0]
                ids = list(completion.token_ids)
                if len(ids) > SAMPLING['max_tokens'] or completion.finish_reason not in ('stop', 'length'):
                    raise ValueError('invalid completion cap/finish reason')
                raw_rows.append({**row, **common, 'model': MODELS_6[model]['model'],
                                 'run_id': manifest.get('run_id', 'phase6-main'),
                                 'registration_commit': REGISTRATION_COMMIT,
                                 'implementation_sha256': implementation_hashes,
                                 'model_version': model, 'output_token_ids': ids,
                                 'output_tokens': len(ids), 'prompt': prompt,
                                 'prompt_token_ids': prompt_ids, 'prompt_tokens': len(prompt_ids),
                                 'engine_text': completion.text,
                                 'finish_reason': completion.finish_reason,
                                 'stop_reason': getattr(completion, 'stop_reason', None),
                                 'sampling': {**SAMPLING, 'n': 1,
                                              'stop_token_ids': list(EOS_TOKEN_IDS),
                                              'skip_special_tokens': False},
                                 'settings': settings, 'pins': model_config(model),
                                 'model_revision': MODELS_6[model]['revision'],
                                 'tokenizer_revision': MODELS_6[model]['revision'],
                                 'dtype': 'bfloat16', 'quantization': 'fp8',
                                 'reasoning_effort': MODELS_6[model]['reasoning_effort'],
                                 'vllm_version': VLLM_VERSION,
                                 'package_versions': settings.get('versions', {'vllm': VLLM_VERSION}),
                                 'source_sha256': manifest.get('source_sha256', {}),
                                 **token_fields(tokenizer, ids, row['thinking'], row['prefill_text'])})
            path = directory / stage / f'chunk-{offset // CHUNK_SIZE:05d}.jsonl'
            atomic_write(path, ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n'
                                       for r in raw_rows), immutable=True)
            for row in raw_rows:
                values = per_cell[cell_id(row)]
                values['count'] += 1
                values['output_tokens'] += row['output_tokens']
                total_tokens += row['output_tokens']
            generation_seconds += elapsed
            chunks.append(dict(path=str(path.relative_to(directory)), sha256=file_hash(path),
                               count=len(chunk), output_tokens=sum(r['output_tokens'] for r in raw_rows),
                               generation_seconds=elapsed))
            state = {**common, 'status': 'running', 'requests': offset + len(chunk),
                     'output_tokens': total_tokens, 'generation_seconds': generation_seconds,
                     'per_cell': dict(per_cell), 'chunks': chunks,
                     'settings': settings, 'updated_unix': time.time()}
            atomic_json(directory / f'{stage}.metrics.json', state)
            atomic_json(directory / 'progress.json', state)
        state = {**state, 'status': 'completed'}
        atomic_json(directory / f'{stage}.metrics.json', state)
        atomic_json(directory / f'{stage}.done.json', state, immutable=True)
    final = dict(status='completed', model=model, requests_sha256=requests_sha256,
                 stages=[s for s, _ in stages], finished_unix=time.time())
    atomic_json(directory / 'worker.done.json', final, immutable=True)
    return final


def run_worker(args):
    check_deadline(args.deadline_unix)
    directory = claim_output(args.output_dir)
    started = time.monotonic()
    try:
        spec = json.loads(args.requests.read_text())
        rows, manifest = validate_spec(spec, args.model)
        request_hash = file_hash(args.requests)
        atomic_json(directory / 'manifest.json', {**manifest, 'requests_sha256': request_hash,
                                                 'selected_model': args.model}, immutable=True)
        atomic_write(directory / 'README.md', INTERFACE_README, immutable=True)
        versions = {p: metadata.version(p) for p in ('vllm', 'transformers', 'tokenizers', 'jinja2', 'torch')}
        if versions['vllm'] != VLLM_VERSION:
            raise ValueError(f'vllm {VLLM_VERSION} required; got {versions["vllm"]}')
        minimal = checkpoint_metadata(args.model)
        atomic_json(directory / 'checkpoint-metadata.json', minimal, immutable=True)
        check_deadline(args.deadline_unix)
        from transformers import AutoTokenizer
        cfg = model_config(args.model)
        tokenizer = AutoTokenizer.from_pretrained(cfg['model'], revision=cfg['revision'])
        for text, expected in [('<think>', THINK_OPEN), ('</think>', THINK_CLOSE),
                               ('<|im_end|>', EOS_TOKEN_IDS[0]), ('<|endoftext|>', EOS_TOKEN_IDS[1])]:
            if tokenizer.encode(text, add_special_tokens=False) != [expected]:
                raise ValueError(f'tokenizer special token mismatch: {text}')
        # Assert every main and prefill prompt fits before loading any weights.
        prepared = prepare_requests(tokenizer, rows, args.model)
        examples = {}
        for row, prompt, ids in prepared:
            examples.setdefault(cell_id(row), dict(prompt=prompt, prompt_tokens=len(ids),
                                                   prefill_tokens=len(tokenizer.encode(
                                                       row['prefill_text'], add_special_tokens=False))))
        atomic_json(directory / 'prompt-preflight.json', dict(
            requests=len(rows), max_prompt_tokens=max(len(ids) for _, _, ids in prepared),
            context=MAX_MODEL_LEN, output_cap=SAMPLING['max_tokens'],
            examples=examples), immutable=True)
        options = engine_kwargs(args.model, args.max_num_seqs)
        settings = dict(engine=options, sampling=SAMPLING, versions=versions,
                        chunk_size=CHUNK_SIZE, python=sys.version,
                        native_template={'preserve_thinking': False,
                                         'reasoning_effort': cfg['reasoning_effort']})
        atomic_json(directory / 'settings.json', settings, immutable=True)
        check_deadline(args.deadline_unix)
        from vllm import LLM
        engine = LLM(**options)
        settings['setup_seconds'] = time.monotonic() - started
        atomic_json(directory / 'setup.done.json', dict(status='completed', **settings), immutable=True)
        return execute_stages(engine, tokenizer, rows, args.model, directory, manifest,
                              settings=settings, requests_sha256=request_hash,
                              deadline_unix=args.deadline_unix, prepared=prepared)
    except BaseException as error:
        atomic_json(directory / 'error.json', dict(status='failed',
                    error=f'{type(error).__name__}: {error}', finished_unix=time.time()), immutable=True)
        raise


def supervise(args, argv):
    """OS-level timeout, independent of the engine's Python/CUDA responsiveness."""
    check_deadline(args.deadline_unix)
    process = subprocess.Popen([sys.executable, '-m', 'src.vast_generate_6', *argv, '--_worker'],
                               start_new_session=True)
    try:
        return process.wait(timeout=max(.001, args.deadline_unix - time.time()))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        directory = Path(args.output_dir)
        if directory.is_dir() and not (directory / 'worker.timeout.json').exists():
            atomic_json(directory / 'worker.timeout.json', dict(status='timeout',
                        deadline_unix=args.deadline_unix, finished_unix=time.time()), immutable=True)
        return 124
    finally:
        # vLLM descendants must not survive a failed or interrupted worker.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            process.wait()


INTERFACE_README = '''# Phase 6 worker/controller interface

Build (CPU only, no weights or inference):
`python -m src.vast_generate_6 --build /fresh/requests.json`
Optional `--phase4-manifest PATH` selects the original Phase 4 manifest. The
default reads the main Mac checkout; when absent, the pinned GSM8K revision
reconstructs the exact fixed200 sample. Its SHA256 must match in both cases.
This writes requests.json and requests.README.md; neither may already exist.

Requests JSON schema v1: {"schema_version": 1, "manifest": {...}, "models":
{"3.8": {"model": "Qwen/Qwen3.8-27B-FP8", "requests": [...]},
 "3.5": {"model": "Qwen/Qwen3.5-27B-FP8", "requests": [...]}}}.
Each model's complete exact grid, manifest, recipe hashes, pins and original
sample hash are checked before engine initialization. No partial input grids.

Run one model in one fresh/empty output directory:
`python -m src.vast_generate_6 --requests /phase6/requests.json --model 3.8
 --output-dir /phase6/output --max-num-seqs 256 --deadline-unix UNIX_SECONDS`
Shell stdout/stderr may use output-dir/worker.log (the sole allowed preexisting
entry); every other preexisting file is refused. 256 is the initial 96/80GB
setting. Explicit 32/64/128 are available for a 48GB attempt. Chunks stay 256.
Use a distinct output directory for 3.5. Existing/mixed output is rejected;
there is no resume/overwrite mode. The CLI supervises an isolated process group
and hard-kills it at the absolute deadline (exit 124), also during engine load,
generation and permit waits. Controller still owns rental destruction.

One engine stays resident throughout this sequence:
1. smoke: 8 rows, writes smoke.done.json, waits throughput.permit.json.
2. throughput: 256 rows, writes throughput.done.json, waits main.permit.json.
3. main: 1600 rows, writes main.done.json; 3.5 exits completed here.
4. 3.8 only: waits prefill.permit.json; runs all 1200 and writes prefill.done.json.

Write each permit with temporary-file + atomic rename. Minimal permit is
{"decision": "run"} or {"decision": "drop"}. Optional "model": "3.8" and
"requests_sha256": "<SHA256 of requests.json bytes>" are validated when supplied.
Any drop writes that stage's .done.json with status=dropped and zero requests,
writes worker.done.json status=dropped, and exits zero without later stages.
Malformed/misbound permits fail closed. No implicit timeout approval.

Done/metrics JSON: status, stage, model (3.8|3.5), model_id, requests,
output_tokens, generation_seconds, per_cell: {cell: {count, output_tokens}},
chunks: [{path, sha256, count, output_tokens, generation_seconds}], settings,
requests_sha256, excluded_from_main. generation_seconds sums timed engine calls
and excludes setup, disk writes and permit waits. Tokens count saved output IDs
including generated EOS; supplied prefills are input. settings.setup_seconds
covers metadata, tokenizer, preflight and engine initialization. Use measured
aggregate generation_seconds, not per-cell parallel wall-time estimates.

Cell IDs in fixed round-robin order: main/off/plain, main/off/explicit,
main/on/plain, main/on/explicit, prefill/reminder, prefill/neutral,
prefill/irrelevant. Each study cell has 400 rows. 3.5 uses the first four.
The 3.8 throughput counts are 37/37/37/37/36/36/36; 3.5 is 64 each.
Smoke covers one of each 3.8 cell plus a second problem in main/off/plain;
3.5 covers four main cells at each of templates 0 and 1.

Incremental pull: <stage>/chunk-00000.jsonl etc. Each file is written once,
fsynced and atomically renamed from .tmp. Pull only final .jsonl files.
Per-stage .metrics.json and progress.json update atomically after each complete
chunk. The stage's .done.json appears only after every row has been published.
error.json records ordinary failures; worker.timeout.json records the
supervisor deadline. No completed done file is published for a failed stage.
worker.done.json marks the whole successful/dropped worker lifecycle.

Smoke and throughput have disjoint stage paths and namespaced request IDs,
excluded_from_main=true, and never contribute to the 1600/1200 study totals.
They exercise original-sample prompts but are separate generations, never
reused as main replies. Original paired seeds are retained. No scoring occurs.
The empty-prefill baseline is main/on/explicit; no duplicate empty study cell.

Raw rows retain exact input messages, original phase4 source_request_id,
seed, prompt text/IDs/count, output_token_ids, raw_text decoded with special
tokens, engine_text, finish_reason, stop_reason, initial_channel (thought|final), literal
prefill_text, pins, settings and protocol token boundaries/spans. Spans are
zero-based half-open continuation-token offsets, excluding closing/EOS markers.
response equals raw_text. decoded_token_segments partition every output token,
isolating each special token; special_token_ids retain all tokenizer specials
plus the four pinned protocol IDs, enabling scoring without a tokenizer.
reasoning_continuation and final_continuation retain literal generated text;
reasoning_with_prefill is an audit field. These are not scored fields.
A capped thinking-on reply without a closing token has an empty final_continuation.
No whitespace is stripped, no numeric/violation classification is performed.

Before weights: checkpoint-metadata.json hashes config.json,
generation_config.json, tokenizer_config.json, tokenizer.json, chat_template.jinja
at the exact model revision. Hash mismatches abort. prompt-preflight.json checks
all prompts + 4096 <= 8192 and records examples. settings.json records runtime
versions and engine settings; setup.done.json includes measured setup_seconds.
vLLM must be exactly 0.32.0. CUDA graphs remain enabled, native enable_thinking
and preserve_thinking=false are used; 3.8 uses reasoning_effort=medium.
'''


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--phase4-manifest', type=Path)
    parser.add_argument('--requests', type=Path)
    parser.add_argument('--model', choices=tuple(MODELS_6))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--max-num-seqs', type=int, choices=(32, 64, 128, 256), default=256)
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.build:
        if args.requests or args._worker:
            parser.error('--build and --requests/--_worker are mutually exclusive')
        spec = build(args.build, args.phase4_manifest)
        print(json.dumps({m: len(cfg['requests']) for m, cfg in spec['models'].items()}))
        return 0
    if not all((args.requests, args.model, args.output_dir, args.deadline_unix)):
        parser.error('generation requires --requests, --model, --output-dir, --deadline-unix')
    check_deadline(args.deadline_unix)
    if args._worker:
        run_worker(args)
        return 0
    return supervise(args, argv)


if __name__ == '__main__':
    raise SystemExit(main())
