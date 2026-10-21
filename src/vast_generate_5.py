"""Phase 5 request builder and isolated vLLM 0.32 generation worker.

Pure CPU APIs: ``build(model_key, limit_problems=200) -> (rows, manifest)``,
``expand_requests(problems, model_key)``, ``pilot_requests(rows)``.
GPU execution is only in ``generate``/the explicit non-build CLI path; there
are no Modal objects, imports, provisioning, or changes to older phases.
``--output-dir`` names the exact run directory, which may already hold logs.
The controller owns lifetime accounting, gates, and complete-arm decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time

from src.build_prompts import ROOT, DATASET_REVISION, request_seed, sample_problems
from src.models_5 import (GEMMA, NEMOTRON, GLM, MODELS_5, PACKAGE_VERSIONS, PREFILLS,
                          REGISTRATION_COMMIT, ORIGINAL_REGISTRATION_COMMIT,
                          RENDER_DATE, SAMPLING, model_spec, resolve_model)


def _inputs():
    return (json.loads((ROOT / 'prompts/templates_4.json').read_text()),
            json.loads((ROOT / 'prompts/templates.json').read_text()),
            (ROOT / 'prompts/system_tutor.txt').read_text())


def expand_requests(problems, model_key, templates=None, phase1_templates=None, system=None):
    """Expand the same 16 registered rows per original GSM8K problem for every family."""
    model = resolve_model(model_key)
    defaults = _inputs()
    templates = defaults[0] if templates is None else templates
    phase1_templates = defaults[1] if phase1_templates is None else phase1_templates
    system = defaults[2] if system is None else system
    if (set(templates['prefills']) != set(PREFILLS)
            or not {'plain', 'explicit'} <= templates['requests'].keys()
            or len(phase1_templates['cues']['none']) < 2):
        raise ValueError('Missing registered Phase 4 prefills/requests or Phase 1 templates')
    if not isinstance(system, str) or not system.strip():
        raise ValueError('The tutor system message is required')
    result, seen = [], set()
    for p in problems:
        if (not isinstance(p, dict) or type(p.get('problem_idx')) is not int
                or p['problem_idx'] < 0 or p['problem_idx'] in seen
                or not isinstance(p.get('question'), str) or not p['question'].strip()
                or type(p.get('gold')) is not int):
            raise ValueError('Expected distinct original problem_idx, question and integer gold')
        seen.add(p['problem_idx'])
        cells = [('main', thinking, request, 'none') for thinking in (False, True)
                 for request in ('plain', 'explicit')]
        cells += [('prefill', True, 'explicit', name) for name in PREFILLS]
        for template_idx, template in enumerate(phase1_templates['cues']['none'][:2]):
            seed_id = f"phase5:p{p['problem_idx']:04d}:t{template_idx}"
            for arm, thinking, request, prefill in cells:
                rid = f"{seed_id}:{model.split('/')[-1]}:{arm}:thinking-{int(thinking)}:{request}:{prefill}"
                result.append({**p, 'problem_id': p['problem_idx'], 'phase': 'phase5',
                    'id': rid, 'model': model, 'template_idx': template_idx, 'template': template_idx,
                    'arm': arm, 'thinking': thinking, 'request_type': request, 'prefill': prefill,
                    'prefill_text': templates['prefills'][prefill] if arm == 'prefill' else '',
                    'seed': request_seed(seed_id), 'messages': [
                        {'role': 'system', 'content': system},
                        {'role': 'user', 'content': template.replace('{problem}', p['question'])
                         + templates['requests'][request]}]})
    return result


def build(model_key, limit_problems=200):
    """Always sample the fixed 200 first; a limit only selects that sample's prefix."""
    model = resolve_model(model_key)
    if type(limit_problems) is not int or not 1 <= limit_problems <= 200:
        raise ValueError('limit_problems must be an integer from 1 to 200')
    from datasets import load_dataset
    full_sample = sample_problems(load_dataset('openai/gsm8k', 'main', split='test', revision=DATASET_REVISION))
    selected = full_sample[:limit_problems]
    rows = expand_requests(selected, model)
    paths = ('DESIGN.md', 'prompts/templates.json', 'prompts/templates_4.json', 'prompts/system_tutor.txt')
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}
    manifest = {'phase': 'phase5', 'registration_commit': REGISTRATION_COMMIT,
        'original_registration_commit': ORIGINAL_REGISTRATION_COMMIT, 'amendment': 1,
        'dataset': 'openai/gsm8k', 'config': 'main', 'split': 'test',
        'dataset_revision': DATASET_REVISION, 'sample_seed': 0, 'sample_size': 200,
        'sample': full_sample, 'selected_sample': selected, 'limit_problems': limit_problems,
        'requests': len(rows), 'model': model, 'model_spec': model_spec(model),
        'models': {m: model_spec(m) for m in MODELS_5},
        'source_sha256': hashes, 'sampling': dict(SAMPLING), 'render_date': RENDER_DATE,
        'package_versions': dict(PACKAGE_VERSIONS),
        'seed_pairing': 'SHA256 phase5:p{original_problem_idx:04d}:t{template_idx}, first 4 bytes, big endian',
        'implementation_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ('src/models_5.py','src/vast_generate_5.py','src/phase5_split.py')}}
    return rows, manifest


def _cell(row):
    return row['arm'], row['thinking'], row['request_type'], row['prefill']


def pilot_requests(rows):
    """16 throughput-only rows: every cell, equally represented in both templates.

    Every family uses one problem/cell/template. Selection is by construction
    order, independent of outputs and throughput measurements.
    """
    rows = list(rows)
    if not rows or len({r['model'] for r in rows}) != 1:
        raise ValueError('pilot needs one nonempty model grid')
    model = resolve_model(rows[0]['model'])
    cells = list(dict.fromkeys(_cell(r) for r in rows))
    expected = 8
    if len(cells) != expected:
        raise ValueError('pilot must cover all registered cells before arm drops')
    per_template = 1
    pilot = []
    for cell in cells:
        for template in (0, 1):
            candidates = [r for r in rows if _cell(r) == cell and r['template_idx'] == template]
            if len(candidates) < per_template:
                raise ValueError('Not enough problems to form the balanced 16-row pilot')
            pilot.extend(candidates[:per_template])
    if len({r['id'] for r in pilot}) != 16:
        raise ValueError('Duplicate pilot request identities')
    return pilot


def initial_channel(row):
    if model_spec(row['model'])['format'] == 'think_tags':
        return 'thought' if row['thinking'] else 'final'
    return 'thought' if row['arm'] == 'prefill' else ('auto' if row['thinking'] else 'final')


def render_prompt(tokenizer, row):
    """Serialize once using the pinned native template and literal prefill bytes."""
    spec = model_spec(row['model'])
    messages = row['messages']
    if (len(messages) != 2 or [m['role'] for m in messages] != ['system', 'user']
            or type(row['thinking']) is not bool):
        raise ValueError('Expected the genuine tutor system turn and one user turn')
    if row['arm'] not in ('main', 'prefill'):
        raise ValueError('Expected a registered main or prefill arm')
    if row['arm'] == 'prefill' and not row['thinking']:
        raise ValueError('Prefills require thinking on')
    base = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=row['thinking'])
    if spec['format'] == 'think_tags':
        tail = spec['native_on_tail'] if row['thinking'] else spec['native_off_tail']
        if not base.endswith(tail):
            raise ValueError('Pinned native think-tag template has an unexpected thinking tail')
        # The native opener, including Nemotron's newline, is already present.
        return base + row['prefill_text'] if row['arm'] == 'prefill' else base
    tail = '<|turn>model\n' + ('' if row['thinking'] else '<|channel>thought\n<channel|>')
    if not base.endswith(tail):
        raise ValueError('Pinned Gemma template has an unexpected native thinking tail')
    if row['arm'] == 'prefill':
        return base + '<|channel>thought\n' + row['prefill_text']
    return base


def runtime_versions():
    versions = {}
    for name in (*PACKAGE_VERSIONS, 'torch', 'huggingface-hub'):
        versions[name] = importlib.metadata.version(name)
    return versions


def verify_tokenizer(model_key, tokenizer):
    """Check metadata bytes, native template and special IDs at the pinned revision."""
    from huggingface_hub import hf_hub_download
    model, spec = resolve_model(model_key), model_spec(model_key)
    for name, expected in spec['file_sha256'].items():
        path = hf_hub_download(model, name, revision=spec['tokenizer_revision'])
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Pinned {name} hash mismatch')
    template = tokenizer.chat_template
    if not isinstance(template, str) or hashlib.sha256(template.encode()).hexdigest() != spec['file_sha256']['chat_template.jinja']:
        raise ValueError('Loaded tokenizer chat template does not match the pinned template')
    for token, expected in spec['special_tokens'].items():
        if tokenizer.convert_tokens_to_ids(token) != expected:
            raise ValueError(f'Pinned special token mismatch: {token}')


def _validate_requests(rows, model):
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty list of Phase 5 requests')
    ids = set()
    expected_cache = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Requests must be objects')
        try:
            pid = row['problem_idx']
            if pid not in expected_cache:
                expected_cache[pid] = {r['id']:r for r in expand_requests([
                    {k:row[k] for k in ('problem_idx','question','gold')}],model)}
            expected = expected_cache[pid].get(row['id'])
            if expected is None or any(type(row.get(k)) is not type(v) or row[k] != v for k,v in expected.items()):
                raise ValueError('Request does not match the registered Phase 5 construction')
            if row['id'] in ids:
                raise ValueError('Duplicate request id')
            ids.add(row['id'])
        except (KeyError, TypeError) as exc:
            raise ValueError('Malformed Phase 5 request') from exc


def generate(model_key, requests, run_id, manifest, model_config=None):
    """Run only the supplied requests, persist every raw ID and CPU-replayable split.

    ``model_config.output_dir`` is the exact run directory (default /raw/run_id).
    No implicit resume/overwrite or main-run release follows a pilot. A separate
    controller must authorize any later work after reading these timing metrics.
    """
    from src.phase5_split import decoded_segments, split_response
    model, spec = resolve_model(model_key), model_spec(model_key)
    _validate_requests(requests, model)
    if not isinstance(manifest,dict):
        raise ValueError('manifest must be an object')
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
        raise ValueError('run_id must be a nonempty path-free name')
    config = {} if model_config is None else dict(model_config)
    stage=manifest.get('stage',config.get('stage','main'))
    if stage not in ('main','pilot') or ('stage' in config and config['stage']!=stage):
        raise ValueError('stage must be main or pilot with consistent provenance')
    batch_size, chunk_size = config.get('batch_size', 16), config.get('chunk_size', 16)
    if any(type(v) is not int or v < 1 for v in (batch_size, chunk_size)):
        raise ValueError('batch_size and chunk_size must be positive integers')
    directory = Path(config.get('output_dir', Path('/raw') / run_id))
    slug = model.split('/')[-1]
    raw_path, metrics_path = directory / f'{slug}.jsonl', directory / f'{slug}.metrics.json'
    if raw_path.exists() or metrics_path.exists():
        raise FileExistsError(f'Refusing to overwrite Phase 5 outputs in {directory}')
    versions = runtime_versions()
    for package, expected in PACKAGE_VERSIONS.items():
        if versions.get(package) != expected:
            raise ValueError(f'Phase 5 requires {package}=={expected}, found {versions.get(package)!r}')
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model, revision=spec['tokenizer_revision'])
    verify_tokenizer(model, tokenizer)
    llm = LLM(model=model, tokenizer=model, revision=spec['revision'],
        tokenizer_revision=spec['tokenizer_revision'], dtype=spec['dtype'], quantization=spec['quantization'],
        tensor_parallel_size=1, max_model_len=spec['max_model_len'], max_num_seqs=batch_size,
        gpu_memory_utilization=.90, generation_config='vllm', seed=0, enforce_eager=True)
    loaded = time.monotonic()
    ordered = list(requests)
    random.Random(0).shuffle(ordered)
    directory.mkdir(parents=True, exist_ok=True)
    total_tokens = prompt_tokens = completed = 0
    generation_seconds = 0.0
    engine = {'batch_size':batch_size, 'chunk_size':chunk_size, 'max_model_len':spec['max_model_len'],
              'tensor_parallel_size':1, 'gpu_memory_utilization':.90, 'generation_config':'vllm',
              'enforce_eager':True}
    provenance = {'model': model, 'model_revision':spec['revision'], 'tokenizer_revision':spec['tokenizer_revision'],
        'dtype':spec['dtype'], 'quantization':spec['quantization'], 'reasoning_effort':spec['reasoning_effort'],
        'sampling': {**SAMPLING, 'skip_special_tokens':False, 'stop_token_ids':spec['stop_token_ids']},
        'run_id':run_id, 'stage':stage, 'pilot':stage=='pilot',
        'generation_started_unix':time.time(),
        'render_date':RENDER_DATE, 'package_versions':versions,
        'engine':engine, 'host':config.get('host',os.environ.get('PHASE5_HOST','unreported')),
        'gpu':config.get('gpu',os.environ.get('PHASE5_GPU','unreported')),
        'runtime_image':os.environ.get('PHASE5_RUNTIME_IMAGE', 'unreported'),
        'python_version':platform.python_version()}

    def metrics():
        elapsed = time.monotonic() - started
        return {**provenance, 'phase':'phase5', 'requests':completed, 'requested_count':len(ordered),
            'output_tokens':total_tokens, 'prompt_tokens':prompt_tokens,
            'generation_seconds':generation_seconds, 'setup_seconds':loaded-started,
            'elapsed_seconds':elapsed, 'function_seconds':elapsed,
            'output_tokens_per_second':total_tokens/generation_seconds if generation_seconds else 0.0,
            'manifest':manifest}

    with raw_path.open('x') as out:
        for offset in range(0,len(ordered),chunk_size):
            batch = ordered[offset:offset+chunk_size]
            rendered = [render_prompt(tokenizer,r) for r in batch]
            prompt_ids = [tokenizer.encode(p,add_special_tokens=False) for p in rendered]
            if any(len(ids)+SAMPLING['max_tokens'] > spec['max_model_len'] for ids in prompt_ids):
                raise ValueError('Prompt plus 4096 output tokens exceeds the pinned context')
            params = [SamplingParams(**SAMPLING, seed=r['seed'], skip_special_tokens=False,
                                      stop_token_ids=spec['stop_token_ids']) for r in batch]
            tick = time.monotonic()
            outputs = llm.generate([{'prompt_token_ids':ids} for ids in prompt_ids],params,use_tqdm=True)
            batch_seconds = time.monotonic()-tick
            generation_seconds += batch_seconds
            if len(outputs) != len(batch):
                raise RuntimeError('vLLM returned an incomplete batch')
            for request,output,prompt,expected_ids in zip(batch,outputs,rendered,prompt_ids):
                if len(output.outputs)!=1 or list(output.prompt_token_ids)!=expected_ids:
                    raise RuntimeError('Unexpected sample count or changed token-ID prompt')
                completion=output.outputs[0]
                ids=list(completion.token_ids)
                if (any(type(i) is not int or i<0 for i in ids) or len(ids)>4096
                        or completion.finish_reason not in ('stop','length')):
                    raise RuntimeError('Malformed completion token IDs/count/finish reason')
                response=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                row={**request, **provenance, 'response':response, 'output_token_ids':ids,
                    'raw_text':response, 'raw_token_ids':ids,
                    'decoded_token_segments':decoded_segments(ids,tokenizer,model),
                    'special_token_ids':sorted(set(tokenizer.all_special_ids) | set(spec['special_tokens'].values())),
                    'initial_channel':initial_channel(request), 'finish_reason':completion.finish_reason,
                    'stop_reason':getattr(completion,'stop_reason',None),
                    'rendered_prompt':prompt, 'rendered_prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                    'prompt_token_ids':expected_ids, 'prompt_tokens':len(expected_ids), 'output_tokens':len(ids),
                    'batch_generation_seconds':batch_seconds, 'elapsed_seconds':time.monotonic()-started,
                    'source_sha256':manifest.get('source_sha256',{}),
                    'implementation_sha256':manifest.get('implementation_sha256',{})}
                row.update(split_response(row,tokenizer))
                out.write(json.dumps(row,ensure_ascii=False)+'\n')
                completed+=1;total_tokens+=len(ids);prompt_tokens+=len(expected_ids)
            out.flush()
            # Persist after each completed chunk for controller timing and recovery.
            progress=metrics()
            (directory/f'{slug}.progress.json').write_text(json.dumps(progress,indent=2)+'\n')
            print(json.dumps({k:progress[k] for k in ('model','requests','output_tokens','generation_seconds','elapsed_seconds')}),flush=True)
    stats=metrics()
    with metrics_path.open('x') as out:
        out.write(json.dumps(stats,indent=2)+'\n')
    return stats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,choices=('gemma','nemotron','glm',GEMMA,NEMOTRON,GLM))
    parser.add_argument('--build',type=Path,help='Write request spec only; never load a model')
    parser.add_argument('--requests',type=Path,help='Optional prebuilt spec or request JSONL')
    parser.add_argument('--limit-problems',type=int,default=200)
    parser.add_argument('--pilot',action='store_true',help='Generate exactly 16 balanced rows, then exit')
    parser.add_argument('--run-id',default='phase5-main')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--chunk-size',type=int,default=16)
    args=parser.parse_args()
    if not args.build and not args.pilot:
        parser.error('full-grid execution requires new authorization; use --pilot or --build')
    model=resolve_model(args.model)
    if args.requests:
        if args.requests.suffix=='.jsonl':
            rows=[json.loads(line) for line in args.requests.read_text().splitlines()]
            manifest={}
        else:
            spec=json.loads(args.requests.read_text())
            spec=spec.get(model,spec)
            rows,manifest=spec['requests'],spec['manifest']
    else:
        rows,manifest=build(model,args.limit_problems)
    _validate_requests(rows,model)
    if args.pilot:
        rows=pilot_requests(rows)
        manifest={**manifest,'stage':'pilot','pilot_ids':[r['id'] for r in rows]}
    if args.build:
        args.build.parent.mkdir(parents=True,exist_ok=True)
        with args.build.open('x') as out:
            out.write(json.dumps({'requests':rows,'manifest':manifest},indent=2)+'\n')
        print(json.dumps({'model':model,'requests':len(rows),'path':str(args.build)}))
        return
    config={'batch_size':args.batch_size,'chunk_size':args.chunk_size}
    if args.output_dir is not None:
        config['output_dir']=str(args.output_dir)
    generate(model,rows,args.run_id,manifest,config)


if __name__=='__main__':
    main()
