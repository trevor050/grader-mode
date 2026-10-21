"""Outcome-blind, staged second-family Modal runner. Default is local/dry.

Review first: uv run python -m src.modal_second_family_5 --dry-run
Explicit launch: uv run python -m src.modal_second_family_5 --execute --run-id FRESH
No `modal run`: the CPU controller reserves before app/image setup. This NEW
$2.60 authorization covers all attempts, setup and teardown, with $0.30 held
outside compute. The historical aggregate/Vast ledger belongs to the parent.

Official rates checked 2026-10-21: https://modal.com/pricing
GPU $0.000842/s + CPU4 $0.0000524/s + GiB32 $0.00007104/s.
The rounded $0.001/s bound starts BEFORE app.run (including image setup), ends
AFTER verified stop, and never assumes billing credits/refunds. No region
premium, non-preemptible multiplier, warm pools, SDK retries or GPU fallback.
Provider outages can defeat stop verification; unresolved exposure blocks reuse.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

import modal
from src.models_5 import NEMOTRON, GLM, PACKAGE_VERSIONS, REGISTRATION_COMMIT, model_spec
from src.phase5_control import ROOT, BudgetError, atomic_json, positive, safe_id
from src.modal_app_5 import PINS, hf_cache, results, RESULT_VOLUME, modal_cli, stop_and_verify
from src.phase5_vast_main import (STAGES, cell_id, cell_ids, stage_rows, balanced_pilot,
                                  measurement_summary, timing_measurement, CachedEngine,
                                  runtime_fix_policy, validate_runtime_smoke)

CAP = 2.60
RESERVE = .30
RATE = .001
TEARDOWN_SECONDS = 120
MEASUREMENT_SECONDS = 600
BATCH_SIZE = 256
AUTHORIZATION_ID = 'phase5-second-family-modal-blackwell-260'
AUTH_PATH = ROOT/'results/phase5/modal-second-family-authorization.json'
APP_NAME = 'grader-mode-phase5-second-family'
# The same base/pins as Gemma, with the matching SM120 precompiled kernels.
# Build steps precede runtime source mounts, as required by Modal.
image = (modal.Image.debian_slim(python_version='3.12')
         .uv_pip_install(*PINS)
         .uv_pip_install('cuda-toolkit[nvcc,nvvm,crt,cudart,cccl]==13.0.3','ninja==1.13.0')
         .apt_install('build-essential')
         .env({'CUDA_HOME':'/usr/local/lib/python3.12/site-packages/nvidia/cu13',
               'CUDA_PATH':'/usr/local/lib/python3.12/site-packages/nvidia/cu13',
               'PATH':'/usr/local/lib/python3.12/site-packages/nvidia/cu13/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'})
         .run_commands('test -d "$CUDA_HOME/lib64" || ln -s lib "$CUDA_HOME/lib64"',
             'test -e "$CUDA_HOME/lib/libcudart.so" || ln -s libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"',
             '"$CUDA_HOME/bin/nvcc" --version',
             'python -c "from pathlib import Path; Path(\'/tmp/phase5_cuda_probe.cu\').write_text(\'#include <cuda_runtime.h>\\n__global__ void phase5_probe(float *x) { x[0] = 1.0f; }\\n\')"',
             '"$CUDA_HOME/bin/nvcc" -arch=sm_120 -ptx /tmp/phase5_cuda_probe.cu -o /tmp/phase5_cuda_probe.ptx',
             'test -s /tmp/phase5_cuda_probe.ptx',
             '"$CUDA_HOME/bin/nvcc" -arch=sm_120 --cudart shared -shared -Xcompiler=-fPIC /tmp/phase5_cuda_probe.cu -o /tmp/phase5_cuda_probe.so',
             'test -s /tmp/phase5_cuda_probe.so')
         .run_commands('python -m flashinfer install-cubin-wheel',
             'python -m flashinfer install-jit-cache-wheel --cuda-version 13.0 --mode minimal --sm sm120f')
         .env({'HF_HOME':'/hf','HF_XET_HIGH_PERFORMANCE':'1',
               'PHASE5_HOST':'modal','PHASE5_GPU':'RTX-PRO-6000',
               'PHASE5_RUNTIME_IMAGE':'modal-gemma-pins-vllm0.32.0-cuda13.0.3-nvcc-flashinfer-sm120f',
               'VLLM_USE_FLASHINFER_SAMPLER':'0','DO_NOT_TRACK':'1'})
         .add_local_python_source('src')
         .add_local_dir(ROOT/'prompts',remote_path='/root/prompts')
         .add_local_file(ROOT/'DESIGN.md',remote_path='/root/DESIGN.md'))
app = modal.App(APP_NAME)


def dry_config():
    return dict(execute=False, cap_usd=CAP, reserve_usd=RESERVE,
                authorization_id=AUTHORIZATION_ID, authorization_record=str(AUTH_PATH),
                aggregate_ledger_modified=False, gpu='RTX-PRO-6000',
                cpu=4, memory_mib=32768, rate_bound_usd_per_second=RATE,
                price_source='https://modal.com/pricing', price_checked='2026-10-21',
                stages=dict(smoke=8, throughput=256, h20_core=800,
                            h21_main=1600, h20_controls=800),
                batch_size=256, chunk_size=256, dropped_cells=[],
                models=[NEMOTRON,GLM], quantization='native checkpoint NVFP4; GLM AWQ fallback only',
                image='Gemma base/pins plus torch-matched CUDA13.0.3 compiler and FlashInfer SM120f cache',
                result_volume=RESULT_VOLUME, cache_volume='grader-mode-phase5-hf-cache',
                min_containers=0, buffer_containers=0, max_containers=1, retries=0,
                outcomes_consulted=False, paid_calls=False,
                fallback_gpu_provenance='RTX-PRO-6000; lead-authorized provider switch')


class Authorization:
    """Separate fixed authorization, flocked across controller and watchdog.

    One active reservation at a time. Verified stop settles UP to conservative
    local wall time, including queue/setup/stop. Unverified attempts keep the
    entire reservation. Earlier charged amounts are never reduced or reset.
    """
    def __init__(self, path=AUTH_PATH, *, cap=CAP, authorization_id=AUTHORIZATION_ID):
        self.path = Path(path).resolve()
        self.cap = positive(cap, 'authorization cap')
        self.authorization_id = safe_id(authorization_id)

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix('.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def read(self):
        if not self.path.exists():
            return dict(schema=1, authorization_id=self.authorization_id, cap_usd=self.cap,
                        reserve_usd=RESERVE, rate_usd_per_second=RATE, attempts=[])
        value=json.loads(self.path.read_text())
        if (value.get('schema') != 1 or value.get('authorization_id') != self.authorization_id
                or value.get('cap_usd') != self.cap or value.get('reserve_usd') != RESERVE
                or value.get('rate_usd_per_second') != RATE or not isinstance(value.get('attempts'),list)):
            raise BudgetError('authorization identity/cap/rate changed; refuse reset')
        ids=set()
        for a in value['attempts']:
            if (safe_id(a['attempt_id']) in ids or a['status'] not in ('active','settled','unresolved')
                    or type(a.get('teardown_verified')) is not bool):
                raise BudgetError('malformed authorization attempt')
            ids.add(a['attempt_id'])
            charged=positive(a['charged_usd'],'charged exposure',zero=True)
            reserved=positive(a['reserved_usd'],'reserved exposure')
            start=positive(a['started_unix'],'start')
            work=positive(a['work_deadline_unix'],'work deadline')
            hard=positive(a['hard_deadline_unix'],'hard deadline')
            overrun=positive(a.get('overrun_usd',0),'overrun',zero=True)
            if ((charged > reserved and abs(charged-reserved-overrun)>1e-6) or not start < work < hard
                    or hard-work < TEARDOWN_SECONDS-.001
                    or (a['status']=='settled') != a['teardown_verified']):
                raise BudgetError('invalid attempt exposure/deadlines')
        if (sum(a['charged_usd'] for a in value['attempts']) > self.cap-RESERVE+1e-9
                and not any(a['status']=='unresolved' and a.get('overrun_usd',0)>0 for a in value['attempts'])):
            raise BudgetError('authorization exhausted/overdrawn')
        return value

    def snapshot(self):
        with self.locked():
            return self.read()

    def begin(self, attempt_id, *, now=None):
        safe_id(attempt_id); now=time.time() if now is None else positive(now,'now')
        with self.locked():
            value=self.read()
            if any(a['status'] != 'settled' for a in value['attempts']):
                raise BudgetError('unresolved/active attempt; verified stop required')
            if any(a['attempt_id']==attempt_id for a in value['attempts']):
                raise BudgetError('attempt id cannot be reused')
            prior=sum(a['charged_usd'] for a in value['attempts'])
            seconds=math.floor(round((self.cap-RESERVE-prior)/RATE,6))
            if seconds < TEARDOWN_SECONDS+60:
                raise BudgetError('remaining budget cannot cover setup and teardown reserve')
            ticket=dict(attempt_id=attempt_id, status='active', app_id=None,
                        started_unix=now, work_deadline_unix=now+seconds-TEARDOWN_SECONDS,
                        hard_deadline_unix=now+seconds, prior_charged_usd=prior,
                        reserved_usd=round(seconds*RATE,6), charged_usd=round(seconds*RATE,6),
                        teardown_verified=False)
            value['attempts'].append(ticket)
            atomic_json(self.path,value)
            return deepcopy(ticket)

    def note(self, attempt_id, **fields):
        if set(fields) & {'attempt_id','charged_usd','reserved_usd','status','teardown_verified',
                          'started_unix','work_deadline_unix','hard_deadline_unix','prior_charged_usd'}:
            raise BudgetError('immutable budget fields')
        with self.locked():
            value=self.read()
            ticket=next(a for a in value['attempts'] if a['attempt_id']==attempt_id)
            ticket.update(fields); atomic_json(self.path,value)

    def bind(self, attempt_id, app_id):
        if not isinstance(app_id,str) or not app_id.startswith('ap-'):
            raise ValueError('explicit Modal app ID required before spawn')
        self.note(attempt_id,app_id=app_id)

    def finish(self, attempt_id, verified, *, now=None, evidence=None):
        now=time.time() if now is None else positive(now,'now')
        with self.locked():
            value=self.read(); a=next(a for a in value['attempts'] if a['attempt_id']==attempt_id)
            if a['teardown_verified']:
                return deepcopy(a)
            elapsed=math.ceil(max(0,now-a['started_unix'])*RATE*1e6-1e-8)/1e6
            # If a watchdog already retained a failed reservation, do not refund.
            charge=elapsed if a['status']=='active' else max(elapsed,a['charged_usd'])
            a.update(status='settled' if verified else 'unresolved',teardown_verified=bool(verified),
                     charged_usd=charge if verified else max(elapsed,a['reserved_usd']),
                     ended_unix=now,teardown=evidence)
            if elapsed > a['reserved_usd']:
                a['overrun_usd']=round(a['charged_usd']-a['reserved_usd'],6)
                a['status']='unresolved';a['teardown_verified']=False
            atomic_json(self.path,value)
            return deepcopy(a)


def project_stage(rows, metrics, observed, *, charged_usd, elapsed_seconds, seconds_left):
    """Whole stage projection from timing/lengths only, refreshed after each stage."""
    summary=measurement_summary(metrics)
    means=summary['mean_output_tokens_by_cell'].copy()
    speed=summary['tokens_per_second']
    max_wall_per_row=summary['measurement_wall_seconds']/summary['measured_requests']
    for item in observed:
        n=item['requests']; tokens=item['output_tokens']
        if type(n) is not int or n <= 0 or type(tokens) is not int or tokens <= 0:
            raise ValueError('invalid stage measurement')
        wall=positive(item['wall_seconds'],'stage wall seconds')
        if wall < positive(item['generation_seconds'],'stage generation seconds'):
            raise ValueError('wall time omits generation')
        speed=min(speed,tokens/wall)
        max_wall_per_row=max(max_wall_per_row,wall/n)
        if sum(v['count'] for v in item['per_cell'].values()) != n or sum(
                v['output_tokens'] for v in item['per_cell'].values()) != tokens:
            raise ValueError('inconsistent stage counts')
        for key, cell in item['per_cell'].items():
            if key not in means or type(cell['count']) is not int or cell['count'] <= 0:
                raise ValueError('invalid stage cell')
            mean=positive(cell['output_tokens'],'cell output tokens',zero=True)/cell['count']
            if mean > 4096: raise ValueError('stage exceeds token cap')
            means[key]=max(means[key],mean)
    seconds=1.25*max(sum(means[cell_id(r)] for r in rows)/speed,
                     len(rows)*max_wall_per_row)
    # A further 15 seconds covers permit/commit/control latency per stage.
    seconds+=15
    cost=seconds*RATE
    charged=positive(charged_usd,'prior charges',zero=True)
    elapsed=positive(elapsed_seconds,'elapsed',zero=True)
    available=max(0,CAP-RESERVE-charged-elapsed*RATE)
    left=positive(seconds_left,'seconds left',zero=True)
    fits=cost<=available and seconds<=left
    return dict(decision='run' if fits else 'defer', count=len(rows),
                projected_seconds=seconds,projected_usd=cost,available_usd=available,
                shortfall_usd=max(0,cost-available,(seconds-left)*RATE),
                seconds_left=left, reserve_usd=RESERVE, runtime_margin=1.25,
                measured_tokens_per_second=speed, mean_output_tokens_by_cell=means,
                outcomes_consulted=False,dropped_cells=[])


def failure_kind(phase, text):
    """Only diagnosed checkpoint load/protocol errors spend the two model tries."""
    low=text.lower()
    if 'deadline' in low or 'budget' in low:
        return 'budget_stop'
    if phase not in ('model_load','runtime_smoke'):
        return 'infrastructure_failure' if phase=='infrastructure' else 'execution_failure'
    if any(s in low for s in ('connection','download','http','dns','timed out','timeout',
                             'no space left','permission','driver error','device unavailable',
                             'no cuda','modulenotfound','importerror','pinned package')):
        return 'infrastructure_failure'
    if any(s in low for s in ('out of memory','outofmemoryerror','quantization','unsupported model',
                             'cache blocks','cuda graph','cudagraph','graph capture',
                             'smoke failed','smoke requires','malformed smoke','saved smoke split',
                             'empty completion','checkpoint','model architecture')):
        return 'model_failure'
    return 'infrastructure_failure'  # Unknown is never evidence against a model.


def run_families(candidate, persist_proof, clock=time.time, *, prior_failures=()):
    attempts=[]; fix=None

    def remember(result, stage):
        nonlocal fix
        if (result.get('phase') not in ('model_load','runtime_smoke')
                or result.get('violation_scored') is not False
                or result.get('stage')!=stage or not result.get('evidence')
                or not result.get('attempt_id') or result['ended_unix']<result['started_unix']
                or (attempts and result['started_unix']<attempts[-1]['ended_unix'])):
            raise ValueError('invalid prospective model failure evidence')
        attempts.append({**result,'status':'failed','purpose':'runtime_smoke'})
        fix=result.get('runtime_fix')
    if len(prior_failures)>2:
        raise BudgetError('more than two historical model failures; refuse reset')
    for result,stage in zip(prior_failures,('startup','fix')):
        remember(result,stage)
    for stage in ('startup','fix')[len(attempts):]:
        result=candidate(NEMOTRON,stage,fix,None)
        if result.get('status')!='model_failure':
            return result
        remember(result,stage)
    proof=dict(kind='runtime_startup_failure', model=NEMOTRON,
               model_revision=model_spec(NEMOTRON)['revision'], registration_commit=REGISTRATION_COMMIT,
               gpu='RTX-PRO-6000',attempts=attempts,selected_before_generation=True,
               outcomes_consulted=False,decision_unix=clock())
    if proof['decision_unix']<attempts[-1]['ended_unix']:
        raise ValueError('fallback decision predates failures')
    persist_proof(proof)
    return candidate(GLM,'fallback',None,proof)


def publish_main(directory, rows, manifest, completed, deferred):
    """Publish ONLY whole completed main stages; partial/pilot files stay separate."""
    directory=Path(directory); stages=stage_rows(rows)
    if list(completed)!=list(STAGES[:len(completed)]) or list(deferred)!=list(STAGES[len(completed):]):
        raise ValueError('only a completed prefix and deferred suffix are permitted')
    target=directory/'main';target.mkdir(exist_ok=True)
    slug=rows[0]['model'].split('/')[-1]; selected=[]; seen=set(); run_ids=set()
    for name in completed:
        expected={r['id']:r for r in stages[name]}
        source=directory/name/(slug+'.jsonl')
        batch=[json.loads(s) for s in source.read_text().splitlines()]
        if len(batch)!=len(expected) or {r['id'] for r in batch}!=set(expected):
            raise ValueError('completed stage identity mismatch')
        for row in batch:
            if (row['id'] in seen or row.get('stage')!='main' or row.get('pilot') is not False
                    or any(row.get(k)!=v for k,v in expected[row['id']].items())):
                raise ValueError('pilot/duplicate/changed main identity')
            seen.add(row['id']);run_ids.add(row['run_id']);selected.append(row)
    if len(run_ids)>1: raise ValueError('main stages must share one run id')
    temporary=target/'generations.jsonl.tmp'
    with temporary.open('w') as out:
        for row in selected: out.write(json.dumps(row,ensure_ascii=False)+'\n')
        out.flush();os.fsync(out.fileno())
    temporary.replace(target/'generations.jsonl')
    atomic_json(target/'manifest.json',{**manifest,'stage':'main','requests':3200,
        'completed_stages':list(completed),'deferred_stages':list(deferred), 'dropped_cells':[],
        'completed_rows':len(selected),'complete_registered_grid':len(selected)==3200,
        'pilot_rows_excluded_from_main':True,'inference_eligible':False,'audit_pending':True})
    return dict(jsonl=str(target/'generations.jsonl'),manifest=str(target/'manifest.json'),rows=len(selected))


def source_hashes():
    names=['DESIGN.md','prompts/templates.json','prompts/templates_4.json','prompts/system_tutor.txt',
           'src/models_5.py','src/phase5_split.py','src/vast_generate_5.py',
           'src/phase5_control.py','src/modal_app_5.py','src/phase5_vast_main.py',
           'src/modal_second_family_5.py']
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def worker_session(payload, channel, *, checkpoint, cache_commit, generate_fn=None):
    """Killable child, Dict permits; one LLM across smoke/measurement/main stages."""
    from src import vast_generate_5 as generator
    directory=Path(payload['directory']); rows=payload['requests']; stages=stage_rows(rows)
    manifest=deepcopy(payload['manifest']);model=payload['model'];completed=[]
    controls_only=payload.get('controls_only',False)
    if type(controls_only) is not bool or (controls_only and (
            model!=NEMOTRON or payload['resident_sequences']!=256 or payload['enforce_eager'] is not False)):
        raise ValueError('controls continuation requires unchanged full-batch Nemotron engine')
    requested_stages=('h20_controls',) if controls_only else STAGES
    manifest.update(stage='main',execution_source_sha256=payload['source_hashes'],
                    execution_authorization_id=payload.get('authorization_id',AUTHORIZATION_ID),fallback_provenance=payload.get('fallback_provenance'))
    manifest['implementation_sha256'].update({k:v for k,v in payload['source_hashes'].items() if k.startswith('src/')})
    state=dict(status='running',phase='infrastructure',stage=payload['attempt_stage'],
               attempt_id=payload['attempt_id'],started_unix=time.time(),violation_scored=False,
               completed_stages=[],deferred_stages=list(requested_stages),dropped_cells=[])
    phase='infrastructure'; stage_deadline=payload['work_deadline_unix'];original_llm=None;engine=None

    def expired():
        if time.time() >= min(stage_deadline,payload['work_deadline_unix'])-10:
            raise TimeoutError('remote absolute work/stage deadline')

    def save_state():
        atomic_json(directory/'worker-state.json',state);checkpoint()
        channel['state']=deepcopy(state)

    def mark_phase(value):
        nonlocal phase
        phase=value;state['phase']=value;save_state()

    def generate(name, selected):
        nonlocal phase
        expired(); mark_phase('runtime_smoke' if name=='smoke' else name)
        tick=time.monotonic()
        stage='pilot' if name in ('smoke','throughput') else 'main'
        stats=(generate_fn or generator.generate)(model,selected,
            payload['attempt_id']+('-main' if stage=='main' else '-'+name),
            {**manifest,'stage':stage,'controller_stage':name,'pilot_rows_excluded_from_main':True},
            dict(output_dir=str(directory/name),batch_size=payload['resident_sequences'],chunk_size=256,
                 enforce_eager=payload['enforce_eager'],before_chunk=expired,checkpoint=checkpoint,
                 host='modal',gpu='RTX-PRO-6000'))
        expired()
        if 'first_generation_unix' not in state and 'generation_started_unix' in stats:
            state['first_generation_unix']=positive(stats['generation_started_unix'],'actual generation start')
            if model==GLM:
                manifest['fallback_provenance']['first_generation_unix']=state['first_generation_unix']
        if stats['requests']!=len(selected):
            raise RuntimeError('incomplete stage; no inference permitted')
        cells={}
        for key,item in stats['per_cell'].items():
            suffix=next((c for c in cell_ids() if key.endswith('/'+c)),None)
            if suffix is None: raise ValueError('unknown timing cell')
            cells[suffix]=item
        if Counter({k:v['count'] for k,v in cells.items()}) != Counter(cell_id(r) for r in selected):
            raise RuntimeError('completed stage cell count mismatch')
        timing=timing_measurement({**stats,'per_cell':cells},wall_seconds=time.monotonic()-tick,
                                 name=name,resident_sequences=payload['resident_sequences'])
        if name=='smoke':
            mark_phase('runtime_smoke')
            timing['protocol']=validate_runtime_smoke(directory/name,selected)
            cache_commit()  # Weights persist even if later stages are deferred.
        atomic_json(directory/(name+'.done.json'),timing);checkpoint()
        return timing

    try:
        if payload['resident_sequences'] not in (128,256) or type(payload['enforce_eager']) is not bool:
            raise ValueError('invalid fixed engine repair')
        save_state()
        if generate_fn is None:
            versions=generator.runtime_versions()
            if any(versions[k]!=v for k,v in PACKAGE_VERSIONS.items()):
                raise ValueError('pinned package version mismatch')
            import torch
            gpu_name=torch.cuda.get_device_name(0)
            capability=torch.cuda.get_device_capability(0)
            if 'RTX PRO 6000' not in gpu_name or capability[0]<10:
                raise RuntimeError('Blackwell RTX PRO 6000 hardware required')
            state['hardware']=dict(name=gpu_name,compute_capability=list(capability))
            import vllm
            original_llm=vllm.LLM
            def factory(**options):
                mark_phase('model_load')
                loaded=original_llm(**options)
                if model==NEMOTRON:
                    actual=loaded.llm_engine.model_config.quantization
                    if actual not in ('modelopt','modelopt_fp4','modelopt_mixed'):
                        raise ValueError('native NVFP4 checkpoint quantization was not selected')
                    state['native_quantization']=actual
                mark_phase('runtime_smoke')
                return loaded
            engine=CachedEngine(factory);vllm.LLM=engine
        timing=generate('smoke',balanced_pilot(rows,8))
        channel['done:smoke']=timing
        for name in (('h20_controls',) if controls_only else ('throughput',*STAGES)):
            mark_phase('await_'+name)
            while True:
                expired()
                command=channel.get('command:'+name)
                if command is not None: break
                if channel.get('stop'):
                    state['status']='deferred';return state
                time.sleep(.2)
            expected=balanced_pilot(rows,256) if name=='throughput' else stages[name]
            if (command.get('attempt_id')!=payload['attempt_id'] or command.get('action')!='run'
                    or command.get('count')!=len(expected)
                    or command.get('work_deadline_unix')!=payload['work_deadline_unix']):
                raise BudgetError('invalid whole-stage permit')
            stage_deadline=positive(command['stage_deadline_unix'],'stage deadline')
            if not time.time()<stage_deadline<=payload['work_deadline_unix']:
                raise BudgetError('invalid stage absolute deadline')
            timing=generate(name,expected)
            if name in STAGES:
                completed.append(name); state['completed_stages']=completed.copy()
                state['deferred_stages']=list(requested_stages[len(completed):])
                if controls_only:
                    from src.phase5_vast_main import validate_pulled_stage
                    validate_pulled_stage(directory/name,expected)
                else:
                    publish_main(directory,rows,manifest,completed,state['deferred_stages'])
                checkpoint()
            channel['done:'+name]=timing
            stage_deadline=payload['work_deadline_unix']
        state['status']='completed'
    except BaseException:
        text=traceback.format_exc()
        (directory/'runtime-trace.txt').write_text(text)
        state.update(status=failure_kind(phase,text),phase=phase,
                     runtime_fix=runtime_fix_policy(text),evidence=str(directory/'runtime-trace.txt'))
    finally:
        state['ended_unix']=time.time()
        state['fallback_provenance']=manifest.get('fallback_provenance')
        # Main publication is repeated on stop so deferred stages remain explicit.
        try:
            if not controls_only:
                publish_main(directory,rows,manifest,completed,list(STAGES[len(completed):]))
            cache_commit()
            save_state()
        finally:
            if original_llm is not None:
                import vllm
                vllm.LLM=original_llm
    return state


@app.function(image=image, gpu='RTX-PRO-6000', volumes={'/hf':hf_cache,'/phase5':results},
              cpu=(4,4),memory=(32768,32768),timeout=2300,startup_timeout=180,
              min_containers=0,buffer_containers=0,max_containers=1,retries=0,
              single_use_containers=True,scaledown_window=2)
def second_family_gpu(payload:dict, channel:modal.Dict):
    return supervise_worker(payload,channel)


def supervise_worker(payload:dict, channel:modal.Dict):
    """GPU supervisor kills the entire child process group, then commits evidence."""
    directory=Path('/phase5')/safe_id(payload['attempt_id'])
    directory.mkdir(exist_ok=False)
    payload={**payload,'directory':str(directory)}
    atomic_json(directory/'input.json',payload);results.commit()
    hard_stop=threading.Timer(max(.01,payload['work_deadline_unix']-time.time()+5),lambda:os._exit(124))
    hard_stop.daemon=True;hard_stop.start()
    process=None
    state=dict(status='infrastructure_failure',attempt_id=payload['attempt_id'])
    channel['remote_started']=True
    try:
        with (directory/'generation.log').open('xb') as log:
            process=subprocess.Popen([sys.executable,'-m','src.modal_second_family_5','--worker',
                str(directory/'input.json'),'--channel',channel.object_id],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            while process.poll() is None:
                if time.time()>=payload['work_deadline_unix']-5:
                    raise TimeoutError('remote supervisor absolute deadline')
                time.sleep(.5)
            path=directory/'worker-state.json'
            if path.exists(): state=json.loads(path.read_text())
            if process.returncode and state.get('status') in ('completed','running','deferred'):
                state.update(status='infrastructure_failure',returncode=process.returncode)
    except BaseException as error:
        state.update(status='budget_stop' if isinstance(error,TimeoutError) else 'infrastructure_failure',
                     error=f'{type(error).__name__}: {error}')
    finally:
        try:
            if process is not None and process.poll() is None:
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait(timeout=8)
        finally:
            state['ended_unix']=time.time()
            atomic_json(directory/'remote-execution.json',state)
            results.commit();channel['remote_result']=state;hard_stop.cancel()
    return state


def watchdog(attempt_id):
    ledger=Authorization()
    while True:
        a=next(t for t in ledger.snapshot()['attempts'] if t['attempt_id']==attempt_id)
        if a['teardown_verified']:return
        if time.time()<a['work_deadline_unix']:
            time.sleep(min(1,a['work_deadline_unix']-time.time()));continue
        # No GPU invocation is permitted before bind. If image/app setup hung
        # before bind, discover only this app name, never stop unrelated work.
        ids=[a['app_id']] if a.get('app_id') else []
        if not ids:
            try:
                listed=json.loads(modal_cli('app','list','--json',timeout=5))
                ids=[row['app_id'] for row in listed if row.get('state') not in ('stopped','disabled')
                     and (row.get('name')==APP_NAME or row.get('description')==APP_NAME)]
            except Exception: pass
        evidence=[];zero=bool(ids)
        for app_id in ids:
            ok,detail=stop_and_verify(app_id);zero=zero and ok;evidence.append(detail)
        ledger.finish(attempt_id,zero,evidence=evidence)
        if zero or time.time()>=a['hard_deadline_unix']:return
        time.sleep(1)


def wait_done(channel,call,name,deadline):
    while time.time()<deadline-10:
        done=channel.get('done:'+name)
        if done is not None:return done
        state=channel.get('state') or {}
        if state.get('status') not in (None,'running'):
            return None
        try:
            call.get(timeout=0)
            return channel.get('done:'+name)
        except TimeoutError:
            time.sleep(.5)
    raise TimeoutError('local absolute stage deadline')


def control_stages(payload,channel,call,record):
    """CPU decision loop. Only explicitly whitelisted timing metadata enters gates."""
    smoke=wait_done(channel,call,'smoke',payload['work_deadline_unix'])
    if smoke is None:return
    now=time.time()
    if now+MEASUREMENT_SECONDS+20>payload['work_deadline_unix']:
        record(dict(stage='throughput',decision='defer',deferred_stages=list(STAGES),
                    shortfall_usd=max(0,(now+MEASUREMENT_SECONDS+20-payload['work_deadline_unix'])*RATE),
                    dropped_cells=[],outcomes_consulted=False));channel['stop']=True;return
    tick=time.monotonic()
    channel['command:throughput']=dict(action='run',attempt_id=payload['attempt_id'],count=256,
        work_deadline_unix=payload['work_deadline_unix'],stage_deadline_unix=now+MEASUREMENT_SECONDS)
    metrics=wait_done(channel,call,'throughput',now+MEASUREMENT_SECONDS)
    if metrics is None:return
    metrics={**metrics,'wall_seconds':max(metrics['wall_seconds'],time.monotonic()-tick)}
    measurement_summary(metrics);observed=[]
    for index,(name,rows) in enumerate(stage_rows(payload['requests']).items()):
        now=time.time()
        decision=project_stage(rows,metrics,observed,charged_usd=payload['prior_charged_usd'],
            elapsed_seconds=now-payload['started_unix'],seconds_left=max(0,payload['work_deadline_unix']-now-20))
        record(dict(stage=name,**decision,deferred_stages=list(STAGES[index:]) if decision['decision']=='defer' else []))
        if decision['decision']!='run':
            channel['stop']=True;return
        # Permit the complete stage. The absolute budget watchdog remains active
        # within generation; unexpected overruns yield partial NON-main evidence.
        tick=time.monotonic()
        channel['command:'+name]=dict(action='run',attempt_id=payload['attempt_id'],count=len(rows),
            work_deadline_unix=payload['work_deadline_unix'],stage_deadline_unix=payload['work_deadline_unix'])
        timing=wait_done(channel,call,name,payload['work_deadline_unix'])
        if timing is None:return
        # Includes publication, Volume commits and control-channel latency,
        # rather than treating GPU generate() time as the entire paid stage.
        observed.append({**timing,'wall_seconds':max(timing['wall_seconds'],time.monotonic()-tick)})


def execute(run_id):
    """Explicit paid path, called only after parent review and authorization."""
    safe_id(run_id);ledger=Authorization()
    root=ROOT/'results/raw'/run_id
    root.mkdir(parents=True,exist_ok=False)
    frozen=source_hashes()
    # Reject unrelated active Modal resources; parent owns legacy Vast cleanup.
    apps=json.loads(modal_cli('app','list','--json'))
    containers=json.loads(modal_cli('container','list','--json'))
    if not isinstance(apps,list) or not isinstance(containers,list) or containers or any(
            a.get('state') not in ('stopped','disabled') for a in apps):
        raise BudgetError('zero active Modal apps/containers required')
    from src.vast_generate_5 import build
    grids={NEMOTRON:build('nemotron',200),GLM:build('glm',200)}
    for rows,_ in grids.values():stage_rows(rows)
    if frozen!=source_hashes():raise BudgetError('inputs changed during CPU preflight')
    prior=ledger.snapshot()['attempts']
    if any(a.get('model') and not a.get('outcome_status') for a in prior):
        raise BudgetError('historical attempt outcome unresolved; do not reset model failures')
    prior_failures=[a['model_failure_record'] for a in prior
                    if a.get('model')==NEMOTRON and a.get('outcome_status')=='model_failure']
    proofs=[]

    def persist(proof):
        proofs.append(proof);atomic_json(root/'fallback-proof.json',proof)

    def candidate(model,stage,fix,proof):
        attempt_id=safe_id(run_id+'-'+stage)
        sdk_id_before=app.app_id
        known_ids={a['app_id'] for a in apps} | {
            a['app_id'] for a in ledger.snapshot()['attempts'] if a.get('app_id')}
        ticket=ledger.begin(attempt_id)
        out=root/attempt_id;out.mkdir()
        payload=dict(ticket,attempt_stage=stage,model=model,requests=grids[model][0],
            manifest=grids[model][1],source_hashes=frozen,fallback_provenance=proof,
            resident_sequences=(fix or {}).get('resident_sequences',256),
            enforce_eager=(fix or {}).get('enforce_eager',False))
        atomic_json(out/'launch.json',{**payload,'requests_count':len(payload['requests'])})
        call=None;app_id=None;zero=False;result=None;error=None;gates=[];teardown=None
        with (out/'watchdog.log').open('ab') as log:
            guard=subprocess.Popen([sys.executable,'-m','src.modal_second_family_5','--watchdog',attempt_id],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,close_fds=True)
        ledger.note(attempt_id,watchdog_pid=guard.pid,model=model,stage=stage,source_hashes=frozen)
        try:
            with modal.enable_output(),app.run(name=APP_NAME):
                app_id=app.app_id;ledger.bind(attempt_id,app_id)
                try:
                    if time.time()>=ticket['work_deadline_unix']-60:
                        raise TimeoutError('setup consumed absolute budget deadline')
                    with modal.Dict.ephemeral() as channel:
                        call=second_family_gpu.spawn(payload,channel)
                        ledger.note(attempt_id,function_call_id=call.object_id)
                        def record(value):
                            gates.append({**value,'decided_unix':time.time()})
                            atomic_json(out/'stage-gates.json',gates)
                            print('MODAL_SECOND_FAMILY_GATE '+json.dumps(gates[-1]),flush=True)
                        control_stages(payload,channel,call,record)
                        # Defer/stop still lets the worker commit its final manifest.
                        while time.time()<ticket['work_deadline_unix']-5:
                            try:
                                result=call.get(timeout=0);break
                            except TimeoutError:time.sleep(.5)
                        if result is None:raise TimeoutError('local absolute work deadline')
                finally:
                    if call is not None:
                        try:call.cancel(terminate_containers=True)
                        except Exception as exc:ledger.note(attempt_id,cancel_error=str(exc))
                    zero,teardown=stop_and_verify(app_id)
                    ledger.finish(attempt_id,zero,evidence=teardown)
        except BaseException as exc:
            error=f'{type(exc).__name__}: {exc}'
        finally:
            # app.run allocates an app before image setup, but yields only after
            # setup succeeds. Recover that allocation when __enter__ raises.
            if app_id is None:
                sdk_id=app.app_id
                app_id=sdk_id if sdk_id and sdk_id!=sdk_id_before and sdk_id not in known_ids else None
                if app_id is None:
                    try:
                        allocated=[a['app_id'] for a in json.loads(modal_cli('app','list','--json'))
                            if a['app_id'] not in known_ids and
                            (a.get('description')==APP_NAME or a.get('name')==APP_NAME)]
                        if len(allocated)==1:app_id=allocated[0]
                    except Exception as exc:
                        error=error or f'app allocation reconciliation failed: {exc}'
                if app_id:ledger.bind(attempt_id,app_id)
            if app_id and not zero:
                zero,teardown=stop_and_verify(app_id)
            settled=ledger.finish(attempt_id,zero,evidence=teardown)
            # Download AFTER stop so retrieval is not billed idle GPU time.
            try:
                modal_cli('volume','get',RESULT_VOLUME,attempt_id,str(out),'--force',timeout=90)
                pulled=True
            except Exception as exc:
                pulled=False;error=error or f'artifact download failed: {exc}'
            if frozen!=source_hashes():error=error or 'frozen inputs changed during execution'
            summary=dict(result or {},attempt_id=attempt_id,model=model,stage=stage,
                status=(result or {}).get('status','budget_stop' if error and 'deadline' in error else 'infrastructure_failure'),
                started_unix=ticket['started_unix'],ended_unix=time.time(),
                zero_containers_verified=zero,artifacts_pulled=pulled,teardown=teardown,
                authorization_attempt=settled,controller_error=error,
                evidence=str(out/'local-execution.json'),violation_scored=False,inference_eligible=False)
            if not pulled or error:
                # Uncertain infrastructure evidence must never select GLM.
                summary['status']='infrastructure_failure'
            ledger.note(attempt_id,outcome_status=summary['status'],
                        model_failure_record=summary if summary['status']=='model_failure' else None)
            atomic_json(out/'local-execution.json',summary)
        if not zero or settled['status']!='settled':
            raise BudgetError('Modal teardown/exposure unresolved; reservation retained')
        if summary.get('fallback_provenance',{}):
            atomic_json(root/'fallback-proof.json',summary['fallback_provenance'])
        return summary

    def terminate(signum,frame):raise KeyboardInterrupt('controller terminated; stop owned Modal app')
    previous=signal.signal(signal.SIGTERM,terminate)
    try:
        outcome=run_families(candidate,persist,prior_failures=prior_failures)
        atomic_json(root/'sequence-execution.json',outcome)
        return outcome
    finally:
        signal.signal(signal.SIGTERM,previous)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run',action='store_true')
    modes.add_argument('--execute',action='store_true')
    modes.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    modes.add_argument('--watchdog',help=argparse.SUPPRESS)
    parser.add_argument('--channel',help=argparse.SUPPRESS)
    parser.add_argument('--run-id')
    args=parser.parse_args(argv)
    if args.worker:
        if not args.channel:parser.error('worker needs Dict channel')
        payload=json.loads(args.worker.read_text())
        worker_session(payload,modal.Dict.from_id(args.channel),checkpoint=results.commit,cache_commit=hf_cache.commit)
        return 0
    if args.watchdog:watchdog(safe_id(args.watchdog));return 0
    if args.execute:
        if not args.run_id:parser.error('--execute requires fresh --run-id')
        print(json.dumps(execute(args.run_id),indent=2));return 0
    print(json.dumps(dry_config(),indent=2));return 0


if __name__=='__main__':
    main()
