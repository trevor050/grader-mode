"""Explicit $2.50 single-L40S neutral teacher-forcing control; no generation."""
from __future__ import annotations
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time
import modal
from src.modal_app_3 import image, hf_cache, volume, write_json
from src.modal_app_3_resume import require_remote_fresh, recover_artifacts
from src.models import MODELS
from src.phase3 import MODEL, RATE, summarize_trajectory, system_end_boundary
from src.phase3_neutral import (CAP_SECONDS, CAP_USD, RESERVE, SOURCE, assert_tokens,
    digest, limits, load_inputs, project, replay_positions, require)

app = modal.App('grader-mode-phase3-neutral-control')


@app.function(image=image, gpu='L40S', volumes={'/hf':hf_cache, '/study':volume},
              timeout=int(CAP_SECONDS-RESERVE-120), startup_timeout=120, retries=0,
              single_use_containers=True, min_containers=0, buffer_containers=0,
              max_containers=1, scaledown_window=2, cpu=4, memory=32768)
def track(package:dict, manifest:dict, run_id:str, progress:modal.Dict,
          allocation_start:float, previous_seconds:float, absolute_deadline_unix:float):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.modal_app import render_prompt
    from src.phase3_resume import file_hash
    started = time.monotonic()
    directory = Path('/study')/run_id; directory.mkdir(exist_ok=False)
    require(digest(package) == manifest['payload_sha256'], 'transport payload changed')
    stats = dict(status='running', run_id=run_id, manifest=manifest, gpu='L40S', model=MODEL,
                 revision=MODELS[MODEL], dtype='bfloat16', attention='sdpa', batch_size=4,
                 previous_seconds=previous_seconds, startup_seconds=time.time()-allocation_start,
                 reasoning_tokens=0, n_traces=0, sampling=False, autoregressive_generation=False)
    rows, trajectories, tokens, offsets = [], [], [], [0]
    def check():
        if time.time()+10 >= absolute_deadline_unix:
            raise TimeoutError('Neutral absolute GPU deadline')
    def checkpoint(stage):
        stats.update(stage=stage, function_seconds=time.monotonic()-started,n_traces=len(rows))
        write_json(directory/'neutral-metrics.json',stats)
        progress['state'] = {k:stats[k] for k in ('stage','n_traces','function_seconds')}
        if 'pilot' in stats:progress['state'] = {**progress['state'],'pilot':stats['pilot']}
        volume.commit()
    try:
        import importlib.metadata
        stats['versions'] = {p:importlib.metadata.version(p) for p in ('torch','transformers','numpy','accelerate')}
        for path, expected in manifest['source_hashes'].items():
            if path.startswith('src/'):
                require(file_hash(Path(__file__).parent/Path(path).name) == expected, 'shipped control source changed')
        for path, expected in package['historical_source_hashes'].items():
            if path.startswith('src/'):
                require(file_hash(Path(__file__).parent/Path(path).name) == expected, 'shipped original source changed')
        check(); checkpoint('loading')
        tokenizer = AutoTokenizer.from_pretrained(MODEL,revision=MODELS[MODEL],padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        traces = package['traces']; rendered = {}; neutral = {}
        # Tokenize both full replays before any forward, asserting every saved span.
        audits = {}
        for r in traces:
            rid = r['id']; prompt = render_prompt(tokenizer,r)
            require(hashlib.sha256(prompt.encode()).hexdigest() == r['tutor_prompt_sha256'], 'original prompt changed')
            other = render_prompt(tokenizer,{**r,'messages':[{'role':'system','content':package['neutral_system']},r['messages'][1]]})
            rendered[rid] = prompt; neutral[rid] = other
            spans = []
            for p in (prompt,other):
                encoded = tokenizer(p+r['response'],add_special_tokens=False,return_offsets_mapping=True)
                prompt_ids = tokenizer.encode(p,add_special_tokens=False)
                require(encoded['input_ids'][:len(prompt_ids)] == prompt_ids, 'prompt-response token boundary changed')
                pos = replay_positions(p,r['response'],encoded['offset_mapping'])
                spans.append([encoded['input_ids'][j] for j in pos])
            assert_tokens(spans[0],spans[1],r['saved_tokens'],rid)
            audits[rid] = dict(n_tokens=len(spans[0]), token_sha256=digest(spans[0]), token_identity_verified=True)
        write_json(directory/'token-verification.json',audits)
        checkpoint('tokens_verified');check()
        model = AutoModelForCausalLM.from_pretrained(MODEL,revision=MODELS[MODEL],torch_dtype=torch.bfloat16,
                                                   device_map='cuda',attn_implementation='sdpa').eval()
        hf_cache.commit()
        require((len(model.model.layers),model.config.intermediate_size) == (40,17408), 'model dimensions changed')
        stats['cuda_device'] = torch.cuda.get_device_name()
        specs = {}
        for si,group in enumerate([package['neurons']['tutor'],*package['neurons']['controls']]):
            for li in sorted({r['layer'] for r in group}):
                sub = [r for r in group if r['layer'] == li]
                specs.setdefault(li,[]).append((si,torch.tensor([r['neuron'] for r in sub],device='cuda'),
                    torch.tensor([r['tutor_mean'] for r in sub],device='cuda'),
                    torch.tensor([r['tutor_sd'] for r in sub],device='cuda')))
        def forward(batch, prompts, full):
            check()
            texts = [prompts[r['id']]+(r['response'] if full else '') for r in batch]
            encoded_cpu = tokenizer(texts,padding=True,add_special_tokens=False,return_offsets_mapping=True,return_tensors='pt')
            mapping = encoded_cpu.pop('offset_mapping').tolist()
            encoded = encoded_cpu.to('cuda')
            scores = torch.zeros(len(batch),encoded.input_ids.shape[1],6,device='cuda',dtype=torch.float32)
            handles = []
            for li,subspecs in specs.items():
                def capture(module,args,subspecs=subspecs):
                    for si,indices,means,sds in subspecs:
                        scores[:,:,si] += ((args[0][...,indices].float()-means)/sds).sum(-1)/20
                handles.append(model.model.layers[li].mlp.down_proj.register_forward_pre_hook(capture))
            try:
                with torch.inference_mode():model(**encoded,use_cache=False,logits_to_keep=1)
                torch.cuda.synchronize()
            finally:
                for h in handles:h.remove()
            boundaries = system_end_boundary(encoded.input_ids,encoded.attention_mask,tokenizer.convert_tokens_to_ids('<|im_end|>')).cpu().tolist()
            cpu = scores.cpu().numpy(); ids = encoded.input_ids.cpu().tolist()
            result = []
            for bi,r in enumerate(batch):
                prompt = prompts[r['id']]
                reading = [j for j,(a,b) in enumerate(mapping[bi]) if j>=boundaries[bi] and b>a and b<=len(prompt)]
                require(bool(reading), 'empty post-system reading span')
                positions = replay_positions(prompt,r['response'],mapping[bi]) if full else []
                actual = [ids[bi][j] for j in positions]
                if full:assert_tokens(r['saved_tokens'],actual,r['saved_tokens'],r['id'])
                result.append((cpu[bi,positions,:].copy(), actual, cpu[bi,reading,:].astype(float).mean(0).tolist(),len(reading)))
            return result
        # Worst-length pilot is discarded; final pass preserves original four-row batches.
        longest = sorted(traces,key=lambda r:len(tokenizer.encode(neutral[r['id']]+r['response'],add_special_tokens=False)),reverse=True)[:4]
        tick = time.monotonic()
        forward(longest,neutral,True);forward(longest,rendered,False)
        pilot_seconds = time.monotonic()-tick
        stats['pilot'] = {**project(pilot_seconds,len(traces),time.time()-allocation_start,previous_seconds),
                          'seconds':pilot_seconds,'ids':[r['id'] for r in longest], 'discarded':True}
        checkpoint('pilot');check()
        if not stats['pilot']['fits']:
            raise TimeoutError('Longest-trace pilot projects above $2.50 GPU cap')
        handle = (directory/'neutral-tracking.jsonl').open('x')
        try:
            for offset in range(0,400,4):
                check();batch = traces[offset:offset+4]
                neutral_values = forward(batch,neutral,True)
                tutor_reading = forward(batch,rendered,False)
                for r, (values,ids,read_neutral,n_neutral), (_,_,read_tutor,n_tutor) in zip(batch,neutral_values,tutor_reading):
                    row = {k:r[k] for k in ('id','problem_idx','selection_rank','template_idx','gold','leak','excluded_from_rates')}
                    from src.phase3 import reasoning_span
                    row.update(closed_reasoning=reasoning_span(r['response']) is not None,
                        n_reasoning_tokens=len(ids),token_identity_verified=True,
                        source_response_sha256=hashlib.sha256(r['response'].encode()).hexdigest(),
                        rendered_prompt_sha256=hashlib.sha256(neutral[r['id']].encode()).hexdigest(),
                        tutor_prompt_sha256=r['tutor_prompt_sha256'], reading_tutor=read_tutor,reading_neutral=read_neutral,
                        n_reading_tutor=n_tutor,n_reading_neutral=n_neutral,**summarize_trajectory(values))
                    rows.append(row);trajectories.append(values);tokens.extend(ids);offsets.append(len(tokens))
                    handle.write(json.dumps(row,allow_nan=False)+'\n')
                handle.flush()
                if (offset+4)%40 == 0:checkpoint('tracking')
        finally:
            handle.close()
        np.savez_compressed(directory/'neutral-trajectories.npz', scores=np.concatenate(trajectories),
                            token_ids=np.asarray(tokens,dtype=np.int32), offsets=np.asarray(offsets), ids=np.asarray([r['id'] for r in rows]))
        stats.update(status='completed',reasoning_tokens=len(tokens))
    except BaseException as error:
        stats.update(status='budget_stopped' if isinstance(error,TimeoutError) else 'failed',stop_reason=f'{type(error).__name__}: {error}')
        import traceback
        traceback.print_exc()
    finally:
        checkpoint('finished')
    return {k:stats[k] for k in ('status','function_seconds','n_traces')}


def snapshot(command):
    return json.loads(subprocess.check_output(['uv','run','modal',*command],timeout=15))


@app.local_entrypoint()
def main(run_id:str='phase3-neutral-20261021'):
    from src.build_prompts import ROOT
    out = ROOT/'results/phase3'; raw = ROOT/'results/raw'/run_id
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(run_id.startswith('phase3-neutral-') and run_id.replace('-','').isalnum(), 'unsafe neutral run ID')
        require(not raw.exists(),'fresh allocation ID required')
        ledger = out/'neutral-gpu-attempts.json'
        attempts = json.loads(ledger.read_text()) if ledger.exists() else []
        limits(attempts,time.time())
        package = load_inputs(ROOT)
        require(snapshot(['container','list','--json']) == [],'zero workspace containers required before launch')
        require_remote_fresh(run_id)
        billing = snapshot(['billing', 'summary','--json'])
        write_json(out/'neutral-billing-before.json',billing)
        write_json(out/'neutral-containers-before.json',[])
        manifest = dict(run_id=run_id,source_run_id=SOURCE,cap_gpu_usd=CAP_USD,rate_usd_per_gpu_second=RATE,
                        payload_sha256=digest(package),source_hashes=package['source_hashes'],
                        historical_source_hashes=package['historical_source_hashes'],
                        pricing_source='https://modal.com/pricing',pricing_checked='2026-10-21',
                        registered_addendum_commit='843719c')
        raw.mkdir(parents=True);write_json(raw/'launch-manifest.json',manifest)
        call, zero, caught, recovery_errors = None,False,None,[]
        with modal.Dict.ephemeral() as progress:
            start = time.time();tick = time.monotonic();bounds = limits(attempts,start)
            attempt = dict(run_id=run_id,status='running',started_unix=start,gpu='L40S',**bounds)
            attempts.append(attempt);write_json(ledger,attempts)
            try:
                call = track.with_options(timeout=bounds['timeout_seconds'],retries=0).spawn(
                    package,manifest,run_id,progress,start,bounds['previous_seconds'],bounds['absolute_deadline_unix'])
                attempt['call_id'] = call.object_id;write_json(ledger,attempts);seen = None
                while True:
                    if time.monotonic()-tick >= bounds['exposure_seconds']-RESERVE:
                        raise TimeoutError('Local neutral cap guard')
                    state = progress.get('state')
                    if state and state != seen:
                        write_json(raw/'local-progress.json',state);print('NEUTRAL '+json.dumps(state),flush=True);seen = state
                    try:result = call.get(timeout=0)
                    except TimeoutError:time.sleep(2);continue
                    attempt.update(result);break
            except BaseException as error:
                caught = error
                attempt.update(status='failed_or_cancelled' if call is not None else 'submission_unresolved',error=f'{type(error).__name__}: {error}')
            finally:
                if call is not None:
                    try:call.cancel(terminate_containers=True)
                    except Exception as error:attempt['cancel_error'] = str(error)
                for _ in range(20):
                    try:
                        containers = snapshot(['container','list','--json'])
                        write_json(out/'neutral-containers-after.json',containers)
                        if containers == []:
                            zero = True;attempt['zero_containers_verified_unix'] = time.time();break
                    except Exception as error:attempt['shutdown_error'] = str(error)
                    if time.monotonic()-tick >= bounds['exposure_seconds']-20:break
                    time.sleep(2)
                elapsed = max(time.monotonic()-tick,time.time()-start)
                attempt.update(ended_unix=time.time(),observed_wall_seconds=elapsed,
                               wall_seconds=elapsed if call is not None and zero else bounds['exposure_seconds'])
                if not zero:attempt['status'] = 'shutdown_unresolved'
                write_json(ledger,attempts)
                if call is not None:recovery_errors = recover_artifacts(raw,run_id)
        after = snapshot(['billing', 'summary','--json']);write_json(out/'neutral-billing-after.json',after)
        execution = dict(run_id=run_id,attempts=attempts,gpu_usd_upper=sum(a['wall_seconds'] for a in attempts)*RATE,
                         zero_containers_verified=zero,cap_gpu_usd=CAP_USD,rate_usd_per_gpu_second=RATE,
                         accounting_basis='local submission through verified zero containers, including provisioning and failures',
                         billing_before=billing,billing_after=after,recovery_errors=recovery_errors)
        write_json(out/'neutral-execution.json',execution)
        print(json.dumps(execution),flush=True)
        if caught is not None:raise caught
        require(zero and not recovery_errors and attempt['status'] == 'completed','control incomplete; see execution evidence')
