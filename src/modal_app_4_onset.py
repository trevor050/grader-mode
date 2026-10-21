"""Authorized $1.50 aggregate Modal onset check, one ephemeral L40S, no generation."""
from __future__ import annotations
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import modal
from src.modal_app_3 import image, hf_cache, write_json
from src.models import MODELS
from src.phase3 import MODEL
from src.phase3_resume import digest, file_hash, require
from src.analyze_4_onset import (ALL_IN_RATE,CAP_USD,GPU_RATE,RESERVE,SOURCE,
                                budget_limits,load_inputs,onset_positions)

app=modal.App('grader-mode-phase4-onset')
volume=modal.Volume.from_name('grader-mode-phase4-onset-results',create_if_missing=True)


@app.function(image=image,gpu='L40S',volumes={'/hf':hf_cache,'/onset':volume},
              timeout=int(CAP_USD/ALL_IN_RATE-RESERVE-120),startup_timeout=120,retries=0,
              single_use_containers=True,min_containers=0,buffer_containers=0,max_containers=1,
              scaledown_window=2,cpu=(4,4),memory=(32768,32768))
def track(package:dict,manifest:dict,run_id:str,progress:modal.Dict,allocation_start:float,deadline:float):
    import os
    import threading
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from src.modal_app import render_prompt
    started=time.monotonic()
    # Independent hard stop also covers a stalled CUDA call or model load.
    watchdog=threading.Timer(max(.01,deadline-time.time()),lambda:os._exit(124))
    watchdog.daemon=True;watchdog.start()
    directory=Path('/onset')/run_id;directory.mkdir(exist_ok=False)
    require(digest(package)==manifest['payload_sha256'],'transport package changed')
    stats=dict(status='running',run_id=run_id,manifest=manifest,gpu='L40S',n_traces=0,
               model=MODEL,revision=MODELS[MODEL],dtype='bfloat16',attention='sdpa',batch_size=4,
               sampling=False,autoregressive_generation=False)
    def check():
        if time.time()+15>=deadline:raise TimeoutError('Absolute aggregate Modal deadline')
    def checkpoint(stage):
        stats.update(stage=stage,function_seconds=time.monotonic()-started,allocation_seconds=time.time()-allocation_start)
        write_json(directory/'onset-metrics.json',stats);volume.commit()
        progress['state']={k:stats[k] for k in ('stage','function_seconds','allocation_seconds','n_traces')}
        print('ONSET '+json.dumps(progress['state']),flush=True)
    try:
        for path,expected in {**package['historical_source_hashes'],**package['source_hashes']}.items():
            if path.startswith('src/'):
                require(file_hash(Path(__file__).parent/Path(path).name)==expected,f'shipped source changed: {path}')
        check();checkpoint('loading')
        tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODELS[MODEL],padding_side='left')
        tokenizer.pad_token=tokenizer.eos_token
        traces=package['traces'];sequences=[];audits=[]
        # Prepare exactly the full replay's token IDs, then slice at last retained token.
        # Causal attention makes future teacher-forced tokens unnecessary.
        for r in traces:
            arms=[];rid=r['id']
            for arm in ('tutor','neutral'):
                row=r if arm=='tutor' else {**r,'messages':[{'role':'system','content':package['neutral_system']},r['messages'][1]]}
                prompt=render_prompt(tokenizer,row)
                if arm=='tutor':require(hashlib.sha256(prompt.encode()).hexdigest()==r['tutor_prompt_sha256'],'tutor prompt changed')
                full=tokenizer(prompt+r['response'],add_special_tokens=False,return_offsets_mapping=True)
                prompt_ids=tokenizer.encode(prompt,add_special_tokens=False)
                require(full['input_ids'][:len(prompt_ids)]==prompt_ids,'prompt-response prefix changed')
                selection=onset_positions(prompt,r['response'],full['offset_mapping'])
                actual=[full['input_ids'][j] for j in selection['all_reasoning']]
                require(actual==r['saved_tokens'],f'full reasoning token identity failed: {rid}/{arm}')
                positions=selection['boundary']+selection['reasoning']
                stop=max(positions)+1
                arms.append(dict(input_ids=full['input_ids'][:stop],positions=positions,
                                 closed_reasoning=selection['closed_reasoning'],n_reasoning=len(actual),
                                 prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
            require([j>=0 for j in arms[0]['positions']]==[j>=0 for j in arms[1]['positions']], 'paired availability changed')
            sequences.append(arms)
            audits.append({k:r[k] for k in ('id','problem_idx','selection_rank','template_idx','gold','leak','excluded_from_rates')} |
                          dict(token_identity_verified=True,closed_reasoning=arms[0]['closed_reasoning'],
                               n_reasoning_tokens=arms[0]['n_reasoning'],source_response_sha256=hashlib.sha256(r['response'].encode()).hexdigest(),
                               full_reasoning_token_sha256=digest(r['saved_tokens']),
                               tutor_prompt_sha256=arms[0]['prompt_sha256'],neutral_prompt_sha256=arms[1]['prompt_sha256']))
        write_json(directory/'onset-traces.json',audits);checkpoint('tokens_verified');check()
        model=AutoModelForCausalLM.from_pretrained(MODEL,revision=MODELS[MODEL],torch_dtype=torch.bfloat16,
                                                 device_map='cuda',attn_implementation='sdpa').eval()
        require((len(model.model.layers),model.config.intermediate_size)==(40,17408),'model dimensions changed')
        stats['cuda_device']=torch.cuda.get_device_name()
        import importlib.metadata
        stats['versions']={p:importlib.metadata.version(p) for p in ('torch','transformers','numpy','accelerate')}
        specs={}
        for si,group in enumerate([package['neurons']['tutor'],*package['neurons']['controls']]):
            require(len(group)==20,'20 neurons per cached set required')
            for li in sorted({r['layer'] for r in group}):
                sub=[r for r in group if r['layer']==li]
                specs.setdefault(li,[]).append((si,torch.tensor([r['neuron'] for r in sub],device='cuda'),
                    torch.tensor([r['tutor_mean'] for r in sub],device='cuda'),torch.tensor([r['tutor_sd'] for r in sub],device='cuda')))
        def forward(indices,arm):
            check();batch=[sequences[i][arm] for i in indices]
            encoded=tokenizer.pad([{'input_ids':r['input_ids'],'attention_mask':[1]*len(r['input_ids'])} for r in batch],
                                  padding=True,return_tensors='pt').to('cuda')
            scores=torch.zeros(len(batch),encoded.input_ids.shape[1],6,device='cuda',dtype=torch.float32)
            handles=[]
            for li,subspecs in specs.items():
                def capture(module,args,subspecs=subspecs):
                    for si,ids,means,sds in subspecs:
                        scores[:,:,si]+=((args[0][...,ids].float()-means)/sds).sum(-1)/20
                handles.append(model.model.layers[li].mlp.down_proj.register_forward_pre_hook(capture))
            try:
                with torch.inference_mode():model(**encoded,use_cache=False,logits_to_keep=1)
                torch.cuda.synchronize()
            finally:
                for handle in handles:handle.remove()
            cpu=scores.cpu().numpy();result=[]
            for bi,r in enumerate(batch):
                values=np.zeros((385,6),dtype=np.float32);ids=np.full(385,-1,dtype=np.int32)
                mask=np.asarray(r['positions'])>=0;pad=encoded.input_ids.shape[1]-len(r['input_ids'])
                selected=np.asarray(r['positions'])[mask]
                values[mask]=cpu[bi,selected+pad,:]
                ids[mask]=np.asarray(r['input_ids'])[selected]
                result.append((values,ids,mask))
            return result
        longest=sorted(range(400),key=lambda i:max(len(a['input_ids']) for a in sequences[i]),reverse=True)[:4]
        tick=time.monotonic();forward(longest,0);forward(longest,1)
        pilot=time.monotonic()-tick
        projected=(time.time()-allocation_start+pilot*100*1.3+RESERVE+manifest['previous_seconds'])*ALL_IN_RATE
        stats['pilot']=dict(seconds=pilot,ids=[traces[i]['id'] for i in longest],discarded=True,
                            headroom=1.3,projected_aggregate_modal_usd=projected)
        checkpoint('timing_gate');require(projected<=CAP_USD,'worst-length timing projection exceeds cap')
        tutor=np.zeros((400,385,6),dtype=np.float32);neutral=np.zeros_like(tutor)
        tutor_ids=np.full((400,385),-1,dtype=np.int32);neutral_ids=tutor_ids.copy();mask=np.zeros((400,385),dtype=bool)
        for offset in range(0,400,4):
            check();indices=list(range(offset,offset+4))
            t=forward(indices,0);n=forward(indices,1)
            for i,tv,nv in zip(indices,t,n):
                require(np.array_equal(tv[2],nv[2]),'paired masks changed')
                tutor[i],tutor_ids[i],mask[i]=tv;neutral[i],neutral_ids[i],_=nv
            stats['n_traces']=offset+4
            if (offset+4)%40==0:checkpoint('tracking')
        archive=dict(tutor=tutor,neutral=neutral,contrast=tutor-neutral,mask=mask,
                     tutor_token_ids=tutor_ids,neutral_token_ids=neutral_ids,ids=np.asarray([r['id'] for r in traces]))
        from src.analyze_4_onset import validate_archive
        validate_archive(archive)
        np.savez_compressed(directory/'onset-tokens.npz',**archive)
        stats.update(status='completed',archive_sha256=file_hash(directory/'onset-tokens.npz'))
    except BaseException as error:
        stats.update(status='budget_stopped' if isinstance(error,TimeoutError) else 'failed',error=f'{type(error).__name__}: {error}')
        import traceback
        traceback.print_exc()
    finally:
        checkpoint('finished')
        # Keep watchdog alive through remote return; single-use container terminates.
    return {k:stats[k] for k in ('status','n_traces','function_seconds')}


def snapshot(command):
    result=subprocess.run(['uv','run','modal',*command],capture_output=True,text=True,timeout=15)
    require(result.returncode==0,result.stderr or result.stdout)
    return json.loads(result.stdout)


def billing_snapshot(path):
    """Preserve literal CLI failure and fall back to the supported billing API."""
    try:
        value=dict(source='uv run modal billing summary --json',captured_unix=time.time(),summary=snapshot(['billing','summary','--json']))
    except Exception as error:
        from datetime import datetime,timedelta,timezone
        from modal.billing import workspace_billing_report
        now=datetime.now(timezone.utc)
        report=workspace_billing_report(start=now.replace(hour=0,minute=0,second=0,microsecond=0)-timedelta(days=1),end=now,resolution='h')
        value=dict(source='modal.billing.workspace_billing_report',captured_unix=time.time(),
                   literal_cli_error=str(error),report=report)
    # Decimal/datetime values in supported API snapshots use exact string values.
    path.write_text(json.dumps(value,indent=2,default=str)+'\n');return value


@app.local_entrypoint()
def main(run_id:str='phase4-onset-20261021'):
    from src.build_prompts import ROOT
    out=ROOT/'results/phase4';out.mkdir(exist_ok=True)
    require(run_id.startswith('phase4-onset-') and run_id.replace('-','').isalnum(),'unsafe run ID')
    with (out/'modal-onset.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=out/'modal-onset-attempts.json'
        attempts=json.loads(ledger.read_text()) if ledger.exists() else []
        budget_limits(attempts,time.time())
        require(not any(a['run_id']==run_id for a in attempts),'fresh run ID required')
        package=load_inputs(ROOT)
        containers=snapshot(['container','list','--json'])
        write_json(out/'modal-onset-containers-before.json',dict(captured_unix=time.time(),containers=containers))
        require(containers==[],'workspace must have zero containers at launch')
        require(not any(str(e.path).strip('/').split('/')[0]==run_id for e in volume.listdir('/')),
                'remote run ID must be fresh')
        billing=billing_snapshot(out/'modal-onset-billing-before.json')
        call=None;zero=False;caught=None;result=None
        with modal.Dict.ephemeral() as progress:
            start=time.time();tick=time.monotonic();bounds=budget_limits(attempts,start)
            manifest=dict(run_id=run_id,source_run_id=SOURCE,phase3_reference_commit='f223175',
                          cap_aggregate_modal_usd=CAP_USD,rate_upper_usd_per_second=ALL_IN_RATE,
                          gpu_rate_usd_per_second=GPU_RATE,previous_seconds=bounds['previous_seconds'],
                          payload_sha256=digest(package),source_hashes=package['source_hashes'],
                          pricing_source='https://modal.com/pricing',pricing_checked='2026-10-21',
                          boundary_positions='-64..64; 0 is <think>',reasoning_positions='1..256')
            write_json(out/'onset-manifest.json',manifest)
            attempt=dict(run_id=run_id,status='running',gpu='L40S',started_unix=start,**bounds)
            attempts.append(attempt);write_json(ledger,attempts)
            try:
                call=track.with_options(timeout=bounds['timeout_seconds'],retries=0).spawn(
                    package,manifest,run_id,progress,start,bounds['absolute_deadline_unix'])
                attempt['call_id']=call.object_id;write_json(ledger,attempts);seen=None
                while True:
                    if time.monotonic()-tick>=bounds['exposure_seconds']-RESERVE:raise TimeoutError('Local aggregate Modal cap guard')
                    state=progress.get('state')
                    if state and state!=seen:
                        write_json(out/'onset-progress.json',state);print('LOCAL '+json.dumps(state),flush=True);seen=state
                    try:result=call.get(timeout=0)
                    except TimeoutError:time.sleep(2);continue
                    attempt.update(result);break
            except BaseException as error:
                caught=error;attempt.update(status='failed_or_cancelled' if call is not None else 'submission_unresolved',error=f'{type(error).__name__}: {error}')
            finally:
                if call is not None:
                    try:call.cancel(terminate_containers=True)
                    except Exception as error:attempt['cancel_error']=str(error)
                else:
                    # Uncertain spawn: stop this app, never any other agent's app.
                    attempt['app_stop']=subprocess.run(['uv','run','modal','app','stop',app.app_id],capture_output=True,text=True,timeout=15).stdout
                for _ in range(25):
                    try:
                        containers=snapshot(['container','list','--json'])
                        write_json(out/'modal-onset-containers-after.json',dict(captured_unix=time.time(),containers=containers))
                        if containers==[]:
                            zero=True;attempt['zero_containers_verified_unix']=time.time();break
                    except Exception as error:attempt['shutdown_error']=str(error)
                    if time.monotonic()-tick>=bounds['exposure_seconds']-20:break
                    time.sleep(2)
                elapsed=max(time.monotonic()-tick,time.time()-start)
                attempt.update(ended_unix=time.time(),observed_wall_seconds=elapsed,zero_containers_verified=zero,
                               wall_seconds=elapsed if call is not None and zero else bounds['exposure_seconds'])
                if not zero:attempt['status']='shutdown_unresolved'
                write_json(ledger,attempts)
                recovery_errors=[]
                if call is not None:
                    try:
                        for entry in volume.listdir(run_id):
                            name=Path(entry.path).name
                            if name.endswith('.tmp'):continue
                            require(name.startswith('onset-'),'unowned remote artifact')
                            with (out/name).open('wb') as handle:
                                for chunk in volume.read_file(entry.path):handle.write(chunk)
                    except Exception as error:recovery_errors.append(str(error))
        after=billing_snapshot(out/'modal-onset-billing-after.json')
        seconds=sum(a['wall_seconds'] for a in attempts)
        execution=dict(run_id=run_id,app_id=app.app_id,attempts=attempts,zero_containers_verified=zero,
                       aggregate_modal_usd_upper=seconds*ALL_IN_RATE,gpu_usd_upper=seconds*GPU_RATE,
                       cap_aggregate_modal_usd=CAP_USD,rate_upper_usd_per_second=ALL_IN_RATE,
                       accounting_basis='all submission-to-verified-zero wall seconds, including failures and provisioning; conservative GPU+CPU+memory rate',
                       billing_before=billing,billing_after=after,recovery_errors=recovery_errors)
        write_json(out/'modal-onset-execution.json',execution)
        print(json.dumps(execution),flush=True)
        if caught is not None:raise caught
        require(zero and not recovery_errors and attempt['status']=='completed','onset incomplete; see execution evidence')
        require(execution['aggregate_modal_usd_upper']<=CAP_USD,'aggregate Modal cap exceeded')
