"""Single-use L40S Phase 3: screen -> teacher-force -> dev dose -> test pushes."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import modal
from src.models import MODELS
from src.phase3 import MODEL, RATE, CAP_SECONDS

app=modal.App('grader-mode-phase3')
hf_cache=modal.Volume.from_name('grader-mode-hf-cache',create_if_missing=True)
volume=modal.Volume.from_name('grader-mode-phase3-results',create_if_missing=True)
image=(modal.Image.from_registry('nvidia/cuda:12.8.1-runtime-ubuntu22.04',add_python='3.12')
       .entrypoint([]).uv_pip_install('torch==2.9.0','transformers==4.57.3','accelerate==1.11.0','numpy==2.3.4')
       .env({'HF_HOME':'/hf','HF_XET_HIGH_PERFORMANCE':'1','DO_NOT_TRACK':'1',
             'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'})
       .add_local_python_source('src'))


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def push_hook(indices,values,state):
    from src.phase3 import push_activations
    def hook(module,args):
        x=args[0]
        changed=push_activations(x,indices,values,generated=state['generated'])
        return (changed,*args[1:])
    return hook


@app.function(image=image,gpu='L40S',volumes={'/hf':hf_cache,'/study':volume},
              timeout=12600,startup_timeout=120,retries=0,single_use_containers=True,
              min_containers=0,buffer_containers=0,max_containers=1,scaledown_window=2,
              cpu=4,memory=32768)
def study(requests:list,traces:list,neutral_system:str,manifest:dict,run_id:str,progress:modal.Dict,
          previous_seconds:float=0.,absolute_deadline_unix:float=0.):
    started=time.monotonic()
    import gc
    import importlib.metadata
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from src.modal_app import render_prompt
    from src.hf_decode import compact_generate
    from src.phase3 import (screen_statistics,choose_neurons,summarize_trajectory,
                           reasoning_span,score_generation,choose_test_templates,CELLS,system_end_boundary)
    directory=Path('/study')/run_id
    if directory.exists():raise FileExistsError(directory)
    directory.mkdir(parents=True)
    deadline=started+absolute_deadline_unix-time.time()
    stats={'run_id':run_id,'model':MODEL,'revision':MODELS[MODEL],'gpu':'L40S','manifest':manifest,
           'sampling':{'enable_thinking':True,'dtype':'bfloat16','temperature':.7,'top_p':.95,
                       'top_k':0,'max_new_tokens':3072,'left_padding':True},
           'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','numpy','accelerate')},
           'previous_seconds':previous_seconds,'status':'running','batches':[],
           'seed_method':'torch.manual_seed(first Phase 1b request seed per ordered batch); global CUDA stream; EOS cache compaction',
           'push_positions':'cached generated-token inputs only; no prompt position, including final prompt token, is changed',
           'cap_usd':7.,'rate_usd_per_gpu_hour':RATE*3600}
    raw=(directory/'generations.jsonl').open('x');tracking=(directory/'tracking.jsonl').open('x')
    def checkpoint(stage,**extra):
        stats.update(extra);stats.update(stage=stage,function_seconds=time.monotonic()-started,
                                        estimated_gpu_usd=(previous_seconds+time.monotonic()-started)*RATE)
        write_json(directory/'metrics.json',stats);volume.commit()
        state={'stage':stage,'function_seconds':stats['function_seconds'],'batches':len(stats['batches']),**extra}
        progress['state']=state
        print('PHASE3 '+json.dumps(state,allow_nan=False),flush=True)
    def check(reserve=60):
        if time.monotonic()+reserve>=deadline:raise TimeoutError('Phase 3 absolute GPU budget deadline')
    try:
        check()
        tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODELS[MODEL],padding_side='left')
        tokenizer.pad_token=tokenizer.eos_token
        rendered={r['id']:render_prompt(tokenizer,r) for r in requests}
        # Verify all frozen rendered prompts before model loading or any activation collection.
        verified=0
        for r in traces:
            digest=hashlib.sha256(rendered[r['id']].encode()).hexdigest()
            old=r.get('source_rendered_prompt_sha256')
            if old is not None:
                if old!=digest:raise ValueError(f'Phase 1b rendered prompt hash mismatch: {r["id"]}')
                verified+=1
        stats['tracking_prompt_hashes_verified']=verified
        model=AutoModelForCausalLM.from_pretrained(MODEL,revision=MODELS[MODEL],torch_dtype=torch.bfloat16,
                     device_map='cuda',attn_implementation='sdpa').eval()
        hf_cache.commit()
        layers=model.model.layers;depth=len(layers);width=model.config.intermediate_size
        if (depth,width)!=(40,17408):raise ValueError('Unexpected Qwen3 neuron dimensions')
        eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos)
        stats['eos_token_ids']=eos
        def encode(rows,texts=None):
            return tokenizer(texts or [rendered[r['id']] for r in rows],padding=True,add_special_tokens=False,
                             return_tensors='pt').to('cuda')
        def select(split,templates=4):return [r for r in requests if r['split']==split and r['template_idx']<templates]
        # Exercise the actual torch forward-prehook and generated/prompt distinction.
        state={'generated':False};linear=torch.nn.Linear(5,3,bias=False,device='cuda',dtype=torch.bfloat16)
        x=torch.zeros(2,2,5,device='cuda',dtype=torch.bfloat16)
        idx=torch.tensor([1,3],device='cuda');val=torch.tensor([2.,-3.],device='cuda',dtype=torch.bfloat16)
        handle=linear.register_forward_pre_hook(push_hook(idx,val,state))
        assert torch.equal(linear(x),linear.weight.new_zeros(2,2,3))
        state['generated']=True
        expected=x.clone();expected[...,idx]+=val
        assert torch.equal(linear(x),torch.nn.functional.linear(expected,linear.weight))
        handle.remove();del linear,x,expected,idx,val
        parity_input=encode(select('dev')[:2])
        with torch.inference_mode():
            ref=model.generate(**parity_input,do_sample=False,max_new_tokens=8,logits_to_keep=1,pad_token_id=tokenizer.pad_token_id)
            actual,meta=compact_generate(model,parity_input,eos_token_id=eos,max_new_tokens=8,deadline=deadline,sample=False)
        assert torch.equal(ref[:,parity_input.input_ids.shape[1]:],actual)
        del ref,actual,parity_input
        checkpoint('screening',setup_seconds=time.monotonic()-started,depth=depth,neurons_per_layer=width,
                   greedy_cache_parity_passed=True,hook_math_runtime_passed=True,cuda_device=torch.cuda.get_device_name())
        # Only running prompt-mean moments are retained; all moments remain float64 on GPU.
        moments=torch.zeros(2,2,depth,width,device='cuda',dtype=torch.float64)
        n=[0,0];dev=select('dev');system_token_lengths=[]
        capture_tick=time.monotonic()
        for arm in range(2):
            for offset in range(0,400,8):
                check();batch=dev[offset:offset+8]
                texts=[rendered[r['id']] for r in batch]
                if arm:
                    texts=[render_prompt(tokenizer,{**r,'messages':[{'role':'system','content':neutral_system},r['messages'][1]]}) for r in batch]
                encoded=encode(batch,texts)
                im_end=tokenizer.convert_tokens_to_ids('<|im_end|>')
                boundary=system_end_boundary(encoded.input_ids,encoded.attention_mask,im_end)
                real_system_lengths=boundary-(encoded.attention_mask==0).sum(-1)
                if not bool((real_system_lengths>0).all()) or len(set(real_system_lengths.cpu().tolist()))!=1:
                    raise ValueError('System boundary must be padding-invariant within a condition')
                valid=encoded.attention_mask.bool() & (torch.arange(encoded.input_ids.shape[1],device='cuda')[None,:]>=boundary[:,None])
                if not bool(valid.any(-1).all()):raise ValueError('No post-system positions')
                system_token_lengths.append({'system':'neutral' if arm else 'tutor',
                    'first_id':batch[0]['id'],'system_end_token_counts':real_system_lengths.cpu().tolist()})
                handles=[]
                for li,layer in enumerate(layers):
                    def capture(module,args,li=li):
                        activation=args[0].double()
                        means=(activation*valid[:,:,None]).sum(1)/valid.sum(1)[:,None]
                        moments[arm,0,li]+=means.sum(0)
                        moments[arm,1,li]+=means.square().sum(0)
                    handles.append(layer.mlp.down_proj.register_forward_pre_hook(capture))
                try:
                    with torch.inference_mode():model(**encoded,use_cache=False,logits_to_keep=1)
                    torch.cuda.synchronize()
                finally:
                    for h in handles:h.remove()
                n[arm]+=len(batch);del encoded
                if n[arm]%80==0:checkpoint('screening',screen_prompts_per_system=n.copy())
        arrays=moments.cpu().numpy();del moments
        numerical=screen_statistics(arrays[0,0],arrays[0,1],arrays[1,0],arrays[1,1],400)
        neurons=choose_neurons(numerical)
        write_json(directory/'neurons.json',neurons)
        np.savez_compressed(directory/'screen-statistics.npz',**{k:v for k,v in numerical.items() if k!='n_prompts'})
        write_json(directory/'system-boundaries.json',system_token_lengths)
        del arrays,numerical;gc.collect();torch.cuda.empty_cache()
        checkpoint('tracking',screen_seconds=time.monotonic()-capture_tick,screen_prompts_per_system=n.copy(),
                   tutor_neuron_ids=[r['id'] for r in neurons['tutor']])
        # Teacher forcing captures only the six aggregate T scores at each token.
        groups=[neurons['tutor'],*neurons['controls']]
        score_specs={}
        for si,group in enumerate(groups):
            for li in sorted({r['layer'] for r in group}):
                subset=[r for r in group if r['layer']==li]
                spec=(si,torch.tensor([r['neuron'] for r in subset],device='cuda'),
                      torch.tensor([r['tutor_mean'] for r in subset],device='cuda'),
                      torch.tensor([r['tutor_sd'] for r in subset],device='cuda'))
                score_specs.setdefault(li,[]).append(spec)
        trajectories=[];token_ids=[];trace_offsets=[0];trace_ids=[];track_tick=time.monotonic()
        for offset in range(0,len(traces),4):
            check();batch=traces[offset:offset+4]
            texts=[rendered[r['id']]+r['response'] for r in batch]
            encoded_cpu=tokenizer(texts,padding=True,add_special_tokens=False,return_offsets_mapping=True,return_tensors='pt')
            offsets=encoded_cpu.pop('offset_mapping').tolist()
            encoded=encoded_cpu.to('cuda')
            scores=torch.zeros(len(batch),encoded.input_ids.shape[1],6,device='cuda',dtype=torch.float32)
            handles=[]
            for li,specs in score_specs.items():
                def capture_scores(module,args,specs=specs):
                    for si,indices,means,sds in specs:
                        scores[:,:,si]+=((args[0][...,indices].float()-means)/sds).sum(-1)/20
                handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(capture_scores))
            try:
                with torch.inference_mode():model(**encoded,use_cache=False,logits_to_keep=1)
                torch.cuda.synchronize()
            finally:
                for h in handles:h.remove()
            cpu_scores=scores.cpu().numpy();cpu_ids=encoded.input_ids.cpu().tolist()
            for bi,r in enumerate(batch):
                span=reasoning_span(r['response']);prefix=len(rendered[r['id']])
                prompt_ids=tokenizer.encode(rendered[r['id']],add_special_tokens=False)
                full_ids=[t for t,m in zip(cpu_ids[bi],encoded_cpu.attention_mask[bi].cpu().tolist()) if m]
                if full_ids[:len(prompt_ids)]!=prompt_ids:raise ValueError('Prompt/response token boundary changed on replay')
                positions=[] if span is None else [j for j,(a,b) in enumerate(offsets[bi]) if b>a and a>=prefix+span[0] and b<=prefix+span[1]]
                values=cpu_scores[bi,positions,:]
                item={k:r[k] for k in ('id','problem_idx','selection_rank','template_idx','gold','leak','excluded_from_rates')}
                item.update(closed_reasoning=span is not None,n_reasoning_tokens=len(positions),
                            rendered_prompt_sha256=hashlib.sha256(rendered[r['id']].encode()).hexdigest(),
                            source_response_sha256=hashlib.sha256(r['response'].encode()).hexdigest(),
                            **summarize_trajectory(values))
                tracking.write(json.dumps(item,allow_nan=False)+'\n')
                trajectories.append(values);token_ids.extend(cpu_ids[bi][j] for j in positions)
                trace_offsets.append(trace_offsets[-1]+len(positions));trace_ids.append(r['id'])
            tracking.flush();del encoded,scores,encoded_cpu
            if (offset+4)%40==0:checkpoint('tracking',tracking_rows=offset+len(batch),tracking_seconds=time.monotonic()-track_tick)
        np.savez_compressed(directory/'trajectories.npz',scores=np.concatenate(trajectories),
                            token_ids=np.asarray(token_ids,dtype=np.int32),offsets=np.asarray(trace_offsets),ids=np.asarray(trace_ids))
        del trajectories,token_ids,cpu_scores,score_specs;gc.collect();torch.cuda.empty_cache()
        checkpoint('dev_baseline',tracking_rows=400,tracking_seconds=time.monotonic()-track_tick)
        # Select a cache-safe batch size using full 3072-token capacity and a 4 GiB work reserve.
        bytes_per_token=2*depth*model.config.num_key_value_heads*getattr(model.config,'head_dim',model.config.hidden_size//model.config.num_attention_heads)*2
        max_prompt=max(len(tokenizer.encode(t,add_special_tokens=False)) for t in rendered.values())
        free,_=torch.cuda.mem_get_info()
        batch_size=max(1,min(64,int((free-4*1024**3)/(bytes_per_token*(max_prompt+3072)))))
        if batch_size>=8:batch_size=(batch_size//8)*8
        stats.update(batch_size=batch_size,kv_bytes_per_token=bytes_per_token,max_prompt_tokens=max_prompt)
        output_rows=0
        def generation(rows,cell,c=0.):
            nonlocal batch_size,output_rows
            results=[];chunk_size=batch_size
            for offset in range(0,len(rows),chunk_size):
                queue=[rows[offset:offset+chunk_size]]
                while queue:
                    check();batch=queue.pop(0);seed=batch[0]['seed'];torch.manual_seed(seed)
                    state={'generated':False};handles=[]
                    if cell!='baseline':
                        control=int(cell.split('_')[-1]) if cell.startswith('control_') else None
                        group=neurons['controls'][control] if control is not None else neurons['tutor']
                        values=[c*abs(r['delta'])*neurons['control_signs'][control][i] if control is not None else c*r['delta'] for i,r in enumerate(neurons['tutor'])]
                        for li in sorted({r['layer'] for r in group}):
                            indices=torch.tensor([r['neuron'] for r in group if r['layer']==li],device='cuda')
                            amounts=torch.tensor([v for r,v in zip(group,values) if r['layer']==li],device='cuda',dtype=torch.float32)
                            handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(push_hook(indices,amounts,state)))
                    def forward(**kwargs):
                        state['generated']=kwargs.get('past_key_values') is not None
                        return model(**kwargs)
                    def decode_progress(step,active):
                        if step%128==0:progress['state']={'stage':stats['stage'],'cell':cell,'c':c,'decode_step':step,'active_rows':active,
                                                          'batches':len(stats['batches']),'function_seconds':time.monotonic()-started}
                    tick=time.monotonic()
                    try:
                        encoded=encode(batch)
                        with torch.inference_mode():
                            generated,decode_meta=compact_generate(forward,encoded,eos_token_id=eos,max_new_tokens=3072,
                                                                  deadline=deadline,progress=decode_progress)
                        torch.cuda.synchronize()
                    except torch.cuda.OutOfMemoryError:
                        if len(batch)==1:raise
                        if 'encoded' in locals():del encoded
                        gc.collect();torch.cuda.empty_cache()
                        half=len(batch)//2;queue[:0]=[batch[:half],batch[half:]]
                        stats.setdefault('oom_splits',[]).append({'cell':cell,'n':len(batch),'seconds':time.monotonic()-tick})
                        batch_size=min(batch_size,half)
                        continue
                    finally:
                        for h in handles:h.remove()
                    record={'cell':cell,'c':c,'split':batch[0]['split'],'n':len(batch),'seed':seed,
                            'first_id':batch[0]['id'],'last_id':batch[-1]['id'],
                            'seconds':time.monotonic()-tick,'prompt_width':encoded.input_ids.shape[1],**decode_meta}
                    stats['batches'].append(record)
                    if decode_meta['budget_terminated']:raise TimeoutError('Generation reached absolute budget deadline; incomplete batch excluded')
                    for r,ids in zip(batch,generated.cpu().tolist()):
                        end=next((i for i,t in enumerate(ids) if t in eos),len(ids));text=tokenizer.decode(ids[:end],skip_special_tokens=True)
                        scored=score_generation(text,r['gold'],'stop' if end<len(ids) else 'length')
                        item={**r,'id':f'{r["id"]}:{cell}:c{c:g}','intervention':'tutor' if cell.startswith('dose_') else cell,
                              'generation_cell':cell,'c':c,'model':MODEL,'model_revision':MODELS[MODEL],'run_id':run_id,
                              'response':text,'output_tokens':end,'finish_reason':'stop' if end<len(ids) else 'length',
                              'end_token_id':ids[end] if end<len(ids) else None,'batch_seed':seed,'budget_terminated':False,
                              'rendered_prompt_sha256':hashlib.sha256(rendered[r['id']].encode()).hexdigest(),**scored,
                              'final_reply_tokens':len(tokenizer.encode(scored['final_reply'],add_special_tokens=False))}
                        results.append(item);raw.write(json.dumps(item,allow_nan=False)+'\n');output_rows+=1
                    raw.flush();del encoded,generated
                    checkpoint(stats['stage'],last_batch=record,output_rows=output_rows)
            return results
        def leak(rows):
            valid=[r for r in rows if not r['excluded_from_rates']]
            if not valid:raise ValueError('No nonempty final replies for development dose selection')
            return sum(r['leak'] for r in valid)/len(valid)
        dev0=select('dev',1)
        baseline=generation(dev0,'baseline');base_rate=leak(baseline)
        checkpoint('dose_selection',dev_baseline_leak=base_rate)
        trials=[];chosen=8;met=False
        for c in (1,2,4,8):
            outcome=generation(dev0,f'dose_{c}',c)
            rate=leak(outcome);reduction=base_rate-rate
            trials.append({'c':c,'n':len(outcome),'n_valid':sum(not r['excluded_from_rates'] for r in outcome),'leak':rate,'reduction':reduction})
            if reduction>=.1-1e-12:chosen=c;met=True;break
        selection={'c_star':chosen,'dose_threshold_met':met,'dev_baseline_leak':base_rate,'dose_trials':trials,
                   'test_templates':None,'test_generation_started':False}
        write_json(directory/'selection.json',selection)
        timed=[b for b in stats['batches'] if b['split']=='dev']
        # Worst measured full-development cell, with 20% headroom for controls and tails.
        per_cell=[]
        for cell in sorted({b['cell'] for b in timed}):
            sub=[b for b in timed if b['cell']==cell];per_cell.append(sum(b['seconds'] for b in sub)/sum(b['n'] for b in sub))
        spr=max(per_cell)*1.2
        spent=previous_seconds+time.monotonic()-started
        test_templates=choose_test_templates(spr,spent)
        selection.update(test_templates=test_templates,timing_template_reduction=test_templates==1,
                         timing_seconds_per_request=spr,spent_gpu_seconds_at_test_gate=spent,
                         projected_test_seconds=100*test_templates*7*spr,
                         projected_gpu_usd=(spent+100*test_templates*7*spr+300)*RATE)
        write_json(directory/'selection.json',selection)
        checkpoint('test',selection=selection)
        test=select('test',test_templates)
        selection['test_generation_started']=True;write_json(directory/'selection.json',selection)
        for cell in CELLS:generation(test,cell,0. if cell=='baseline' else chosen)
        stats.update(status='completed',selection=selection,output_rows=output_rows)
    except BaseException as error:
        stats.update(status='budget_stopped' if isinstance(error,TimeoutError) else 'failed',
                     stop_reason=f'{type(error).__name__}: {error}')
        import traceback
        traceback.print_exc()
    finally:
        raw.close();tracking.close();checkpoint('finished')
    return {'status':stats['status'],'function_seconds':time.monotonic()-started,
            'artifacts':[f'{run_id}/{p.name}' for p in directory.iterdir() if p.is_file()]}


@app.local_entrypoint()
def main(run_id:str='phase3-20261020'):
    import fcntl
    import subprocess
    from src.build_prompts import ROOT
    from src.phase3 import build_requests,validate_tracking_sources,remote_timeout,close_attempt
    from src.score import read_rows
    out=ROOT/'results/phase3';out.mkdir(exist_ok=True)
    lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger=out/'gpu-attempts.json';attempts=json.loads(ledger.read_text()) if ledger.exists() else []
    if any(a['status'] in ('running','submission_unresolved') for a in attempts):raise ValueError('Reconcile running Phase 3 allocation before launch')
    previous=sum(a['wall_seconds'] for a in attempts)
    timeout=remote_timeout(previous)
    if not run_id.replace('-','').isalnum():raise ValueError('Unsafe run id')
    rawdir=ROOT/'results/raw'/run_id
    if rawdir.exists():raise FileExistsError(rawdir)
    original=json.loads((ROOT/'results/raw/phase1-stage2/manifest.json').read_text())
    requests=build_requests(original['sample'])
    raw_path=ROOT/'results/raw/phase1b-full/Qwen3-14B.jsonl';scored_path=raw_path.with_name('scored.jsonl')
    traces=validate_tracking_sources(requests,list(read_rows(raw_path)),list(read_rows(scored_path)))
    paths=('DESIGN.md','prompts/system_tutor.txt','prompts/system_neutral.txt','prompts/templates_1b.json',
           'src/models.py','src/phase3.py','src/modal_app_3.py','src/hf_decode.py','src/score.py',
           'results/raw/phase1b-full/Qwen3-14B.jsonl','results/raw/phase1b-full/scored.jsonl')
    manifest={k:original[k] for k in ('dataset','config','split','revision','sample_seed','sample')}
    manifest.update(phase='phase3',preregistration_commit='0e6a330',cap_usd=7.,
                    source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
    write_json(out/'manifest.json',manifest)
    write_json(out/'billing-at-launch.json',json.loads(subprocess.check_output(['uv','run','modal','billing','summary','--json'])))
    rawdir.mkdir(parents=True)
    attempt={'run_id':run_id,'status':'running','started_unix':time.time(),'previous_seconds':previous,
             'gpu':'L40S','rate_usd_per_hour':RATE*3600,'timeout_seconds':timeout}
    attempts.append(attempt);write_json(ledger,attempts);tick=time.monotonic();call=None
    with modal.Dict.ephemeral() as progress:
        try:
            call=study.with_options(timeout=timeout,retries=0).spawn(requests,traces,
                (ROOT/'prompts/system_neutral.txt').read_text(),manifest,run_id,progress,previous,time.time()+CAP_SECONDS-previous-180)
            attempt['call_id']=call.object_id;write_json(ledger,attempts);seen=None
            while True:
                state=progress.get('state')
                if state and state!=seen:
                    write_json(out/'progress.json',state);print('LOCAL '+json.dumps(state),flush=True);seen=state
                if previous+time.monotonic()-tick>CAP_SECONDS-60:raise TimeoutError('Local Phase 3 allocation cap')
                try:result=call.get(timeout=0)
                except TimeoutError:time.sleep(2);continue
                attempt.update(status=result['status'],function_seconds=result['function_seconds']);break
        except BaseException:
            if call is not None:call.cancel(terminate_containers=True)
            attempt['status']='failed_or_cancelled' if call is not None else 'submission_unresolved'
            raise
        finally:
            attempt.update(close_attempt(call_known=call is not None,elapsed=time.monotonic()-tick,
                                         previous_seconds=previous,status=attempt['status']),ended_unix=time.time())
            write_json(ledger,attempts)
            # Recover checkpointed partial results even after remote failure/cancellation.
            if call is not None:
                for entry in volume.listdir(run_id):
                    name=Path(entry.path).name
                    target=rawdir/name if name in ('generations.jsonl','tracking.jsonl','trajectories.npz') else out/name
                    if name.endswith('.tmp'):continue
                    with target.open('wb') as handle:
                        for chunk in volume.read_file(entry.path):handle.write(chunk)
    print(json.dumps({'run_id':run_id,'status':attempt['status'],'wall_seconds':attempt['wall_seconds'],
                      'allocation_gpu_usd_upper':sum(a['wall_seconds'] for a in attempts)*RATE}),flush=True)
