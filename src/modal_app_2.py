"""Single-L40S, single-use HF Transformers Phase 2 study. No vLLM execution."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import modal
from src.phase2 import MODEL, RATE, CAP_SECONDS, build_requests
from src.models import MODELS

app = modal.App('grader-mode-phase2')
hf_cache = modal.Volume.from_name('grader-mode-hf-cache', create_if_missing=True)
volume = modal.Volume.from_name('grader-mode-phase2-results', create_if_missing=True)
image = (modal.Image.from_registry('nvidia/cuda:12.8.1-runtime-ubuntu22.04', add_python='3.12')
         .entrypoint([]).uv_pip_install('torch==2.9.0', 'transformers==4.57.3',
                                      'accelerate==1.11.0', 'numpy==2.3.4')
         .env({'HF_HOME':'/hf', 'HF_XET_HIGH_PERFORMANCE':'1', 'DO_NOT_TRACK':'1',
               'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'})
         .add_local_python_source('src'))


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def extract_hidden(output):
    return output[0] if isinstance(output, tuple) else output


def residual_hook(direction, mode, alpha=0.):
    from src.phase2 import transform_residual
    def hook(module, args, output):
        hidden = extract_hidden(output)
        changed = transform_residual(hidden.float(), direction, mode, alpha).to(hidden.dtype)
        return (changed, *output[1:]) if isinstance(output,tuple) else changed
    return hook


@app.function(image=image, gpu='L40S', volumes={'/hf':hf_cache,'/study':volume},
              timeout=10500, startup_timeout=120, retries=0, single_use_containers=True,
              min_containers=0, buffer_containers=0, max_containers=1,
              scaledown_window=2, cpu=4, memory=32768)
def study(requests: list, manifest: dict, run_id: str, progress: modal.Dict,
          previous_seconds: float = 0., absolute_deadline_unix: float = 0.):
    started = time.monotonic()
    import gc
    import importlib.metadata
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.modal_app import render_prompt
    from src.phase2 import (compute_directions, degeneration, middle_layers,
                            choose_plan, test_cells, transform_residual)
    from src.score import score_response
    # Leave 180s for shutdown/allocation overhead. Local controller also enforces cap.
    if absolute_deadline_unix<=time.time():raise TimeoutError('Remote absolute deadline expired')
    deadline = started + absolute_deadline_unix-time.time()
    directory = Path('/study')/run_id
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    artifacts = []
    completed_batches = []
    stats = {'run_id':run_id,'model':MODEL,'revision':MODELS[MODEL], 'gpu':'L40S',
             'sampling':{'temperature':.7,'top_p':.95,'top_k':0,'max_new_tokens':768,
                         'enable_thinking':False,'dtype':'bfloat16', 'left_padding':True},
             'seed_method':'torch.manual_seed(first Phase 1 request seed per ordered batch); global CUDA stream; EOS rows removed from DynamicCache; boundaries recorded',
             'decoder':'HF forward + TopPLogitsWarper(.95), torch.multinomial, EOS-row DynamicCache compaction',
             'manifest':manifest,'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','numpy','accelerate')},
             'previous_seconds':previous_seconds,'batches':completed_batches}
    def checkpoint(stage, **extra):
        stats.update(extra)
        stats.update(stage=stage,function_seconds=time.monotonic()-started,
                     estimated_gpu_usd=(previous_seconds+time.monotonic()-started)*RATE)
        write_json(directory/'metrics.json',stats)
        volume.commit()
        progress['state'] = {'stage':stage,'function_seconds':stats['function_seconds'],
                             'batches':len(completed_batches), **extra}
        print('PHASE2 '+json.dumps(progress['state']), flush=True)
    def remaining():
        return deadline-time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODELS[MODEL], padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODELS[MODEL],
             torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa').eval()
    hf_cache.commit()
    layers = model.model.layers
    depth, hidden_size = len(layers), model.config.hidden_size
    # Small tensor runtime test covers actual torch hook path, including tuple output.
    x = torch.randn(2,3,hidden_size,device='cuda',dtype=torch.float32)
    d = torch.randn(hidden_size,device='cuda'); d /= d.norm()
    assert torch.max(torch.abs(transform_residual(x,d,'ablate')@d)).item()<1e-4
    assert torch.max(torch.abs((transform_residual(x,d,'add',4.)-x)@d-4.)).item()<1e-4
    for mode in ('ablate','add'):
        hook = residual_hook(d,mode,4.)
        assert torch.equal(hook(None,(),(x,'sentinel'))[0], transform_residual(x,d,mode,4.))
        assert hook(None,(),(x,'sentinel'))[1] == 'sentinel'
    del x,d
    rendered = {r['id']:render_prompt(tokenizer,r) for r in requests}
    def encode(rows):
        return tokenizer([rendered[r['id']] for r in rows],padding=True,
                         add_special_tokens=False,return_tensors='pt').to('cuda')
    def select(split,fmt,n=100,templates=4):
        return sorted([r for r in requests if r['split']==split and r['format']==fmt
                       and (split=='test' or r['selection_rank']<=n) and r['template_idx']<templates],
                      key=lambda r:(r['selection_rank'],r['template_idx']))
    from src.hf_decode import compact_generate
    # Greedy parity on unequal left-padded prompts checks cache positions/compaction.
    parity_rows=select('dev','none',1)[:2]
    parity_input=encode(parity_rows)
    with torch.inference_mode():
        reference=model.generate(**parity_input,do_sample=False,max_new_tokens=8,
                    logits_to_keep=1,pad_token_id=tokenizer.pad_token_id)
        actual,parity_meta=compact_generate(model,parity_input,eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=8,deadline=deadline,sample=False)
    assert torch.equal(reference[:,parity_input.input_ids.shape[1]:],actual)
    del reference,actual,parity_input
    stats['greedy_cache_parity_passed']=True
    checkpoint('loaded', setup_seconds=time.monotonic()-started,depth=depth,hidden_size=hidden_size,
               cuda_device=torch.cuda.get_device_name(),hook_math_runtime_passed=True)
    # Capture only final prompt token at decoder OUTPUT, before final model norm.
    captured = {}
    capture_started = time.monotonic()
    for fmt in ('none','boxed'):
        values = []
        rows = select('dev',fmt)
        for offset in range(0,len(rows),32):
            if remaining()<120:
                raise TimeoutError('Budget deadline during capture')
            batch = rows[offset:offset+32]
            final = [None]*depth
            handles=[]
            for li,layer in enumerate(layers):
                def capture(module,args,output,li=li):
                    final[li] = extract_hidden(output)[:,-1,:].float().detach().cpu().numpy()
                handles.append(layer.register_forward_hook(capture))
            try:
                with torch.inference_mode():
                    model(**encode(batch),use_cache=False,logits_to_keep=1)
            finally:
                for handle in handles: handle.remove()
            values.append(np.stack(final,axis=1))
        captured[fmt] = np.concatenate(values,axis=0)
    directions, layer_stats = compute_directions(captured['none'],captured['boxed'])
    del captured
    gc.collect(); torch.cuda.empty_cache()
    randoms = np.random.default_rng(0).standard_normal((5,hidden_size))
    randoms /= np.linalg.norm(randoms,axis=1,keepdims=True)
    np.savez_compressed(directory/'directions.npz',directions=directions,random_directions=randoms.astype(np.float32))
    write_json(directory/'layer-stats.json',{'layers':layer_stats,'random_seed':0,
               'capture_seconds':time.monotonic()-capture_started,'unit_norms':np.linalg.norm(directions,axis=1).tolist()})
    artifacts += ['directions.npz','layer-stats.json']
    checkpoint('captured',capture_seconds=time.monotonic()-capture_started)
    # KV budget for max tokens plus prompt; leave 8GiB beyond weights for activations/sampling.
    bytes_per_token = 2*depth*model.config.num_key_value_heads*(hidden_size//model.config.num_attention_heads)*2
    free,_ = torch.cuda.mem_get_info()
    max_prompt = max(len(tokenizer.encode(p,add_special_tokens=False)) for p in rendered.values())
    batch_size = max(16,min(256,int((free-8*1024**3)/(bytes_per_token*(max_prompt+768)))))
    batch_size = max(16,(batch_size//16)*16)
    stats.update(initial_batch_size=batch_size,max_prompt_tokens=max_prompt,
                 kv_bytes_per_token=bytes_per_token)
    raw = (directory/'generations.jsonl').open('x')
    artifacts.append('generations.jsonl')
    cache = {}
    def generation(rows, cell, mode=None, layer=None, direction=None, alpha=0.):
        nonlocal batch_size
        pending = [r for r in rows if (r['id'],cell) not in cache]
        chunk_size=batch_size
        for offset in range(0,len(pending),chunk_size):
            # OOM retry uses smaller sub-batches; no successful generation is overwritten.
            queue=[pending[offset:offset+chunk_size]]
            while queue:
                batch=queue.pop(0)
                if remaining()<60:
                    raise TimeoutError('GPU budget deadline before batch')
                seed = batch[0]['seed']
                torch.manual_seed(seed)
                handle = None
                if mode:
                    vector = torch.tensor(direction,device='cuda',dtype=torch.float32)
                    handle = layers[layer].register_forward_hook(residual_hook(vector,mode,alpha))
                tick=time.monotonic()
                try:
                    encoded=encode(batch)
                    with torch.inference_mode():
                        def decode_progress(step,active):
                            if step%64==0:
                                progress['state']={'stage':stats.get('stage','generation'),
                                    'function_seconds':time.monotonic()-started,'batches':len(completed_batches),
                                    'decoding_cell':cell,'decode_step':step,'active_rows':active}
                        generated,decode_meta=compact_generate(model,encoded,
                          eos_token_id=tokenizer.eos_token_id,max_new_tokens=768,deadline=deadline,
                          progress=decode_progress)
                    torch.cuda.synchronize()
                except torch.cuda.OutOfMemoryError:
                    if len(batch)==1: raise
                    # Release traceback-local tensors before retrying.
                    if 'encoded' in locals(): del encoded
                    gc.collect(); torch.cuda.empty_cache()
                    half=len(batch)//2
                    queue[:0]=[batch[:half],batch[half:]]
                    stats.setdefault('oom_splits',[]).append({'cell':cell,'size':len(batch),'seconds':time.monotonic()-tick})
                    batch_size=min(batch_size,half)
                    continue
                finally:
                    if handle is not None: handle.remove()
                seconds=time.monotonic()-tick
                prompt_width=encoded.input_ids.shape[1]
                tokens=generated.cpu().tolist()
                batch_record={'cell':cell,'n':len(batch),'seed':seed,'seconds':seconds,
                              'first_id':batch[0]['id'],'last_id':batch[-1]['id'],
                              'prompt_width':prompt_width,**decode_meta}
                completed_batches.append(batch_record)
                for request,ids in zip(batch,tokens):
                    eos=tokenizer.eos_token_id
                    stop=ids.index(eos) if eos in ids else len(ids)
                    text=tokenizer.decode(ids[:stop],skip_special_tokens=True)
                    item={**request,'id':f"{request['id']}:{cell}",'intervention':cell,
                          'model':MODEL,'model_revision':MODELS[MODEL],'run_id':run_id,
                          'response':text,'output_tokens':stop,'finish_reason':'stop' if stop<len(ids) else 'length',
                          'budget_terminated':decode_meta['budget_terminated'],
                          'batch_seed':seed,'layer':layer,'alpha':alpha,
                          'rendered_prompt_sha256':hashlib.sha256(rendered[request['id']].encode()).hexdigest(),
                          **score_response(text,request['gold']),**degeneration(text,request['question'])}
                    raw.write(json.dumps(item,allow_nan=False)+'\n')
                    # Keep scored data for selection, never prompts/responses in progress logs.
                    cache[(request['id'],cell)] = item
                raw.flush()
                if cell=='baseline' and len(completed_batches)<=2:
                    pilot_spr=sum(b['seconds'] for b in completed_batches)/sum(b['n'] for b in completed_batches)*1.15
                    early_plan=choose_plan(pilot_spr,previous_seconds+time.monotonic()-started,len(middle_layers(depth)),batch_size=batch_size)
                    stats['early_projection']=early_plan
                del encoded, generated
                checkpoint(stats.get('stage','generation'),last_batch=batch_record)
                if remaining()<=0:
                    raise TimeoutError('GPU budget deadline reached')
        return [cache[(r['id'],cell)] for r in rows]
    def leak(rows): return float(np.mean([r['leak'] for r in rows]))
    try:
        # Full development baselines supply pilot throughput and paired selection references.
        base_box=generation(select('dev','boxed'),'baseline')
        base_none=generation(select('dev','none'),'baseline')
        pilot=[b for b in completed_batches if b['cell']=='baseline']
        # Conservative padded throughput: include each batch's max decode steps.
        seconds_per_request=sum(b['seconds'] for b in pilot)/sum(b['n'] for b in pilot)*1.15
        candidates=middle_layers(depth)
        plan=choose_plan(seconds_per_request,previous_seconds+time.monotonic()-started,len(candidates),batch_size=batch_size)
        # Price small screens with a measured-batch floor rather than zero overhead.
        write_json(directory/'budget-plan.json',plan)
        artifacts.append('budget-plan.json')
        checkpoint('screen',budget_plan=plan)
        n=plan['screen_problems']
        screen=[]
        for layer in candidates:
            outcome=generation(select('dev','boxed',n),f'screen_L{layer}','ablate',layer,directions[layer])
            screen.append({'layer':layer,'n_problems':n,'leak':leak(outcome)})
        top=sorted(screen,key=lambda r:(r['leak'],r['layer']))[:3]
        checkpoint('confirm',screen_results=screen)
        confirm=[]
        for candidate in top:
            layer=candidate['layer']
            # Distinct stage IDs; full confirm independently seeds the full-size batches.
            outcome=generation(select('dev','boxed'),f'confirm_L{layer}','ablate',layer,directions[layer])
            confirm.append({'layer':layer,'leak':leak(outcome),'reduction':leak(base_box)-leak(outcome)})
        selected=min(confirm,key=lambda r:(r['leak'],r['layer']))['layer']
        scale=layer_stats[selected]['boxed_projection_abs_mean']
        alpha_trials=[]
        multiplier=16
        checkpoint('alpha',confirmed_layers=confirm,selected_layer=selected)
        for mult in (2,4,8,16):
            outcome=generation(select('dev','none'),f'alpha_m{mult}','add',selected,directions[selected],mult*scale)
            trial={'multiplier':mult,'alpha':mult*scale,'leak':leak(outcome),'increase':leak(outcome)-leak(base_none)}
            alpha_trials.append(trial)
            if trial['increase']>=.1-1e-12:
                multiplier=mult
                break
        selection={'layer':selected,'layer_indexing':'zero-based decoder output','screen':screen,
                   'confirmation':confirm,'dev_baseline_boxed_leak':leak(base_box),
                   'dev_baseline_none_leak':leak(base_none),'alpha_scale':scale,
                   'alpha_multiplier':multiplier,'alpha':multiplier*scale,'alpha_trials':alpha_trials,
                   'alpha_threshold_met':alpha_trials[-1]['increase']>=.1-1e-12,
                   'test_templates':plan['test_templates']}
        write_json(directory/'selection.json',selection)
        artifacts.append('selection.json')
        # Test-template reduction is allowed on timing alone, before any test generation.
        all_batches=completed_batches
        timed=[b for b in all_batches if b['n']>=min(64,batch_size)]
        if not timed:raise ValueError('No nonempty timing batches for test projection')
        test_spr=sum(b['seconds'] for b in timed)/sum(b['n'] for b in timed)*1.15
        projected_remaining=100*selection['test_templates']*20*test_spr
        if projected_remaining>remaining()-120 and selection['test_templates']==4:
            selection['test_templates']=2
            selection['timing_template_reduction']=True
            projected_remaining=4000*test_spr
            write_json(directory/'selection.json',selection)
        if projected_remaining>remaining()-60:
            raise TimeoutError(f'Test grid timing projection exceeds remaining budget: {projected_remaining:.1f}s > {remaining():.1f}s')
        checkpoint('test',selection=selection,projected_test_seconds=projected_remaining)
        for fmt,cell in test_cells():
            mode=None; direction=None
            if cell=='ablate_d':mode='ablate';direction=directions[selected]
            elif cell.startswith('ablate_r'):mode='ablate';direction=randoms[int(cell[-1])]
            elif cell=='add_d':mode='add';direction=directions[selected]
            elif cell.startswith('add_r'):mode='add';direction=randoms[int(cell[-1])]
            generation(select('test',fmt,templates=selection['test_templates']),cell,mode,
                       selected if mode else None,direction,selection['alpha'] if mode=='add' else 0.)
        stats['status']='completed'
    except (TimeoutError,ValueError) as error:
        stats.update(status='budget_stopped',stop_reason=str(error))
    finally:
        raw.close()
        stats['output_rows']=len(cache)
        checkpoint('finished')
    artifacts+=['metrics.json']
    return {'status':stats['status'],'artifacts':[f'{run_id}/{name}' for name in artifacts],
            'function_seconds':time.monotonic()-started}


@app.local_entrypoint()
def main(run_id: str = 'phase2-20261020'):
    import fcntl
    import subprocess
    from src.build_prompts import ROOT
    out=ROOT/'results/phase2';out.mkdir(exist_ok=True)
    lock=(out/'run.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    attempts_path=out/'gpu-attempts.json'
    attempts=json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    if any(a['status']=='running' for a in attempts):
        raise ValueError('Resolve previous running GPU allocation before another launch')
    previous=sum(a['wall_seconds'] for a in attempts)
    if previous>=CAP_SECONDS-300: raise ValueError('GPU budget exhausted')
    if not run_id.replace('-','').isalnum():raise ValueError('Unsafe run id')
    rawdir=ROOT/'results/raw'/run_id
    if rawdir.exists():raise FileExistsError(rawdir)
    # Original manifest is the exact sample already used in both Phase 1 stages.
    original=json.loads((ROOT/'results/raw/phase1-stage2/manifest.json').read_text())
    requests=build_requests(original['sample'])
    manifest={k:original[k] for k in ('dataset','config','split','revision','sample_seed','sample')}
    manifest.update(phase='phase2',source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                    for p in ('DESIGN.md','prompts/system_tutor.txt','prompts/templates.json','prompts/templates_2.json')})
    write_json(out/'manifest.json',manifest)
    write_json(out/'billing-at-launch.json',json.loads(subprocess.check_output(['uv','run','modal','billing','summary','--json'])))
    rawdir.mkdir(parents=True)
    attempt={'run_id':run_id,'status':'running','started_unix':time.time(),'previous_seconds':previous}
    attempts.append(attempt);write_json(attempts_path,attempts)
    tick=time.monotonic()
    with modal.Dict.ephemeral() as progress:
        from src.phase2 import remote_timeout
        call=None
        seen=None
        try:
            call=study.with_options(timeout=remote_timeout(previous),retries=0).spawn(
                requests,manifest,run_id,progress,previous,time.time()+CAP_SECONDS-previous-180)
            attempt['call_id']=call.object_id
            write_json(attempts_path,attempts)
            while True:
                state=progress.get('state')
                if state and state!=seen:
                    write_json(out/'progress.json',state)
                    print('LOCAL '+json.dumps(state),flush=True);seen=state
                if previous+time.monotonic()-tick>CAP_SECONDS-60:
                    raise TimeoutError('Local allocation cap reached')
                try: result=call.get(timeout=0)
                except TimeoutError:
                    time.sleep(2);continue
                break
            attempt['status']=result['status']
            attempt['function_seconds']=result['function_seconds']
            for remote in result['artifacts']:
                name=Path(remote).name
                target=rawdir/name if name=='generations.jsonl' else out/name
                with target.open('wb') as handle:
                    for chunk in volume.read_file(remote):handle.write(chunk)
        except BaseException:
            if call is not None:call.cancel(terminate_containers=True)
            attempt['status']='failed_or_cancelled' if call is not None else 'not_spawned'
            raise
        finally:
            attempt.update(wall_seconds=time.monotonic()-tick if call is not None else 0.,ended_unix=time.time())
            write_json(attempts_path,attempts)
    print(json.dumps({'run_id':run_id,'status':attempt['status'],'wall_seconds':attempt['wall_seconds'],
                      'allocation_gpu_usd_upper':sum(a['wall_seconds'] for a in attempts)*RATE}),flush=True)
