"""Round 2: frozen gating, exact token-ID pulse/recompute, command-driven jobs."""
import json, os, pathlib, re, time, traceback
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING','0')
os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER','0')
import numpy as np
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from gate import load, score
from pulse_core import REMINDER, measures, continuation

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'out'; OUT.mkdir(exist_ok=True)
MODEL='Qwen/Qwen3.8-27B'; REV='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
manifest=json.loads((ROOT/'manifest.json').read_text())
detector=load(ROOT/'detector.json'); features=detector['features']
tutor=(ROOT/'system_tutor.txt').read_text()
ns={}; code=(ROOT/'explore.py').read_text(); exec(code[code.index('RL_TEMPLATE ='):code.index('CONDITIONS = {')],ns)

def event(kind,**kw):
    row=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=kind,**kw)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    (ROOT/'heartbeat').touch(); print(json.dumps(row),flush=True)


def render(cond,q):
    kind,system=cond.split('_'); templates={'rl':ns['RL_TEMPLATE'],'student':ns['STUDENT_TEMPLATE'],'plain':'{q}'}
    messages=[{'role':'system','content':tutor}] if system=='tutor' else []
    messages.append({'role':'user','content':templates[kind].format(q=q)})
    p=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True,reasoning_effort='medium')
    assert p.endswith('<think>\n')
    return p

def install_capture(model):
    model._pulse_handles=[];model._pulse_capture={};model._pulse_expected=None
    by={}
    for col,(layer,neuron,shift) in enumerate(features):by.setdefault(layer,[]).append((col,neuron))
    supplement={16,24,32}
    found=set()
    for name,mod in model.named_modules():
        match=re.search(r'layers\.(\d+)\.mlp\.down_proj$',name)
        if not match:continue
        layer=int(match.group(1))
        if layer not in by and layer not in supplement:continue
        found.add(layer)
        cols=[c for c,n in by.get(layer,[])];neurons=[n for c,n in by.get(layer,[])]
        def pre(mod,args,layer=layer,cols=cols,neurons=neurons):
            expected=model._pulse_expected
            if expected is None:return
            x=args[0]
            assert x.ndim==2,(layer,x.shape)
            assert expected<=x.shape[0],(expected,x.shape)
            assert layer not in model._pulse_capture,('multiple forwards',layer)
            # One sequence, no chunked prefill; ignore any trailing scheduler padding.
            last=x[expected-1].detach()
            selected=last[neurons].float().cpu().numpy() if neurons else np.empty(0,dtype=np.float32)
            full=last.to(torch.float16).cpu().numpy() if layer in supplement else None
            model._pulse_capture[layer]=(cols,selected,full,tuple(x.shape))
        model._pulse_handles.append(mod.register_forward_pre_hook(pre))
    assert set(by)|supplement==found,(set(by)|supplement,found)
    return {'feature_layers':sorted(by),'hook_layers':sorted(found),'features':60}

def prepare(model,n):
    model._pulse_expected=n;model._pulse_capture={}

def retrieve(model):
    result=model._pulse_capture
    model._pulse_expected=None
    return result

def remove(model):
    for h in model._pulse_handles:h.remove()
    model._pulse_handles=[]
    return len(model._pulse_handles)


def install_shift(model,spec):
    assert not getattr(model,'_shift_hooks',[])
    assert not getattr(model,'_pulse_handles',[])
    model._shift_hooks=[]; by={}; mods={}
    for name,mod in model.named_modules():
        match=re.search(r'layers\.(\d+)\.mlp\.down_proj$',name)
        if match:mods[int(match.group(1))]=mod
    for l,n,v in spec:by.setdefault(l,[]).append((n,v))
    dev=next(model.parameters()).device
    for l,values in by.items():
        idx=torch.tensor([n for n,v in values],device=dev)
        val=torch.tensor([v for n,v in values],device=dev,dtype=torch.bfloat16)
        def pre(mod,args,idx=idx,val=val):
            x=args[0].clone(); x[...,idx]+=val
            return (x,)+tuple(args[1:])
        model._shift_hooks.append(mods[l].register_forward_pre_hook(pre))
    assert len(model._shift_hooks)==len(by)
    return len(model._shift_hooks)


def remove_shift(model):
    n=len(model._shift_hooks)
    for h in model._shift_hooks:h.remove()
    model._shift_hooks=[]
    return {'removed':n,'remaining':len(model._shift_hooks)}


def capture_gates():
    jobs=[]
    for cond in ('rl_tutor','plain_nosys','student_tutor'):
        for p in manifest['generation']:
            prompt=render(cond,p['q']); ids=tok.encode(prompt,add_special_tokens=False)
            jobs.append(dict(cond=cond,idx=p['idx'],gold=p['gold'],prompt=prompt,prompt_token_ids=ids))
    event('gate_capture_start',n=len(jobs),hooks=llm.apply_model(install_capture))
    started=time.time()
    with (OUT/'gates.jsonl').open('w') as f:
        for i,j in enumerate(jobs):
            ids=j['prompt_token_ids']; llm.apply_model(lambda m:prepare(m,len(ids)))
            o=llm.generate([{'prompt_token_ids':ids}],SamplingParams(temperature=0,max_tokens=1,seed=0),use_tqdm=False)[0]
            assert list(o.prompt_token_ids)==ids and len(o.outputs[0].token_ids)==1
            data=llm.apply_model(retrieve)[0]; vector=np.full(60,np.nan,dtype=np.float64)
            for layer,(cols,values,full,shape) in data.items():vector[cols]=values
            assert np.isfinite(vector).all()
            s=float(score(vector,detector)); j.update(features=vector.tolist(),score=s,gate=s>=detector['threshold'])
            f.write(json.dumps(j)+'\n'); f.flush()
            if (i+1)%12==0:event('gate_capture_progress',done=i+1,total=len(jobs))
    event('gate_capture_complete',n=len(jobs),seconds=time.time()-started,removed_hooks=llm.apply_model(remove),
          gates={c:sum(j['gate'] for j in jobs if j['cond']==c) for c in ('rl_tutor','plain_nosys','student_tutor')})
    return jobs


def base_fields(j,arm):
    return {k:j[k] for k in ('cond','idx','gold','gate','score','prompt_token_ids')}|{'arm':arm}


def baseline_rows(jobs,arm):
    if not jobs:return []
    event('baseline_start',arm=arm,n=len(jobs)); started=time.time()
    outs=llm.generate([{'prompt_token_ids':j['prompt_token_ids']} for j in jobs],
                     SamplingParams(temperature=.7,top_p=.95,top_k=-1,max_tokens=4096,seed=0),use_tqdm=True)
    rows=[]
    for j,o in zip(jobs,outs):
        c=o.outputs[0]; assert list(o.prompt_token_ids)==j['prompt_token_ids']
        rows.append(base_fields(j,arm)|dict(text=c.text,token_ids=list(c.token_ids),finish_reason=c.finish_reason,
                    stop_reason=c.stop_reason,source='round2_fresh_baseline',**measures(c.text,c.finish_reason,j['gold'])))
    event('baseline_complete',arm=arm,seconds=time.time()-started,n=len(rows))
    return rows


def pulse_rows(jobs,arm,k,spec_name,prefill=False):
    if not jobs:return []
    spec=json.loads((ROOT/spec_name).read_text()); injected=REMINDER+'\n\n' if prefill else ''
    prompts=[tok.encode(j['prompt']+injected,add_special_tokens=False) for j in jobs]
    event('stage_a_start',arm=arm,k=k,n=len(jobs),hooks=llm.apply_model(lambda m:install_shift(m,spec)))
    started=time.time()
    try:
        aa=llm.generate([{'prompt_token_ids':p} for p in prompts],
            SamplingParams(temperature=.7,top_p=.95,top_k=-1,max_tokens=k,seed=0,
                           stop_token_ids=[closing_id],include_stop_str_in_output=True),use_tqdm=True)
    finally:
        removed=llm.apply_model(remove_shift)
        assert all(r['remaining']==0 for r in removed)
        event('stage_a_hooks_removed',arm=arm,hooks=removed)
    assert len(aa)==len(jobs)
    plans=[]; bjobs=[]; bparams=[]
    for j,p,o in zip(jobs,prompts,aa):
        assert list(o.prompt_token_ids)==p
        c=o.outputs[0]; a=list(c.token_ids)
        prefix,remaining,actual,eos=continuation(p,a,closing_id,eos_ids,k)
        if c.stop_reason==closing_id:assert a[-1]==closing_id
        if a[-1]==closing_id:assert '</think>' in tok.decode(a,skip_special_tokens=True)
        plans.append((j,p,c,prefix,actual,eos))
        if not eos:
            bjobs.append({'prompt_token_ids':prefix})
            bparams.append(SamplingParams(temperature=.7,top_p=.95,top_k=-1,max_tokens=remaining,seed=0))
    event('stage_b_start',arm=arm,n=len(bjobs),stage_a_seconds=time.time()-started)
    bb=iter(llm.generate(bjobs,bparams,use_tqdm=True) if bjobs else [])
    rows=[]
    for j,p,a,prefix,actual,eos in plans:
        b=None if eos else next(bb)
        if b is not None:assert list(b.prompt_token_ids)==prefix
        bc=b.outputs[0] if b else None
        bids=list(bc.token_ids) if bc else []; ids=list(a.token_ids)+bids
        assert len(ids)<=4096
        text=tok.decode(ids,skip_special_tokens=True)
        finish=bc.finish_reason if bc else a.finish_reason
        rows.append(base_fields(j,arm)|dict(text=text,token_ids=ids,generation_prompt_token_ids=p,
            injected=injected,stage_a_token_ids=list(a.token_ids),stage_b_token_ids=bids,
            stage_a_finish_reason=a.finish_reason,stage_a_stop_reason=a.stop_reason,
            stage_b_finish_reason=bc.finish_reason if bc else None,stage_b_stop_reason=bc.stop_reason if bc else None,
            actual_pulse_reasoning_tokens=actual,k=k,spec=spec_name,finish_reason=finish,
            source='round2_fresh_pulse',**measures(text,finish,j['gold'],injected)))
    event('pulse_complete',arm=arm,n=len(rows),seconds=time.time()-started,
          finished=sum(r['finished'] for r in rows),leak=sum(r['leak'] for r in rows),
          tokens=sum(len(r['token_ids']) for r in rows))
    return rows


def execute(command):
    arm=command['arm']; cond=command.get('cond','rl_tutor')
    jobs=[j for j in gates if j['cond']==cond]
    if command['kind']=='baseline':rows=baseline_rows(jobs,arm)
    elif command['kind']=='pulse':
        yes=[j for j in jobs if j['gate']]; rows=pulse_rows(yes,arm,command['k'],command['spec'],command.get('prefill',False))
        # Gate-negative paths use exact baseline output, with no staging or reminder.
        if len(yes)<len(jobs):
            baseline_path=OUT/command.get('baseline_file','baseline.jsonl')
            base={r['idx']:r for r in map(json.loads,baseline_path.read_text().splitlines())}
            rows += [base[j['idx']]|base_fields(j,arm)|{'source':'exact_gate_negative_baseline_reuse'} for j in jobs if not j['gate']]
        by={r['idx']:r for r in rows}; assert len(by)==len(jobs)
        rows=[by[j['idx']] for j in jobs]
    else:raise ValueError(command)
    assert len(rows)==40 and len({r['idx'] for r in rows})==40
    with (OUT/(arm+'.jsonl')).open('w') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    result=dict(arm=arm,n=len(rows),finished=sum(r['finished'] for r in rows),leak=sum(r['leak'] for r in rows),
                boxed=sum(r['boxed'] for r in rows),rule_restatement=sum(r['rule_restatement'] for r in rows),
                tokens=sum(len(r['token_ids']) for r in rows))
    (OUT/(arm+'.done.json')).write_text(json.dumps(result,indent=2)+'\n')
    event('job_complete',**result)

start=time.time(); event('model_load_start')
tok=AutoTokenizer.from_pretrained(MODEL,revision=REV)
assert tok.encode('1 + 1',add_special_tokens=False)
closing=tok.encode('</think>',add_special_tokens=False); assert len(closing)==1,closing
closing_id=closing[0]; eos_ids={tok.eos_token_id}
config=json.loads((ROOT/'generation_config.json').read_text()) if (ROOT/'generation_config.json').exists() else {}
e=config.get('eos_token_id',[]); eos_ids.update(e if isinstance(e,list) else [e]); eos_ids.discard(None)
assert closing_id not in eos_ids
llm=LLM(MODEL,revision=REV,dtype='bfloat16',enforce_eager=True,max_model_len=8192,gpu_memory_utilization=.92,
        seed=0,max_num_seqs=40,max_num_batched_tokens=16384,enable_prefix_caching=False,enable_chunked_prefill=False)
event('model_loaded',seconds=time.time()-start,torch=torch.__version__,closing_think_token_id=closing_id,eos_ids=sorted(eos_ids))
gates=capture_gates(); (OUT/'ready').touch()
seen=set()
while True:
    p=ROOT/'command.json'
    if p.exists():
        command=json.loads(p.read_text()); arm=command['arm']
        if arm not in seen:
            seen.add(arm)
            try:execute(command)
            except Exception as exc:
                (OUT/(arm+'.error.json')).write_text(json.dumps({'arm':arm,'type':type(exc).__name__,'message':str(exc)}))
                event('job_error',arm=arm,type=type(exc).__name__,message=str(exc)); traceback.print_exc(); raise
    time.sleep(2)
