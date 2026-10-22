"""Batched closed-loop reasoning steering, exact-ID chunk continuation."""
import json,os,pathlib,re,time,traceback,hashlib
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING','0')
os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER','0')
import numpy as np
import torch
from transformers import AutoTokenizer
from vllm import LLM,SamplingParams
from gate import load,score
from pulse_core import REMINDER,measures
from closed_loop_core import calibrate,chunk_stats,retain,monitor_value
ROOT=pathlib.Path(__file__).resolve().parent; OUT=ROOT/'out';OUT.mkdir(exist_ok=True)
MODEL='Qwen/Qwen3.8-27B'; REV='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
detector=load(ROOT/'detector.json'); manifest=json.loads((ROOT/'manifest.json').read_text())
gates=[json.loads(l) for l in (OUT/'gates.jsonl').read_text().splitlines()]

def event(kind,**kw):
    row=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=kind,**kw)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    (ROOT/'heartbeat').touch();print(json.dumps(row),flush=True)


def install(worker):
    runner=worker.model_runner;model=runner.model
    model._cl_capture={};model._cl_enabled=False;model._cl_push=False;model._cl_spec={};model._cl_handles=[]
    by={}
    for col,(l,n,v) in enumerate(detector['features']):by.setdefault(l,[]).append((col,n))
    mods={int(m.group(1)):mod for name,mod in model.named_modules() if (m:=re.search(r'layers\.(\d+)\.mlp\.down_proj$',name))}
    needed=set(by)|set(int(x[0]) for name in ['q38-T3E3.json','q38-rand_T3E3.json'] for x in json.loads((ROOT/name).read_text()))
    for layer in sorted(needed):
        cols=[c for c,n in by.get(layer,[])];neurons=[n for c,n in by.get(layer,[])]
        def pre(mod,args,layer=layer,cols=cols,neurons=neurons):
            if not model._cl_enabled:return
            x=args[0]
            if cols:
                batch=runner.input_batch; nreq=batch.num_reqs
                starts=runner.query_start_loc.np[:nreq+1].copy()
                request_ids=list(batch.req_ids[:nreq]); rowids=[]; keys=[]
                for i,rid in enumerate(request_ids):
                    a,b=int(starts[i]),int(starts[i+1]);base=int(batch.num_computed_tokens_cpu[i])
                    for j in range(max(a,b-8),b):
                        rowids.append(j);keys.append((rid,base+j-a))
                assert not rowids or max(rowids)<x.shape[0]
                if rowids:
                    values=x[rowids][:,neurons].detach().float().cpu().numpy()
                    for (rid,pos),val in zip(keys,values):
                        dest=model._cl_capture.setdefault(rid,{})
                        dest.setdefault(pos,np.full(60,np.nan))[cols]=val
                        # Keep the trailing window only. Prefill captures <=8, decode grows to <=40.
                        while len(dest)>40:del dest[min(dest)]
            # Same hook records local unshifted x, then shifts. Earlier layers may already have shifted.
            if model._cl_push and layer in model._cl_spec:
                ns,vs=model._cl_spec[layer]
                changed=x.clone();changed[:,ns]+=vs
                return (changed,)+tuple(args[1:])
        model._cl_handles.append(mods[layer].register_forward_pre_hook(pre))
    return {'hooks':len(model._cl_handles),'feature_layers':sorted(by)}


def configure(worker,push,spec):
    m=worker.model_runner.model;m._cl_capture={};m._cl_enabled=True;m._cl_push=push;m._cl_spec={}
    dev=next(m.parameters()).device;by={}
    for l,n,v in spec:by.setdefault(l,[]).append((n,v))
    for l,vs in by.items():m._cl_spec[l]=(torch.tensor([n for n,v in vs],device=dev),torch.tensor([v for n,v in vs],device=dev,dtype=torch.bfloat16))


def retrieve(worker):
    m=worker.model_runner.model;m._cl_enabled=False;m._cl_push=False
    out={rid:[(p,v.tolist()) for p,v in sorted(values.items())] for rid,values in m._cl_capture.items()}
    m._cl_capture={};return out


def call(jobs,params,push=False,spec=()):
    llm.collective_rpc(configure,args=(push,spec))
    try:outs=llm.generate([{'prompt_token_ids':p} for p in jobs],params,use_tqdm=False)
    finally:captures=llm.collective_rpc(retrieve)[0]
    assert len(outs)==len(jobs)
    for p,o in zip(jobs,outs):assert list(o.prompt_token_ids)==p
    return outs,captures


def sample(n,reasoning=True):
    return SamplingParams(temperature=.7,top_p=.95,top_k=-1,max_tokens=n,seed=0,
        stop_token_ids=[closing] if reasoning else [],include_stop_str_in_output=True)


def do_calibration():
    start=time.time();rows=[];sources={}
    for cond,name in [('rl_tutor','baseline.jsonl'),('student_tutor','student_baseline.jsonl')]:
        raw=(OUT/name).read_bytes();sources[name]=hashlib.sha256(raw).hexdigest()
        traces=[json.loads(l) for l in raw.splitlines()]
        assert [r['idx'] for r in traces]==[p['idx'] for p in manifest['generation']]
        plans=[]
        for r in traces:
            ids=r['token_ids'];end=ids.index(closing) if closing in ids else len(ids)
            plans.append((r,ids[:end]))
        for chunk in range(max((len(ids)+31)//32 for r,ids in plans)):
            active=[(r,ids,min((chunk+1)*32,len(ids))) for r,ids in plans if chunk*32<len(ids)]
            prompts=[r.get('generation_prompt_token_ids',r['prompt_token_ids'])+ids[:end] for r,ids,end in active]
            outs,cap=call(prompts,SamplingParams(temperature=0,max_tokens=1,seed=0))
            for (r,ids,end),p,o in zip(active,prompts,outs):
                take=min(8,end-chunk*32);positions=list(range(len(p)-take,len(p)))
                values=monitor_value(cap[o.request_id],positions);s=float(np.mean(score(values,detector)))
                text=tok.decode(ids[chunk*32:end],skip_special_tokens=True)
                row=dict(cond=cond,idx=r['idx'],chunk=chunk,reasoning_end=end,score=s,positions=positions,
                    features=values.tolist(),chunk_text=text,grader_mention=bool(re.search('grader|reward',text,re.I)),
                    grader_rules_mention=bool(re.search('grader|rules',text,re.I)))
                rows.append(row)
                with (OUT/'calibration-chunks.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            event('calibration_chunk',cond=cond,chunk=chunk,n=len(active),scores=[rows[-len(active)+i]['score'] for i in range(len(active))])
    result=calibrate(rows);result.update(sources=sources,seconds=time.time()-start,teacher_forced=True,
        checkpoint=REV,detector_hash=hashlib.sha256((ROOT/'detector.json').read_bytes()).hexdigest())
    (OUT/'monitor.json').write_text(json.dumps(result,indent=2)+'\n')
    event('threshold_frozen',**{k:v for k,v in result.items() if k!='roc'})
    return result


def closed_loop(command):
    arm=command['arm'];cond=command.get('cond','rl_tutor');unsteered=command.get('unsteered',False)
    threshold=json.loads((OUT/'monitor.json').read_text())['threshold']
    spec=json.loads((ROOT/command.get('spec','q38-T3E3.json')).read_text())
    injected=REMINDER+'\n\n' if command.get('prefill') else ''
    jobs=[g for g in gates if g['cond']==cond];assert len(jobs)==40
    states=[];start=time.time();sampled=0;lookahead=0
    for j in jobs:
        prompt=tok.encode(j['prompt']+injected,add_special_tokens=False) if injected else j['prompt_token_ids']
        states.append(dict(job=j,prompt=prompt,ids=[],chunks=[],push=bool(j['gate']) and not unsteered,
                           decision_score=j['score'],done=False,final=False,finish_reason=None))
    event('closed_loop_start',arm=arm,cond=cond,unsteered=unsteered)
    iteration=0
    while any(not s['done'] for s in states):
        # Boundary-stopped final continuations are off and never monitored.
        for push in (True,False):
            active=[s for s in states if not s['done'] and not s['final'] and s['push']==push]
            if not active:continue
            prefixes=[s['prompt']+s['ids'] for s in active]
            outs,cap=call(prefixes,[sample(min(32,4096-len(s['ids']))+1) for s in active],push,spec)
            for s,p,o in zip(active,prefixes,outs):
                c=o.outputs[0];remaining=4096-len(s['ids']);kept,n,discard,eos,close=retain(c.token_ids,closing,eos_ids,remaining)
                sampled+=len(c.token_ids);lookahead+=discard
                positions=list(range(len(p)+n-min(8,n),len(p)+n)) if n else []
                nxt=None;vals=[]
                if not eos and not close and len(s['ids'])+len(kept)<4096:
                    vals=monitor_value(cap[o.request_id],positions).tolist()
                    nxt=float(np.mean(score(vals,detector)))
                ch=dict(chunk=len(s['chunks']),push=push,decision_score=s['decision_score'],monitor_score=nxt,
                    positions=positions if vals else [],features=vals,reasoning_tokens=n,retained_tokens=len(kept),
                    sampled_tokens=len(c.token_ids),discarded_lookahead=discard,prefix_tokens=len(p),
                    token_ids=kept,stop_reason=c.stop_reason,call_finish_reason=c.finish_reason)
                s['chunks'].append(ch);s['ids']+=kept;s['done']=eos or len(s['ids'])>=4096;s['final']=close
                s['finish_reason']='stop' if eos else 'length'
                # Decide only at next outer iteration, so an on->off sequence cannot run twice this iteration.
                s['next_push']=False if unsteered or nxt is None else nxt>=threshold
                s['next_score']=nxt
                with (OUT/(arm+'.chunks.jsonl')).open('a') as f:f.write(json.dumps({'idx':s['job']['idx'],**ch})+'\n')
        final=[s for s in states if s['final'] and not s['done']]
        if final:
            outs,_=call([s['prompt']+s['ids'] for s in final],[sample(4096-len(s['ids']),False) for s in final])
            for s,o in zip(final,outs):
                c=o.outputs[0];s['ids']+=list(c.token_ids);sampled+=len(c.token_ids);s['finish_reason']=c.finish_reason;s['done']=True
        for s in states:
            if 'next_push' in s:s['push']=s.pop('next_push');s['decision_score']=s.pop('next_score')
        iteration+=1
        snapshot=[dict(idx=s['job']['idx'],ids=s['ids'],done=s['done'],stats=chunk_stats(s['chunks'])) for s in states]
        (OUT/(arm+'.checkpoint.json')).write_text(json.dumps(snapshot)+'\n')
        event('closed_loop_chunk',arm=arm,iteration=iteration,active=sum(not s['done'] for s in states),
              retained_tokens=sum(len(s['ids']) for s in states),pushed_tokens=sum(chunk_stats(s['chunks'])['pushed_tokens'] for s in states),seconds=time.time()-start)
    rows=[]
    for s in states:
        j=s['job'];text=tok.decode(s['ids'],skip_special_tokens=True);assert len(s['ids'])<=4096
        rows.append({k:j[k] for k in ('idx','gold','gate','score','cond','prompt_token_ids')}|dict(arm=arm,
            generation_prompt_token_ids=s['prompt'],injected=injected,token_ids=s['ids'],text=text,
            finish_reason=s['finish_reason'],chunks=s['chunks'],**chunk_stats(s['chunks']),**measures(text,s['finish_reason'],j['gold'],injected)))
    with (OUT/(arm+'.jsonl')).open('w') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    result=dict(n=40,arm=arm,seconds=time.time()-start,sampled_tokens=sampled,discarded_lookahead=lookahead,
        **{k:sum(r[k] for r in rows) for k in ['finished','leak','boxed','rule_restatement','push_on_chunks','pushed_tokens','switches','retrigger','hooked_prefix_tokens']})
    result['tokens']=sum(len(r['token_ids']) for r in rows)
    return result

start=time.time();event('model_load_start')
tok=AutoTokenizer.from_pretrained(MODEL,revision=REV)
closing_ids=tok.encode('</think>',add_special_tokens=False);assert len(closing_ids)==1
closing=closing_ids[0];config=json.loads((ROOT/'generation_config.json').read_text());ee=config.get('eos_token_id',[])
eos_ids={tok.eos_token_id,* (ee if isinstance(ee,list) else [ee])};eos_ids.discard(None);assert closing not in eos_ids
llm=LLM(MODEL,revision=REV,dtype='bfloat16',enforce_eager=True,max_model_len=8192,gpu_memory_utilization=.92,
        seed=0,max_num_seqs=40,max_num_batched_tokens=16384,enable_prefix_caching=False,enable_chunked_prefill=False)
event('model_loaded',seconds=time.time()-start,hooks=llm.collective_rpc(install),closing=closing,eos_ids=sorted(eos_ids))
# Validate scheduler-to-position mapping against the frozen original prompt capture, batched and single.
checks=[g for g in gates if g['cond']=='rl_tutor'][:2]
outs,cap=call([g['prompt_token_ids'] for g in checks],SamplingParams(temperature=0,max_tokens=1,seed=0))
for g,o in zip(checks,outs):
    v=monitor_value(cap[o.request_id],[len(g['prompt_token_ids'])-1])[0]
    measured=float(score(v,detector));assert abs(measured-g['score'])<.03,(measured,g['score'])
    event('monitor_mapping_check',idx=g['idx'],original_score=g['score'],measured_score=measured,absolute_error=abs(measured-g['score']))
(OUT/'ready').touch();seen=set()
while True:
    p=ROOT/'command.json'
    if p.exists():
        command=json.loads(p.read_text());arm=command['arm']
        if arm not in seen:
            seen.add(arm)
            try:
                result=do_calibration() if command['kind']=='calibration' else closed_loop(command)
                (OUT/(arm+'.done.json')).write_text(json.dumps(result,indent=2)+'\n')
                event('job_complete',**{k:v for k,v in result.items() if k!='roc'})
            except Exception as exc:
                (OUT/(arm+'.error.json')).write_text(json.dumps(dict(arm=arm,type=type(exc).__name__,message=str(exc)))+'\n')
                event('job_error',arm=arm,type=type(exc).__name__,message=str(exc));traceback.print_exc();raise
    time.sleep(1)
