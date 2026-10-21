"""Frozen Phase 3 numerical, prompt and budget contracts; no GPU dependencies."""
from __future__ import annotations
import json
import math
import re
import numpy as np
from src.build_prompts import ROOT, expand_requests_1b
from src.score import score_response, split_thinking

MODEL = 'Qwen/Qwen3-14B'
RATE = .000542  # Modal published L40S USD/second; rounded-up relative to $1.95/hour.
CAP_USD = 7.
CAP_SECONDS = CAP_USD / RATE
CELLS = ('baseline','tutor',*(f'control_{i}' for i in range(5)))
# Literal expansion of the Phase 1b FINDINGS.md post-hoc expression.
TUTOR_REGEX = r'tutor|system (?:prompt|message|instruction)|not (?:give|provide|reveal|state) the answer|guide the student|help the student learn'
TUTOR_PATTERN = re.compile(TUTOR_REGEX,re.IGNORECASE)


def build_requests(problems):
    if len(problems)!=200 or len({p['problem_idx'] for p in problems})!=200:
        raise ValueError('Require exactly Phase 1 200 problems in selection order')
    ranks={p['problem_idx']:i+1 for i,p in enumerate(problems)}
    rows=expand_requests_1b(problems,json.loads((ROOT/'prompts/templates_1b.json').read_text()),
                            (ROOT/'prompts/system_tutor.txt').read_text())
    return [{**r,'phase':'phase3','phase1b_id':r['id'],'selection_rank':ranks[r['problem_idx']],
             'split':'dev' if ranks[r['problem_idx']]<=100 else 'test'}
            for r in rows if r['thinking'] and r['cue']=='none']


def screen_statistics(tutor_sum,tutor_sq,neutral_sum,neutral_sq,n):
    arrays=[np.asarray(x,dtype=np.float64) for x in (tutor_sum,tutor_sq,neutral_sum,neutral_sq)]
    if n<2 or any(a.shape!=arrays[0].shape or not np.isfinite(a).all() for a in arrays) or arrays[0].ndim!=2:
        raise ValueError('finite layer/neuron moments and at least two prompts required')
    ts,tq,ns,nq=arrays
    tm,nm=ts/n,ns/n
    tv,nv=np.maximum(0,(tq-ts*tm)/(n-1)),np.maximum(0,(nq-ns*nm)/(n-1))
    pooled=np.sqrt((tv+nv)/2)
    d=np.divide(tm-nm,pooled,out=np.full_like(tm,-np.inf),where=pooled>0)
    return {'tutor_mean':tm,'neutral_mean':nm,'tutor_sd':np.sqrt(tv),
            'neutral_sd':np.sqrt(nv),'delta':tm-nm,'d':d,'n_prompts':n}


def choose_neurons(stats,k=20,seed=0):
    d=stats['d'];depth,width=d.shape
    candidates=[int(i) for i in np.argsort(-d.ravel(),kind='stable')
                if np.isfinite(d.ravel()[i]) and d.ravel()[i]>0 and stats['tutor_sd'].ravel()[i]>0]
    if len(candidates)<k:raise ValueError('Not enough positive finite neurons')
    selected=candidates[:k]
    def record(flat):
        l,n=divmod(int(flat),width)
        return {'id':f'{l}.{n}','layer':l,'neuron':n,
                **{key:float(stats[key][l,n]) for key in ('d','tutor_mean','neutral_mean','tutor_sd','neutral_sd','delta')}}
    tutor=[record(i) for i in selected]
    rng=np.random.default_rng(seed);controls=[];signs=[]
    for _ in range(5):
        used=set(selected);group=[]
        for neuron in tutor:
            l=neuron['layer']
            choices=[l*width+j for j in range(width) if l*width+j not in used and stats['tutor_sd'][l,j]>0 and np.isfinite(d[l,j])]
            if not choices:raise ValueError('Insufficient non-tutor control neurons in layer')
            flat=int(rng.choice(choices));used.add(flat);group.append(record(flat))
        controls.append(group)
        signs.append(rng.choice([-1,1],size=k).tolist())
    return {'k':k,'indexing':'zero-based L.N; input to down_proj','random_seed':seed,
            'tutor':tutor,'controls':controls,'control_signs':signs,'n_prompts_per_system':stats['n_prompts']}


def push_activations(x,indices,values,*,generated):
    if not generated:return x
    y=x.clone() if hasattr(x,'clone') else x.copy()
    y[...,indices]+=values
    return y


def reasoning_span(text):
    tags=list(re.finditer(r'<\s*(/?)\s*think\s*>',text,re.IGNORECASE))
    explicit=bool(tags and not tags[0].group(1) and not text[:tags[0].start()].strip())
    depth=0 if explicit else 1
    start=0
    for tag in tags:
        if not tag.group(1):
            if depth==0:start=tag.end()
            depth+=1
        else:
            depth-=1
            if depth==0:return start,tag.start()
    return None


def score_generation(response,gold,finish_reason):
    if finish_reason not in ('stop','length'):raise ValueError('finish reason must be stop/length')
    reasoning,final=split_thinking(response)
    words=final.split();grams=[tuple(words[i:i+4]) for i in range(max(0,len(words)-3))]
    rate=(len(grams)-len(set(grams)))/len(grams) if grams else 0.
    empty=not final.strip();truncated=finish_reason=='length';rep=rate>.3
    return {**score_response(final,gold),'reasoning':reasoning,'final_reply':final,
            'truncated':truncated,'excluded_from_rates':empty,'empty':empty,'repetition':rep,
            'repetition_rate':rate,'truncation_without_final':truncated and empty,
            'degenerate':empty or rep or (truncated and empty),
            'tutor_mention':bool(TUTOR_PATTERN.search(reasoning))}


def summarize_trajectory(values):
    values=np.asarray(values,dtype=float)
    if values.ndim!=2 or values.shape[1]!=6 or not np.isfinite(values).all():raise ValueError('finite tokens x six sets required')
    n=len(values)
    if not n:return {'slopes':None,'first_quarter':None,'deciles':None}
    t=np.linspace(0,1,n)
    slopes=((t-t.mean())[:,None]*values).sum(0)/np.square(t-t.mean()).sum() if n>=2 else None
    bins=np.minimum((t*10).astype(int),9)
    return {'slopes':slopes.tolist() if slopes is not None else None,
            'first_quarter':values[:max(1,math.ceil(n/4))].mean(0).tolist(),
            'deciles':[values[bins==i].mean(0).tolist() if (bins==i).any() else None for i in range(10)]}


def choose_test_templates(seconds_per_request,spent_seconds,reserve_seconds=300):
    if not math.isfinite(seconds_per_request) or seconds_per_request<=0 or spent_seconds<0:raise ValueError('invalid timing')
    for templates in (2,1):
        if spent_seconds+100*templates*7*seconds_per_request+reserve_seconds<=CAP_SECONDS:return templates
    raise ValueError('Reduced test grid exceeds GPU budget')


def remote_timeout(previous_seconds,startup_seconds=120,shutdown_seconds=180):
    value=int(CAP_SECONDS-previous_seconds-startup_seconds-shutdown_seconds)
    if value<60:raise ValueError('GPU budget exhausted')
    return value


def validate_tracking_sources(requests,raw,scored):
    """Match exact frozen requests and existing Phase 1b labels before GPU use."""
    expected={r['phase1b_id']:r for r in requests if r['split']=='test'}
    labels={}
    for row in scored:
        if row.get('model')==MODEL and row.get('id') in expected:
            if row['id'] in labels:raise ValueError('duplicate Phase 1b label')
            labels[row['id']]=row
    selected={}
    from src.models import MODELS
    for row in raw:
        rid=row.get('id')
        if row.get('model')!=MODEL or rid not in expected:continue
        if rid in selected:raise ValueError('duplicate Phase 1b trace')
        source=expected[rid]
        for key in ('messages','seed','problem_idx','gold','question','thinking','cue','format','template_idx','arm'):
            if row.get(key)!=source[key]:raise ValueError(f'Phase 1b {key} mismatch: {rid}')
        if row.get('model_revision')!=MODELS[MODEL]:raise ValueError('Phase 1b revision mismatch')
        if rid not in labels:raise ValueError(f'missing Phase 1b label: {rid}')
        label=labels[rid];computed=score_generation(row['response'],row['gold'],row['finish_reason'])
        for key in ('leak','commit','correct_commit','excluded_from_rates'):
            if label.get(key)!=computed[key]:raise ValueError(f'Phase 1b label mismatch: {key} {rid}')
        if label['response']!=row['response']:raise ValueError('label response differs from raw')
        selected[rid]={**source,'response':row['response'],'finish_reason':row['finish_reason'],
                       'source_rendered_prompt_sha256':row.get('rendered_prompt_sha256'),
                       'leak':bool(label['leak']),'excluded_from_rates':label['excluded_from_rates']}
    if set(selected)!=set(expected) or len(selected)!=400:raise ValueError('Require 400 matched Phase 1b test traces and labels')
    return [selected[r['phase1b_id']] for r in requests if r['phase1b_id'] in selected]


def close_attempt(*,call_known,elapsed,previous_seconds,status):
    """An uncertain paid submission never becomes a zero-cost retry."""
    return {'status':status if call_known else 'submission_unresolved',
            'observed_wall_seconds':elapsed,
            'wall_seconds':elapsed if call_known else max(0.,CAP_SECONDS-previous_seconds),
            'accounting_basis':'observed allocation wall time' if call_known else 'entire remaining exposure reserved pending reconciliation'}


def system_end_boundary(input_ids,attention_mask,im_end):
    """First real end-of-turn, never an EOS-valued left-padding token."""
    matches=(input_ids==im_end)*attention_mask
    if not bool(matches.any(-1).all()):raise ValueError('System end token missing from a valid position')
    return matches.argmax(-1)+1
