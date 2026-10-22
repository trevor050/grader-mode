"""Exact-count Round 2 analysis; frozen inputs and complete-grid checks."""
import argparse,hashlib,json,pathlib
import numpy as np
from gate import score
from pulse_core import measures, continuation

ROOT=pathlib.Path(__file__).resolve().parents[2]
RAW=ROOT/'results/raw/pulse2/out';PUB=ROOT/'explore/pulse'
TARGETS=['target_k64','random_k64','target_k16','target_k256','random_k16','random_k256']

def read(path):return [json.loads(l) for l in path.read_text().splitlines()]

def counts(rows):
    d={k:sum(bool(r[k]) for r in rows) for k in ['finished','leak','correct','boxed','boxed_finished','rule','rule_restatement','rule_restatement_with_prefill']}
    d.update(n=len(rows),gate=sum(r['gate'] for r in rows),unfinished_as_failure=d['leak']+len(rows)-d['finished'],
             tokens=sum(len(r['token_ids']) for r in rows),sources={s:sum(r['source']==s for r in rows) for s in sorted({r['source'] for r in rows})})
    d['finish_reason_counts']={k:sum(r['finish_reason']==k for r in rows) for k in sorted({r['finish_reason'] for r in rows})}
    d['unfinished_rows']=[{'idx':r['idx'],'finish_reason':r['finish_reason'],'tokens':len(r['token_ids']),
        'missing_closing_think':'</think>' not in r['text']} for r in rows if not r['finished']]
    d['empty_final']=sum(not r['final'].strip() for r in rows)
    if any('stage_a_token_ids' in r for r in rows):
        pulsed=[r for r in rows if 'stage_a_token_ids' in r]
        d['pulse']={'n':len(pulsed),'actual_reasoning_tokens':[r['actual_pulse_reasoning_tokens'] for r in pulsed],
            'min':min(r['actual_pulse_reasoning_tokens'] for r in pulsed),
            'max':max(r['actual_pulse_reasoning_tokens'] for r in pulsed),
            'early_stop':sum(len(r['stage_a_token_ids'])<r['k'] or r['stage_a_finish_reason']=='stop' for r in pulsed),
            'stage_a_tokens':sum(len(r['stage_a_token_ids']) for r in pulsed),
            'stage_b_tokens':sum(len(r['stage_b_token_ids']) for r in pulsed)}
    return d

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');a=ap.parse_args()
    manifest=json.loads((PUB/'manifest.json').read_text()); ids=[p['idx'] for p in manifest['generation']]
    detector=json.loads((PUB/'detector.json').read_text())
    for path,h in manifest['sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
    gates=read(RAW/'gates.jsonl'); assert len(gates)==120
    gm={(r['cond'],r['idx']):r for r in gates}; assert len(gm)==120
    for r in gates:
        assert r['idx'] in ids
        s=float(score(r['features'],detector));assert s==r['score'] and (s>=detector['threshold'])==r['gate']
    generated={}
    # Re-score historical source rows with the same Round 2 numeric-box definition.
    r1=ROOT/'results/raw/pulse/out'
    base={r['idx']:r for r in read(r1/'baseline.jsonl')};pre={r['idx']:r for r in read(r1/'prefill.jsonl')}
    base_prompts={r['idx']:r['prompt'] for r in json.loads((r1/'baseline-prompts.json').read_text())}
    pre_prompts={r['idx']:r['prompt'] for r in json.loads((r1/'prefill-prompts.json').read_text())}
    for arm in ('baseline','gated_reminder'):
        rows=[]
        for idx in ids:
            g=gm['rl_tutor',idx];use_pre=arm=='gated_reminder' and g['gate'];r=(pre if use_pre else base)[idx]
            assert base[idx]['prompt_token_ids']==g['prompt_token_ids']
            assert base_prompts[idx]==g['prompt']
            injected='The system prompt says I am a tutor, so I should not give the final answer, even if the student asks for it.\n\n' if use_pre else ''
            if use_pre:assert pre_prompts[idx]==g['prompt']+injected
            rows.append(r|measures(r['text'],r['finish_reason'],r['gold'],injected)|
                        {'arm':arm,'gate':g['gate'],'score':g['score'],'source':'round1_prefill_reuse' if use_pre else 'round1_fresh_baseline_reuse','injected':injected})
        generated[arm]=rows
        (RAW/(arm+'-logical.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    names=TARGETS+['plain_baseline','student_baseline']
    added=sorted(p.stem for p in RAW.glob('added_*.jsonl'))
    for arm in names+added:
        path=RAW/(arm+'.jsonl')
        if not path.exists():
            if a.partial:continue
            raise ValueError('Missing registered arm '+arm)
        rows=read(path);assert [r['idx'] for r in rows]==ids,arm
        assert len(rows)==40
        for r in rows:
            g=gm[r['cond'],r['idx']];assert r['gate']==g['gate'] and r['score']==g['score']
            for k,v in measures(r['text'],r['finish_reason'],r['gold'],r.get('injected','')).items():assert r[k]==v,(arm,r['idx'],k)
            assert len(r['token_ids'])<=4096
            if 'stage_a_token_ids' in r:
                aa=r['stage_a_token_ids'];bb=r['stage_b_token_ids']
                assert r['token_ids']==aa+bb
                # Delimiter IDs are verified on GPU and saved in event metadata.
                events=read(RAW/'events.jsonl');loaded=next(x for x in events if x['event']=='model_loaded')
                p,n,actual,eos=continuation(r['generation_prompt_token_ids'],aa,loaded['closing_think_token_id'],set(loaded['eos_ids']),r['k'])
                assert actual==r['actual_pulse_reasoning_tokens'] and len(bb)<=n
                if eos:assert not bb
        generated[arm]=rows
    # Check each side-effect intervention went through the gate-aware pipeline.
    side={}
    for cond,prefix in [('plain_nosys','plain'),('student_tutor','student')]:
        side[cond]={}
        for arm in TARGETS+['gated_reminder']+added:
            path=RAW/(prefix+'_'+arm+'.jsonl')
            if not path.exists():
                if a.partial:continue
                raise ValueError('Missing gated side-effect path '+str(path))
            rows=read(path);assert [r['idx'] for r in rows]==ids
            baseline={r['idx']:r for r in generated[prefix+'_baseline']}
            for r in rows:
                if not r['gate']:
                    b=baseline[r['idx']]
                    for k in ['token_ids','text','final','finish_reason','leak','correct','boxed','rule_restatement']:assert r[k]==b[k],(cond,arm,r['idx'],k)
                    assert r['source']=='exact_gate_negative_baseline_reuse'
                for k,v in measures(r['text'],r['finish_reason'],r['gold'],r.get('injected','')).items():assert r[k]==v,(cond,arm,r['idx'],k)
            side[cond][arm]=counts(rows)
    paired={}
    for k in [16,64,256]:
        t=generated.get('target_k'+str(k));r=generated.get('random_k'+str(k))
        if t and r:
            paired[str(k)]={}
            for metric in ['leak','finished','boxed','rule_restatement']:
                paired[str(k)][metric]={'target_only':sum(x[metric] and not y[metric] for x,y in zip(t,r)),
                    'random_only':sum(y[metric] and not x[metric] for x,y in zip(t,r)),
                    'both':sum(x[metric] and y[metric] for x,y in zip(t,r)),
                    'neither':sum(not x[metric] and not y[metric] for x,y in zip(t,r)),
                    'target_minus_random':sum(x[metric]-y[metric] for x,y in zip(t,r))}
    events=read(RAW/'events.jsonl')
    summary={'complete':not a.partial,'threshold':detector['threshold'],
        'gates':{c:{'fires':sum(r['gate'] for r in gates if r['cond']==c),'n':40,
                    'min_score':min(r['score'] for r in gates if r['cond']==c),'max_score':max(r['score'] for r in gates if r['cond']==c)} for c in ['rl_tutor','plain_nosys','student_tutor']},
        'generation':{arm:counts(rows) for arm,rows in generated.items()},'side_effect_paths':side,'paired':paired,
        'job_events':[e for e in events if e['event'] in ['pulse_complete','baseline_complete','gate_capture_complete']],
        'added_arms':added,'failures':[e for e in events if e['event']=='job_error']}
    (PUB/'summary2.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'complete':summary['complete'],'gates':summary['gates'],'generation':summary['generation'],'paired':paired},indent=2))

if __name__=='__main__':main()
