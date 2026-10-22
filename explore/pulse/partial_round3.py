"""Validate the interrupted checkpoint without manufacturing a completed 40-row arm."""
import json,pathlib,hashlib
import numpy as np
from gate import load,score
from pulse_core import measures
from closed_loop_core import chunk_stats

def summarize_partial(raw):
    raw=pathlib.Path(raw);out=raw/'out';cp=out/'closed_loop_target.checkpoint.json'
    scored=raw/'partial-target-scored.json'
    if not cp.exists() or not scored.exists():return None
    states=json.loads(cp.read_text());decoded=json.loads(scored.read_text())
    gates={r['idx']:r for r in map(json.loads,(out/'gates.jsonl').read_text().splitlines()) if r['cond']=='rl_tutor'}
    manifest=json.loads((raw/'manifest.json').read_text());expected=[r['idx'] for r in manifest['generation']]
    assert [r['idx'] for r in states]==[r['idx'] for r in decoded]==expected
    chunks=[json.loads(l) for l in (out/'closed_loop_target.chunks.jsonl').read_text().splitlines()]
    by={}
    for c in chunks:
        key=(c['idx'],c['chunk']);assert key not in by;by[key]=c
    detector=load(raw/'detector.json');threshold=json.loads((out/'monitor.json').read_text())['threshold']
    summary=[];used=set()
    for s,r in zip(states,decoded):
        assert s['ids']==r['token_ids'] and s['done']==r['done'] and len(s['ids'])<=4096
        assert r['gold']==gates[s['idx']]['gold']
        if s['done']:assert s['ids'][-1] in {248044,248046} and 248069 in s['ids']
        else:assert 248069 not in s['ids']
        for k,v in measures(r['text'],r['finish_reason'],r['gold']).items():assert r[k]==v
        cs=[];prefix=list(gates[s['idx']]['prompt_token_ids']);new=[]
        for i in range(s['stats']['reasoning_chunks']):
            c=by[(s['idx'],i)];used.add((s['idx'],i));cs.append(c)
            assert c['prefix_tokens']==len(prefix)+len(new)
            assert c['retained_tokens']==len(c['token_ids'])<=32
            assert c['push']==(bool(gates[s['idx']]['gate']) if i==0 else cs[i-1]['monitor_score']>=threshold)
            if c['monitor_score'] is not None:
                assert len(c['positions'])==min(8,c['reasoning_tokens'])
                v=np.asarray(c['features']);assert v.shape==(len(c['positions']),60) and np.isfinite(v).all()
                assert abs(float(np.mean(score(v,detector)))-c['monitor_score'])<1e-12
                assert c['positions']==list(range(len(prefix)+len(new)+c['reasoning_tokens']-len(c['positions']),len(prefix)+len(new)+c['reasoning_tokens']))
            new+=c['token_ids']
        assert s['stats']==chunk_stats(cs)
        assert s['ids'][:len(new)]==new
        if not s['done']:assert s['ids']==new
        summary.append(dict(idx=s['idx'],retained_tokens=len(s['ids']),completed_at_checkpoint=s['done'],
          finished_observed=r['finished'],leak_observed=r['leak'],numeric_box_observed=r['boxed'],rule_observed=r['rule_restatement'],**s['stats']))
    sums={k:sum(s['stats'][k] for s in states) for k in ['pushed_tokens','push_on_chunks','switches','retrigger','reasoning_chunks','hooked_prefix_tokens']}
    return dict(status='INCOMPLETE - controller teardown during active target generation',n_started=40,
        completed_finals_observed=sum(r['finished'] for r in decoded),pending_at_checkpoint=sum(not s['done'] for s in states),
        finished_leaks_observed=sum(r['leak'] for r in decoded),numeric_boxes_observed=sum(r['boxed'] for r in decoded),
        generated_rule_observed_all_prefixes=sum(r['rule_restatement'] for r in decoded),
        generated_rule_completed_subset=sum(r['rule_restatement'] for r in decoded if r['done']),
        retained_tokens=sum(len(s['ids']) for s in states),**sums,
        discarded_lookahead_at_checkpoint=sum(by[key]['discarded_lookahead'] for key in used),
        sampled_tokens_at_checkpoint=sum(len(s['ids']) for s in states)+sum(by[key]['discarded_lookahead'] for key in used),
        checkpoint_matched_chunk_records=len(used),newer_chunk_records_not_merged=len(set(by)-used),
        newer_pushed_reasoning_tokens_not_merged=sum(c['reasoning_tokens'] for key,c in by.items() if key not in used and c['push']),
        complete_cell_leak_rate=None,complete_cell_finish_rate=None,unfinished_as_model_failure=None,
        per_sequence=summary,
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [cp,scored,out/'closed_loop_target.chunks.jsonl']},
        caveat='Checkpoint and chunk files were copied at different instants. Only checkpoint-matched chunks are combined; no missing final is scored as a non-leak, and interruption is not a model length-limit failure.')

if __name__=='__main__':
    raw=pathlib.Path(__file__).resolve().parents[2]/'results/raw/pulse3'
    r=summarize_partial(raw);(raw/'partial-summary.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ['per_sequence','source_hashes','caveat']}))
