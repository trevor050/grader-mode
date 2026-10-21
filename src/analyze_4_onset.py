"""Phase 4 exploratory onset: frozen inputs, token alignment and paired CIs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import numpy as np
from src.phase3 import build_requests, choose_neurons, reasoning_span, validate_tracking_sources
from src.phase3_resume import digest, file_hash, require
from src.phase3_validation import _local_sources, validate_trajectories
from src.score import read_rows

CAP_USD = 1.50
GPU_RATE = .000542
# Conservative per-second envelope: L40S + up to 4 physical cores + 32 GiB.
# Modal cpu=4 requests vCPUs, so charging four physical cores overstates CPU.
ALL_IN_RATE = .00070
RESERVE = 120
SOURCE = 'phase3-20261020-v2'
SETS = ('tutor', *(f'control_{i}' for i in range(5)))
POSITIONS = [f'boundary:{i}' for i in range(-64,65)]+[f'reasoning:{i}' for i in range(1,257)]


def budget_limits(attempts, now):
    require(isinstance(attempts,list), 'attempt ledger required')
    seen = set()
    for a in attempts:
        require(isinstance(a,dict) and a.get('status') in ('completed','failed','budget_stopped','failed_or_cancelled')
                and a.get('zero_containers_verified') is True, 'uncertain allocation must be reconciled')
        require(a.get('run_id') not in seen, 'duplicate allocation')
        seen.add(a.get('run_id'))
        require(type(a.get('wall_seconds')) in (int,float) and math.isfinite(a['wall_seconds']) and a['wall_seconds']>=0,
                'invalid charged wall time')
    previous = sum(a['wall_seconds'] for a in attempts)
    exposure = CAP_USD/ALL_IN_RATE-previous
    require(exposure > RESERVE+180, 'aggregate Modal cap exhausted')
    return dict(previous_seconds=previous,exposure_seconds=exposure,
                timeout_seconds=int(exposure-RESERVE-120),
                absolute_deadline_unix=now+exposure-RESERVE)


def onset_positions(prompt,response,offsets):
    tag = re.match(r'\s*(<think>)',response)
    require(tag is not None,'explicit opening <think> required')
    start,end = (len(prompt)+x for x in tag.span(1))
    anchors = [j for j,(a,b) in enumerate(offsets) if a==start and b==end]
    require(len(anchors)==1, 'exact opening tag token required')
    anchor = anchors[0]
    boundary = [j if 0<=j<len(offsets) and offsets[j][1]>offsets[j][0] else -1 for j in range(anchor-64,anchor+65)]
    span = reasoning_span(response)
    reasoning = [] if span is None else [j for j,(a,b) in enumerate(offsets)
                if b>a and a>=len(prompt)+span[0] and b<=len(prompt)+span[1]]
    return dict(boundary=boundary,reasoning=(reasoning[:256]+[-1]*256)[:256],
                all_reasoning=reasoning,anchor=anchor,closed_reasoning=span is not None)


def validate_archive(a,n=400):
    require(type(n) is int and n>0, 'positive trace count required')
    mask=np.asarray(a['mask'])
    require(mask.shape==(n,385) and mask.dtype==np.bool_, 'boolean 385-position mask required')
    for key in ('tutor','neutral','contrast'):
        values=np.asarray(a[key])
        require(values.shape==(n,385,6) and np.isfinite(values[mask]).all(), 'finite six-set scores required')
    require(np.array_equal(a['contrast'][mask],(a['tutor']-a['neutral'])[mask]), 'paired contrast changed')
    for key in ('tutor_token_ids','neutral_token_ids'):
        ids=np.asarray(a[key])
        require(ids.shape==mask.shape and ids.dtype.kind in 'iu' and (ids[mask]>=0).all(), 'valid token IDs required')
    require(np.array_equal(a['tutor_token_ids'][:,129:][mask[:,129:]],
                           a['neutral_token_ids'][:,129:][mask[:,129:]]),'reasoning tokens changed')


def position_summary(contrast,mask,problems,positions,resamples=2000,seed=0):
    contrast=np.asarray(contrast);mask=np.asarray(mask)
    require(contrast.shape==(len(problems),len(positions),6) and mask.shape==contrast.shape[:2]
            and mask.dtype==np.bool_ and np.isfinite(contrast[mask]).all(), 'invalid per-position data')
    require(type(resamples) is int and resamples>0 and type(seed) is int and seed>=0,'invalid bootstrap')
    clusters=sorted(set(problems));index={p:i for i,p in enumerate(clusters)}
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(clusters),(resamples,len(clusters)))
    weights=np.array([np.bincount(d,minlength=len(clusters)) for d in draws])
    counts=np.zeros((len(clusters),len(positions)))
    sums=np.zeros((len(clusters),len(positions),6))
    for i,p in enumerate(problems):
        counts[index[p]]+=mask[i]
        sums[index[p]]+=np.where(mask[i,:,None],contrast[i],0.)
    records=[]
    # Work one position at a time to bound bootstrap memory.
    for j,pos in enumerate(positions):
        total=counts[:,j].sum();den=weights@counts[:,j]
        sample=np.divide(weights@sums[:,j,:],den[:,None],out=np.full((resamples,6),np.nan),where=den[:,None]>0)
        for si,name in enumerate(SETS):
            finite=sample[:,si][np.isfinite(sample[:,si])]
            ci=np.percentile(finite,[2.5,97.5]).tolist() if len(finite) else [None,None]
            records.append(dict(position=pos,set=name,mean=float(sums[:,j,si].sum()/total) if total else None,
                ci95_low=ci[0],ci95_high=ci[1],n_traces=int(total),n_problems=int((counts[:,j]>0).sum()),
                undefined_resamples=int(resamples-len(finite))))
    return records


def load_inputs(root):
    """Verify Phase 3 immutable sources without requiring the old DESIGN tip."""
    from src.analyze_3 import _validate_tracking
    root=Path(root);out=root/'results/phase3';raw=root/'results/raw'/SOURCE
    manifest=json.loads((out/'manifest.json').read_text())
    historical=subprocess.check_output(['git','show','a18dfb7:DESIGN.md'],cwd=root)
    require(hashlib.sha256(historical).hexdigest()==manifest['source_sha256']['DESIGN.md'],'historical DESIGN changed')
    expected=subprocess.check_output(['git','show','f223175:src/modal_app_3.py'],cwd=root)
    require(expected==(root/'src/modal_app_3.py').read_bytes(),'Phase 3 model runner changed')
    historical_hashes={k:v for k,v in manifest['source_sha256'].items() if k!='DESIGN.md'}
    source,scored,_=_local_sources(root,{**manifest,'source_sha256':historical_hashes})
    traces=validate_tracking_sources(build_requests(manifest['sample']),list(source),list(scored))
    tracks=list(read_rows(raw/'tracking.jsonl'));_validate_tracking(tracks)
    validate_trajectories(tracks,raw/'trajectories.npz')
    neurons=json.loads((out/'neurons.json').read_text())
    with np.load(out/'screen-statistics.npz',allow_pickle=False) as a:stats={k:a[k] for k in a.files}
    require(choose_neurons({**stats,'n_prompts':400})==neurons,'cached neuron selection changed')
    with np.load(raw/'trajectories.npz',allow_pickle=False) as a:
        ids=a['ids'].tolist()
        tokens={rid:a['token_ids'][a['offsets'][i]:a['offsets'][i+1]].tolist() for i,rid in enumerate(ids)}
    by_id={r['id']:r for r in tracks};sources={r['id']:r for r in traces}
    require(len(ids)==400 and len(set(ids))==400 and set(ids)==set(sources),'400 original traces required')
    rows=[]
    for rid in ids:
        r,s=sources[rid],by_id[rid]
        require(all(type(r[k]) is type(s[k]) and r[k]==s[k] for k in
            ('problem_idx','selection_rank','template_idx','gold','leak','excluded_from_rates')),'source labels changed')
        require(hashlib.sha256(r['response'].encode()).hexdigest()==s['source_response_sha256'],'source response changed')
        rows.append({**r,'saved_tokens':tokens[rid],'tutor_prompt_sha256':s['rendered_prompt_sha256']})
    paths=['DESIGN.md','prompts/system_tutor.txt','prompts/system_neutral.txt','prompts/templates_1b.json',
           'src/analyze_4_onset.py','src/modal_app_4_onset.py','results/phase3/neurons.json',
           'results/phase3/screen-statistics.npz','results/phase3/manifest.json',
           f'results/raw/{SOURCE}/tracking.jsonl',f'results/raw/{SOURCE}/trajectories.npz']
    return dict(traces=rows,neurons=neurons,neutral_system=(root/'prompts/system_neutral.txt').read_text(),
                historical_source_hashes=historical_hashes,source_hashes={p:file_hash(root/p) for p in paths})


def write_csv(path,rows):
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    from src.build_prompts import ROOT
    out=ROOT/'results/phase4';manifest=json.loads((out/'onset-manifest.json').read_text())
    require(args.run_id==manifest['run_id'],'run ID mismatch')
    execution=json.loads((out/'modal-onset-execution.json').read_text())
    require(execution['zero_containers_verified'] and execution['attempts'][-1]['status']=='completed'
            and execution['aggregate_modal_usd_upper']<=CAP_USD,'completed budget-safe shutdown required')
    package=load_inputs(ROOT)
    require(digest(package)==manifest['payload_sha256'],'analysis inputs changed since execution')
    metrics=json.loads((out/'onset-metrics.json').read_text())
    require(metrics['status']=='completed' and metrics['manifest']==manifest,'completed matching remote execution required')
    require(file_hash(out/'onset-tokens.npz')==metrics['archive_sha256'],'archive checksum mismatch')
    with np.load(out/'onset-tokens.npz',allow_pickle=False) as a:archive={k:a[k] for k in a.files}
    validate_archive(archive)
    require(archive['ids'].tolist()==[r['id'] for r in package['traces']],'trace IDs or order changed')
    metadata=json.loads((out/'onset-traces.json').read_text())
    require(len(metadata)==400 and all(r['token_identity_verified'] for r in metadata),'all token audits required')
    for r,s in zip(metadata,package['traces']):
        require(r['id']==s['id'] and r['source_response_sha256']==hashlib.sha256(s['response'].encode()).hexdigest(),
                'token audit source mismatch')
    problems=[r['problem_idx'] for r in package['traces']]
    table=position_summary(archive['contrast'],archive['mask'],problems,POSITIONS)
    write_csv(out/'onset-positions.csv',table)
    # Predefined window summaries: mean within trace, then equal trace mean.
    windows={'before_tag_64':list(range(64)),'after_tag_64':list(range(65,129)),
             'reasoning_first64':list(range(129,193)),'reasoning_first256':list(range(129,385))}
    values=[];valid=[]
    for positions in windows.values():
        m=archive['mask'][:,positions];count=m.sum(1)
        sums=np.where(m[:,:,None],archive['contrast'][:,positions,:],0.).sum(1)
        values.append(np.divide(sums,count[:,None],out=np.zeros_like(sums),where=count[:,None]>0));valid.append(count>0)
    headlines=position_summary(np.stack(values,axis=1),np.stack(valid,axis=1),problems,list(windows))
    write_csv(out/'onset-windows.csv',headlines)
    report=dict(run_id=args.run_id,n_traces=400,n_problems=len(set(problems)),exploratory=True,
        bootstrap=dict(resamples=2000,seed=0,unit='problem_idx',interval='percentile 95%',paired=True),
        token_identity_verified=400,closed_reasoning=sum(r['closed_reasoning'] for r in metadata),
        boundary='0 is the <think> tag; -64..-1 before; 1..64 after',
        reasoning='1..256 tokens strictly within closed reasoning, excluding tags; absent positions omitted',
        n_position_rows=len(table),windows=headlines,aggregate_modal_usd_upper=execution['aggregate_modal_usd_upper'])
    (out/'onset-summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# Phase 4 exploratory onset values','',
           '400 traces; 100 problems; six cached neuron sets. Tutor minus neutral, in Phase 3 T units.',
           '2,000 paired problem-cluster resamples; seed 0; unadjusted percentile 95% intervals.',
           'Boundary 0 is the `<think>` token; reasoning positions exclude tag tokens. Missing positions are omitted.',
           '', '| Window | Tutor contrast mean [95% CI] | Traces |','|---|---|---|']
    for r in headlines:
        if r['set']=='tutor':lines.append(f"| {r['position']} | {r['mean']:.6f} [{r['ci95_low']:.6f}, {r['ci95_high']:.6f}] | {r['n_traces']} |")
    lines+=['',f"Per-position CSV: {len(table)} rows, 385 positions × six sets. All controls and window values are in the CSVs.",
            f"Verified token identity: 400/400. Closed reasoning: {report['closed_reasoning']}/400.",
            f"Conservative aggregate Modal allocation bound: ${execution['aggregate_modal_usd_upper']:.6f}; zero containers verified."]
    (out/'onset-summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:report[k] for k in ('n_traces','n_problems','n_position_rows','aggregate_modal_usd_upper')}))


if __name__=='__main__':main()
