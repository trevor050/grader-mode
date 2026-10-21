"""CPU contracts and paired numerical report for the registered neutral control."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from src.phase3 import RATE, build_requests, choose_neurons, reasoning_span, summarize_trajectory, validate_tracking_sources
from src.phase3_resume import digest, file_hash, require, verify_hashes
from src.phase3_validation import _local_sources, validate_trajectories
from src.score import read_rows

CAP_USD = 2.50
CAP_SECONDS = CAP_USD/RATE
RESERVE = 120
SOURCE = 'phase3-20261020-v2'
SETS = ('tutor', *(f'control_{i}' for i in range(5)))


def limits(attempts, now):
    require(isinstance(attempts, list), 'attempt ledger required')
    for a in attempts:
        require(a.get('status') in ('completed', 'failed', 'budget_stopped', 'failed_or_cancelled')
                and type(a.get('wall_seconds')) in (int, float)
                and math.isfinite(a['wall_seconds']) and a['wall_seconds'] >= 0,
                'reconcile uncertain allocation before launch')
    previous = sum(a['wall_seconds'] for a in attempts)
    exposure = CAP_SECONDS-previous
    require(exposure > RESERVE+180, 'neutral GPU cap exhausted')
    return dict(previous_seconds=previous, exposure_seconds=exposure,
                timeout_seconds=int(exposure-RESERVE-120),
                absolute_deadline_unix=now+exposure-RESERVE)


def project(pilot_seconds, total_traces, spent_seconds, previous_seconds):
    require(pilot_seconds > 0 and total_traces > 0 and spent_seconds >= 0 and previous_seconds >= 0, 'invalid pilot')
    # Pilot is the longest four traces, padded to their maximum length.
    seconds = previous_seconds+spent_seconds+pilot_seconds*math.ceil(total_traces/4)*1.3+RESERVE
    return dict(projected_gpu_usd=seconds*RATE, fits=seconds <= CAP_SECONDS,
                headroom=1.3, shutdown_reserve_seconds=RESERVE)


def replay_positions(prompt, response, offsets):
    span = reasoning_span(response)
    if span is None:
        return []
    prefix = len(prompt)
    return [j for j, (a, b) in enumerate(offsets)
            if b > a and a >= prefix+span[0] and b <= prefix+span[1]]


def assert_tokens(tutor, neutral, saved, rid):
    assert list(tutor) == list(neutral) == list(saved), f'reasoning token mismatch: {rid}'


def load_inputs(root):
    from src.analyze_3 import _validate_tracking
    root = Path(root)
    out, raw = root/'results/phase3', root/'results/raw'/SOURCE
    manifest = json.loads((out/'manifest.json').read_text())
    # DESIGN gained the requested addendum; all other frozen inputs stay exact.
    old_design = subprocess.check_output(['git', 'show', 'a18dfb7:DESIGN.md'], cwd=root)
    require(hashlib.sha256(old_design).hexdigest() == manifest['source_sha256']['DESIGN.md'], 'historical DESIGN hash changed')
    require((root/'DESIGN.md').read_bytes() == subprocess.check_output(['git', 'show', '843719c:DESIGN.md'], cwd=root), 'registered addendum changed')
    local_manifest = {**manifest, 'source_sha256': {k:v for k,v in manifest['source_sha256'].items() if k != 'DESIGN.md'}}
    source_raw, scored, _ = _local_sources(root, local_manifest)
    requests = build_requests(manifest['sample'])
    traces = validate_tracking_sources(requests, list(source_raw), list(scored))
    tracks = list(read_rows(raw/'tracking.jsonl'))
    _validate_tracking(tracks)
    validate_trajectories(tracks, raw/'trajectories.npz')
    neurons = json.loads((out/'neurons.json').read_text())
    with np.load(out/'screen-statistics.npz', allow_pickle=False) as archive:
        stats = {k:archive[k] for k in archive.files}
    stats['n_prompts'] = 400
    require(choose_neurons(stats) == neurons, 'screen selection changed')
    with np.load(raw/'trajectories.npz', allow_pickle=False) as a:
        ids = a['ids'].tolist()
        tokens = {rid:a['token_ids'][a['offsets'][i]:a['offsets'][i+1]].tolist() for i,rid in enumerate(ids)}
    by_id = {r['id']:r for r in tracks}
    sources = {r['id']:r for r in traces}
    require(set(ids) == set(sources) and len(ids) == 400, '400 matched traces required')
    trace_rows = []
    for rid in ids:
        src, saved = sources[rid], by_id[rid]
        for k in ('problem_idx','selection_rank','template_idx','gold','leak','excluded_from_rates'):
            require(type(src[k]) is type(saved[k]) and src[k] == saved[k], 'source labels changed')
        require(hashlib.sha256(src['response'].encode()).hexdigest() == saved['source_response_sha256'], 'response changed')
        trace_rows.append({**src, 'saved_tokens':tokens[rid], 'tutor_prompt_sha256':saved['rendered_prompt_sha256']})
    paths = [out/'neurons.json',out/'selection.json',out/'screen-statistics.npz',out/'manifest.json',
             raw/'tracking.jsonl',raw/'trajectories.npz',root/'DESIGN.md',root/'src/phase3_neutral.py',
             root/'src/modal_app_3_neutral.py',root/'prompts/system_neutral.txt']
    hashes = {str(p.relative_to(root)):file_hash(p) for p in paths}
    return dict(traces=trace_rows, neurons=neurons, neutral_system=(root/'prompts/system_neutral.txt').read_text(),
                source_hashes=hashes, historical_source_hashes=manifest['source_sha256'])


def analyze(rows):
    from src.analyze_3 import _validate_tracking
    require(isinstance(rows, list) and len(rows) == 400, 'complete 400-trace grid required')
    _validate_tracking(rows)
    require(all(r.get('token_identity_verified') is True for r in rows), 'token identity must be verified')
    for r in rows:
        for k in ('reading_tutor', 'reading_neutral'):
            a = np.asarray(r[k], dtype=float)
            require(a.shape == (6,) and np.isfinite(a).all(), 'finite six-set reading values required')
    problems = sorted({r['problem_idx'] for r in rows})
    index = {p:i for i,p in enumerate(problems)}
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(problems), (2000, len(problems)))
    weights = np.array([np.bincount(d, minlength=len(problems)) for d in draws])
    result = dict(bootstrap=dict(resamples=2000, seed=0, unit='problem_idx', interval='percentile 95%', paired=True),
                  n_traces=400, n_problems=len(problems), token_identity_verified=400,
                  weighting='trace mean; whole-problem paired resamples; source empty-final labels excluded only from outcome strata',
                  deciles=[], reading=[], slopes=[])
    for kind in ('deciles','reading','slopes'):
        for d in (range(10) if kind == 'deciles' else [None]):
            for si, name in enumerate(SETS):
                for outcome, leak in (('all',None),('leak',True),('nonleak',False)):
                    sums, counts = np.zeros(len(problems)), np.zeros(len(problems))
                    used = []
                    for r in rows:
                        if leak is not None and (r['excluded_from_rates'] or r['leak'] != leak):
                            continue
                        if kind == 'reading':
                            value = r['reading_tutor'][si]-r['reading_neutral'][si]
                        else:
                            a = r[kind] if d is None else (None if r[kind] is None else r[kind][d])
                            if a is None:
                                continue
                            value = a[si]
                        i = index[r['problem_idx']]
                        sums[i] += value; counts[i] += 1; used.append(r)
                    denominator = weights@counts
                    boot = np.divide(weights@sums, denominator, out=np.full(2000,np.nan), where=denominator>0)
                    valid = boot[np.isfinite(boot)]
                    row = dict(set=name, outcome=outcome, estimate=float(sums.sum()/counts.sum()) if counts.sum() else None,
                               ci95=np.percentile(valid,[2.5,97.5]).tolist() if len(valid) else [None,None],
                               n_traces=len(used), n_problems=int((counts>0).sum()), undefined_resamples=int((~np.isfinite(boot)).sum()))
                    if d is not None:row['decile'] = d+1
                    if kind == 'reading':
                        for arm in ('tutor','neutral'):
                            row[arm+'_mean'] = float(np.mean([r['reading_'+arm][si] for r in used])) if used else None
                    result[kind].append(row)
    return result


def write_report(root, run_id):
    root = Path(root); raw = root/'results/raw'/run_id; out = root/'results/phase3'
    metrics = json.loads((raw/'neutral-metrics.json').read_text())
    execution = json.loads((out/'neutral-execution.json').read_text())
    require(metrics['status'] == 'completed' and execution['zero_containers_verified'], 'completed allocation and shutdown required')
    verify_hashes(root, metrics['manifest']['source_hashes'])
    original = root/'results/raw'/SOURCE
    tracks = list(read_rows(original/'tracking.jsonl')); neutral = list(read_rows(raw/'neutral-tracking.jsonl'))
    validate_trajectories(neutral, raw/'neutral-trajectories.npz')
    by_id = {r['id']:r for r in neutral}; differences = []
    with np.load(original/'trajectories.npz', allow_pickle=False) as a, np.load(raw/'neutral-trajectories.npz', allow_pickle=False) as b:
        require(np.array_equal(a['ids'], b['ids']) and np.array_equal(a['offsets'], b['offsets']), 'trajectory alignment changed')
        assert np.array_equal(a['token_ids'],b['token_ids']), 'saved reasoning tokens changed'
        for i, rid in enumerate(a['ids']):
            r = dict(by_id[str(rid)])
            values = a['scores'][a['offsets'][i]:a['offsets'][i+1]].astype(float)-b['scores'][b['offsets'][i]:b['offsets'][i+1]].astype(float)
            r.update(summarize_trajectory(values)); differences.append(r)
    # Recheck every identity and label against the original table, independently of order.
    old = {r['id']:r for r in tracks}
    for r in differences:
        require(all(r[k] == old[r['id']][k] for k in ('problem_idx','selection_rank','template_idx','gold','leak','excluded_from_rates','closed_reasoning','n_reasoning_tokens','source_response_sha256')), 'original tracking metadata changed')
    report = analyze(differences)
    report.update(run_id=run_id, source_run_id=SOURCE, exploratory=True, manifest=metrics['manifest'], execution=execution,
                  reasoning_tokens=int(metrics['reasoning_tokens']), pilot=metrics['pilot'],
                  reading_positions='valid tokens strictly after system im_end, through rendered assistant prefix, before generated think tag')
    (out/'neutral-control.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (out/'neutral-control.csv').open('w',newline='') as f:
        table = [{**{k:v for k,v in r.items() if k != 'ci95'}, 'ci95_low':r['ci95'][0], 'ci95_high':r['ci95'][1]} for r in report['deciles']]
        writer = csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    with (out/'neutral-traces.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=['id','problem_idx','leak','excluded_from_rates','n_reasoning_tokens',*SETS]);writer.writeheader()
        for r in differences:
            writer.writerow({**{k:r[k] for k in ('id','problem_idx','leak','excluded_from_rates','n_reasoning_tokens')}, **dict(zip(SETS,r['slopes'] or [None]*6))})
    lines = ['# Phase 3 neutral control','', 'Contrast: tutor prompt minus neutral prompt; unchanged tutor-reading z-score reference.',
             'Bootstrap: 2,000 problem-cluster resamples; seed 0; 95% percentile intervals.',
             f"Traces: 400; problems: 100; identical reasoning-token spans: 400; reasoning tokens: {report['reasoning_tokens']}.",
             f"GPU allocation upper bound: ${execution['gpu_usd_upper']:.6f}; zero containers: true."]
    for kind in ('deciles','reading','slopes'):
        lines += ['',f'## {kind}','','| Set | Outcome | Decile | Traces | Estimate | CI low | CI high | Tutor reading | Neutral reading |','|---|---|---|---|---|---|---|---|---|']
        for r in report[kind]:
            fields = [r['set'],r['outcome'],r.get('decile',''),r['n_traces'],r['estimate'],*r['ci95'],r.get('tutor_mean',''),r.get('neutral_mean','')]
            lines.append('| '+' | '.join(f'{x:.6f}' if isinstance(x,float) else str(x) for x in fields)+' |')
    (out/'neutral-control.md').write_text('\n'.join(lines)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True)
    args = p.parse_args()
    from src.build_prompts import ROOT
    write_report(ROOT,args.run_id)

if __name__ == '__main__':main()
