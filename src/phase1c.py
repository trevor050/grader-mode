"""Phase 1c measured budget gate, heldout calibration, and staged execution."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
from src.models_1c import MODELS_1C
from src.score import score_file, read_rows

RATE = 1.95 / 3600
BUDGET = 10.0


def metrics_for(run_id):
    return [json.loads(p.read_text()) for p in sorted((Path('results/raw')/run_id).glob('*.metrics.json'))]


@contextmanager
def launch_lock():
    """One local Phase 1c launcher owns the ledger across its entire stage."""
    import fcntl
    path=Path('results/raw/phase1c-launch.lock')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another Phase 1c launcher is active') from error
        try:
            yield
        finally:
            fcntl.flock(handle,fcntl.LOCK_UN)


def read_attempts():
    path=Path('results/phase1c/gpu-attempts.json')
    return json.loads(path.read_text()) if path.exists() else []


def accounted_seconds():
    attempts=read_attempts()
    if any(a['status']=='running' for a in attempts):
        raise ValueError('unfinished GPU attempt must be reconciled before another spawn')
    recorded={(a['run_id'],a['model']) for a in attempts}
    unrecorded=0.0
    for path in Path('results/raw').glob('phase1c-*/*.metrics.json'):
        metric=json.loads(path.read_text())
        if (path.parent.name,metric['model']) not in recorded:
            unrecorded+=metric['function_seconds']
    return sum(a['wall_seconds'] for a in attempts)+unrecorded



def remaining_gpu_seconds():
    return BUDGET/RATE-accounted_seconds()-600


def project_budget(smoke_metrics, spent_seconds=0):
    """Use observed tokens/request and tok/s, with no model-size heuristics."""
    estimates={}
    for metric in smoke_metrics:
        ladder=MODELS_1C[metric['model']]['ladder']
        throughput=metric['output_tokens']/metric['generation_seconds']
        # At most measured smoke throughput; Phase 1's observed 3,900 tok/s is
        # retained as a reference scenario, never assumed as an OLMo speedup.
        avg=metric['output_tokens']/metric['requests']
        requests=52*(200 if ladder=='instruct' else 100)
        generation=requests*avg/throughput
        estimates[ladder]={'requests_per_checkpoint':requests,'mean_output_tokens':avg,
            'output_tokens_per_second':throughput,'generation_seconds_per_checkpoint':generation,
            'setup_seconds_per_checkpoint':metric['setup_seconds'],
            'gpu_seconds_three_checkpoints':3*(generation+metric['setup_seconds'])}
    if set(estimates)!={'instruct','think'}:
        raise ValueError('Both ladders require measured smoke throughput')
    # Explicit allowance for startup/billing outside function timer: 60 s per
    # GPU invocation across two smoke, two calibration, and six main calls.
    allocation_reserve_seconds=600
    main_seconds=sum(e['gpu_seconds_three_checkpoints'] for e in estimates.values())
    projected=spent_seconds+main_seconds+allocation_reserve_seconds
    return {'budget_usd':BUDGET,'hourly_usd':1.95,'spent_function_seconds':spent_seconds,
        'ladder_estimates':estimates,'main_function_seconds':main_seconds,
        'allocation_reserve_seconds':allocation_reserve_seconds,
        'total_gpu_seconds':projected,'total_gpu_usd':projected*RATE,
        'launch_full':projected*RATE<=BUDGET,
        'method':'Observed tokens/request and output tok/s from each ladder smoke, same measured throughput for its three checkpoints; no parameter scaling.',
        'phase1_qwen8b_reference_output_tokens_per_second':3900,
        'reference_only_main_generation_seconds':sum(e['requests_per_checkpoint']*e['mean_output_tokens']*3/3900 for e in estimates.values()),
        'limitations':'Point projection; other checkpoints and full batching may differ. Startup reserve is an allowance, not measured GPU billing. CPU/RAM/storage excluded.'}


def allocate_timeouts(projection):
    """Distribute remaining budget by measured runtime, capped at 5,400 s/call."""
    remaining=BUDGET/RATE-projection['spent_function_seconds']-projection.get('allocation_reserve_seconds',600)
    if remaining <= 0 or projection['main_function_seconds'] <= 0:
        raise ValueError('No remaining GPU budget')
    return {ladder:min(5400,int(remaining*(spec['gpu_seconds_three_checkpoints']/3)/projection['main_function_seconds']))
            for ladder,spec in projection['ladder_estimates'].items()}


def calibration_result(rows):
    result={'phase':'phase1c-calibration','threshold':0.8,'comparison':'>',
            'analysis_included':False,'ladders':{}}
    for ladder in ('instruct','think'):
        group=[r for r in rows if r.get('ladder')==ladder]
        if len({r['id'] for r in group}) != len(group):
            raise ValueError('duplicate calibration IDs')
        if len(group)!=80 or len({r['problem_idx'] for r in group})!=20:
            raise ValueError('Calibration requires 20 problems x 4 templates per ladder')
        if any(r['phase']!='phase1c-calibration' or r['checkpoint']!='RLVR' or
               r['cue']!='plain' or r['format']!='none' or r['arm']!='tutor' for r in group):
            raise ValueError('Unexpected calibration cells')
        valid=[r for r in group if not r['excluded_from_rates']]
        if not valid:
            raise ValueError('Calibration has no valid final replies')
        leak=sum(r['leak'] for r in valid)/len(valid)
        result['ladders'][ladder]={'model':group[0]['model'],
            'model_revision':group[0]['model_revision'],'n_responses':len(group),'n_valid':len(valid),
            'n_leaks':sum(r['leak'] for r in valid),'leak_rate':leak,
            'n_truncated':sum(r['truncated'] for r in group),
            'n_empty_finals':len(group)-len(valid),
            'system_prompt':'prompts/system_tutor_strong.txt' if leak>.8 else 'prompts/system_tutor.txt',
            'problem_indices':sorted({r['problem_idx'] for r in group})}
    return result


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def write_budget_stop(output_dir, projection, calibration):
    out=Path(output_dir)
    diagnostic_counts={}
    all_metrics=[]
    for stage in ('smoke','calibration'):
        run_id='phase1c-'+stage
        all_metrics.extend(metrics_for(run_id))
        scored=Path('results/raw')/run_id/'scored.jsonl'
        if scored.exists():
            rows=list(read_rows(scored))
            diagnostic_counts[stage]={l:{
                'n_responses':len(group), 'n_truncated':sum(r['truncated'] for r in group),
                'n_empty_finals':sum(r['excluded_from_rates'] for r in group),
                'n_truncated_empty':sum(r['truncated'] and r['excluded_from_rates'] for r in group)}
                for l in ('instruct','think') if (group:=[r for r in rows if r['ladder']==l])}
    function_seconds=sum(m['function_seconds'] for m in all_metrics)
    for min_gold,name in ((None,'summary'),(10,'summary_gold_ge_10')):
        result={'phase':'phase1c','status':'budget_stop_after_calibration','min_gold':min_gold,
            'main_responses':0,'main_problems':0,'calibration_data_included':False,
            'smoke_data_included':False,'hypothesis_contrasts':None,
            'projected_total_gpu_usd':projection['total_gpu_usd'],'budget_usd':BUDGET,
            'gpu_function_seconds':function_seconds,'gpu_function_usd':function_seconds*RATE,
            'accounted_wall_seconds':accounted_seconds(),'accounted_wall_usd':accounted_seconds()*RATE,
            'diagnostic_truncation_counts':diagnostic_counts,
            'launch_full':False,'calibration_path':'results/phase1c/calibration.json',
            'smoke_summary_path':'results/phase1c-smoke/summary.md',
            'system_prompts':{l:d['system_prompt'] for l,d in calibration['ladders'].items()}}
        write_json(out/(name+'.json'),result)
        lines=['# Phase 1c execution summary','',
            'Status: budget_stop_after_calibration.',f'Gold minimum: {min_gold}.','',
            f"Projected total GPU cost: ${projection['total_gpu_usd']:.4f}; approved limit: $10.00.",
            'Main-run responses: 0. Main-run problems: 0.',
            f'Measured GPU function time: {function_seconds:.3f} seconds; estimated GPU cost: ${function_seconds*RATE:.4f}.',
            f'Budget-accounted time: {accounted_seconds():.3f} seconds; cost estimate: ${accounted_seconds()*RATE:.4f} (historical function clocks plus new attempt wall clocks).',
            'H6–H8, common-ceiling contrasts, and exploratory main-run estimates: unavailable.',
            'Calibration observations in analysis: 0. Smoke observations in main analysis: 0.','',
            '| Ladder | Main tutor prompt decision |','|---|---|']
        lines += [f"| {l} | {d['system_prompt']} |" for l,d in calibration['ladders'].items()]
        lines += ['', '| Diagnostic stage | Ladder | Responses | Truncated | Empty finals | Truncated empty |', '|---|---|---|---|---|---|']
        for stage,ladders in diagnostic_counts.items():
            for ladder,c in ladders.items():
                lines.append(f"| {stage} | {ladder} | {c['n_responses']} | {c['n_truncated']} | {c['n_empty_finals']} | {c['n_truncated_empty']} |")
        lines += ['','Calibration: [calibration.json](calibration.json).',
            'Projection: [budget-projection.json](budget-projection.json).',
            'Smoke diagnostics: [separate summary](../phase1c-smoke/summary.md).']
        (out/(name+'.md')).write_text('\n'.join(lines)+'\n')


def run_generation(stage,run_id,models='rlvr',timeout=5400):
    subprocess.run([sys.executable,'-m','modal','run','-m','src.modal_app_1c',
        '--stage',stage,'--run-id',run_id,'--models',models,'--timeout-seconds',str(timeout)],check=True)
    raw=Path('results/raw')/run_id
    score_file(raw,raw/'scored.jsonl')
    return list(read_rows(raw/'scored.jsonl'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('smoke','calibration','full'),required=True)
    args=parser.parse_args()
    out=Path('results/phase1c');out.mkdir(parents=True,exist_ok=True)
    if args.stage=='smoke':
        run_generation('smoke','phase1c-smoke')
        subprocess.run([sys.executable,'-m','src.analyze_1c','--input','results/raw/phase1c-smoke/scored.jsonl',
            '--output-dir','results/phase1c-smoke'],check=True)
        return
    smoke=metrics_for('phase1c-smoke')
    if len(smoke)!=2:
        parser.error('Run the two-checkpoint smoke first')
    if args.stage=='calibration':
        rows=run_generation('calibration','phase1c-calibration')
        write_json(out/'calibration.json',calibration_result(rows))
    calibration=metrics_for('phase1c-calibration')
    if len(calibration)!=2 or not (out/'calibration.json').exists():
        parser.error('Complete calibration before the full run')
    spent=accounted_seconds()
    projection=project_budget(smoke,spent)
    write_json(out/'budget-projection.json',projection)
    print(json.dumps(projection,indent=2),flush=True)
    if not projection['launch_full']:
        write_budget_stop(out,projection,json.loads((out/'calibration.json').read_text()))
    if args.stage=='full':
        if not projection['launch_full']:
            parser.error('Measured projection exceeds $10; full run blocked after calibration')
        # The Modal entrypoint allocates per-ladder timeouts from this budget.
        run_generation('full','phase1c-full',models='all')
        subprocess.run([sys.executable,'-m','src.analyze_1c','--input','results/raw/phase1c-full/scored.jsonl',
            '--output-dir',str(out)],check=True)


if __name__=='__main__':
    main()
