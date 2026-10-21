"""Publish descriptive pilot evidence and outcome-blind whole-grid costs only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from src.models_5 import GEMMA
from src.phase5_control import BudgetLedger, registered_cells
from src.phase5_parser_agreement import agreement, write_audit_sample, write_splitter_sample
from src.score_5 import score_rows


def cell_id(row):
    if row['model'] != GEMMA:
        raise ValueError('This task authorizes only a Gemma pilot report')
    model = 'gemma'
    if row['arm'] == 'prefill':
        return f"{model}/prefill/{row['prefill']}"
    mode = 'on' if row['thinking'] else 'off'
    return f"{model}/main/{mode}/{row['request_type']}"


def load_pilot(path, model):
    path = Path(path)
    rows = score_rows([json.loads(line) for line in path.read_text().splitlines()])
    metrics = json.loads(path.with_suffix('.metrics.json').read_text())
    if len(rows) != 16 or {r['model'] for r in rows} != {model}:
        raise ValueError('exactly 16 replies from the expected model required')
    if any(r.get('stage') != 'pilot' or r.get('pilot') is not True for r in rows):
        raise ValueError('pilot provenance required')
    if metrics['requests'] != 16 or metrics['requested_count'] != 16:
        raise ValueError('incomplete pilot metrics')
    if metrics['output_tokens'] != sum(len(r['output_token_ids']) for r in rows):
        raise ValueError('token accounting mismatch')
    cells = defaultdict(list)
    for r in rows:
        cells[cell_id(r)].append(r)
        if (r['response'] != r['raw_text'] or r['raw_token_ids'] != r['output_token_ids']
                or r['output_tokens'] != len(r['output_token_ids'])):
            raise ValueError('raw ID preservation failed')
        if hashlib.sha256(r['rendered_prompt'].encode()).hexdigest() != r['rendered_prompt_sha256']:
            raise ValueError('rendered prompt hash mismatch')
    expected = {c['id'] for c in registered_cells() if c['model'] == 'gemma'}
    if set(cells) != expected:
        raise ValueError('pilot does not cover every registered cell')
    for group in cells.values():
        counts = Counter(r['template_idx'] for r in group)
        if counts[0] != counts[1] or set(counts) != {0, 1}:
            raise ValueError('pilot templates are unbalanced')
    return rows, metrics, cells


def report(gemma_path, out, *, modal_hourly=2.70):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    all_rows, means, models = [], {}, {}
    rates = {'gemma': modal_hourly}
    checks, prompts = [], []
    selections = [('gemma', GEMMA, gemma_path)]
    for key, model, path in selections:
        rows, metrics, cells = load_pilot(path, model)
        all_rows.extend(rows)
        for name, group in cells.items():
            means[name] = statistics.mean(r['output_tokens'] for r in group)
        models[key] = dict(model=model, replies=16, output_tokens=metrics['output_tokens'],
            generation_seconds=metrics['generation_seconds'], setup_seconds=metrics['setup_seconds'],
            aggregate_tokens_per_second=metrics['output_tokens']/metrics['generation_seconds'],
            mean_generated_tokens=statistics.mean(r['output_tokens'] for r in rows),
            hourly_usd=rates[key], mean_generated_tokens_by_cell={c:means[c] for c in cells},
            package_versions=metrics['package_versions'], engine=metrics['engine'],
            parse_status_counts=dict(Counter(r['parse_status'] for r in rows)),
            no_final=sum(r['no_final'] for r in rows), cap_hit=sum(r['cap_hit'] for r in rows),
            cap_hit_no_final=sum(r['cap_hit_no_final'] for r in rows),
            protocol_failure=sum(r['protocol_failure'] for r in rows))
        for r in sorted(rows, key=lambda r:r['id']):
            checks.append({k:r[k] for k in ('model','id','arm','thinking','request_type','prefill',
                'template_idx','problem_id','gold','final_reply','parse_status','protocol_failure',
                'unexpected_off_thought','finish_reason','output_tokens','cap_hit','no_final',
                'cap_hit_no_final','initial_channel','rendered_prompt_sha256')} |
                {'raw_response_sha256':hashlib.sha256(r['response'].encode()).hexdigest(),
                 'output_ids_sha256':hashlib.sha256(json.dumps(r['output_token_ids']).encode()).hexdigest(),
                 'boundary_segments':[s for s in r['decoded_token_segments']
                     if s['end']==s['start']+1 and r['output_token_ids'][s['start']] in r['special_token_ids']],
                 'token_aware_replay_matches':True})
            prompts.append({k:r[k] for k in ('model','id','arm','thinking','request_type','prefill',
                'seed','rendered_prompt','rendered_prompt_sha256','prompt_tokens','initial_channel')})
    snapshot = BudgetLedger().snapshot()
    cap = snapshot['cap_microusd']/1_000_000
    previous = snapshot['spent_usd'] + snapshot['outstanding_usd']
    projections, dropped = [], []
    for drop_group in ((), ('gemma/prefill/neutral','gemma/prefill/irrelevant')):
        dropped.extend(drop_group)
        per_model = {}
        for model, item in models.items():
            tokens = sum(400*value for cell,value in means.items()
                         if cell.startswith(model+'/') and cell not in dropped)
            seconds = tokens/item['aggregate_tokens_per_second']
            cost = (item['setup_seconds'] + 1.20*seconds)*rates[model]/3600
            per_model[model] = dict(projected_generated_tokens=tokens,
                projected_generation_seconds=seconds, projected_cost_usd=cost,
                projected_cost_at_launch_reservation_rate_usd=(cost*2.70/rates[model] if model=='gemma' else cost))
        total = previous + .30 + sum(v['projected_cost_usd'] for v in per_model.values())
        projections.append(dict(scope='gemma_only',
            dropped_cells=dropped.copy(), models=per_model,
            prior_exposure_usd=previous, teardown_and_delayed_charge_reserve_usd=.30,
            projected_measured_models_lifetime_total_usd=total,
            measured_models_fit_cap=total<=cap,
            remaining_for_unmeasured_model_usd=cap-total,
            combined_fits_cap=None))
    affordable = next((p for p in projections if p['measured_models_fit_cap']), None)
    result = dict(phase='phase5', stage='pilot', full_grid_started=False,
        inference_eligible=False, cap_usd=cap, original_cap_usd=6.80, models=models, projections=projections,
        registered_fallback_needed=None, minimum_grid_affordable=None,
        gemma_budget_required_drop=None if affordable is None else affordable['dropped_cells'],
        measured_models_minimum_grid_affordable=affordable is not None,
        previous_exposure_usd=previous,
        combined_grid_projection_complete=False,
        H20_H21_status='Nemotron_registered_GLM_unused_fallback_no_pilot_authorized',
        generation_runtime_multiplier=1.20, pilot_rows_reused_for_main=False,
        projection_limit='Sixteen prompts per model measure a small batch; lengths and throughput may vary on 200 problems.',
        spending_limit='Ledger is conservative metered exposure, not settled billing. Consult provider balance snapshots separately.')
    for name, value in [('pilot-summary.json',result), ('split-checks.json',checks),
                        ('rendered-prompts.json',prompts), ('parser-agreement.json',agreement(all_rows))]:
        (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    write_audit_sample(all_rows, out/'parser-audit-sample.jsonl')
    write_splitter_sample(all_rows, out/'splitter-audit-sample.jsonl')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gemma',type=Path,required=True)
    p.add_argument('--modal-hourly',type=float,default=2.70,
                   help='All-in resource rate; default conservative launch reservation rate')
    p.add_argument('--output-dir',type=Path,default=Path('results/phase5-pilot'))
    a=p.parse_args()
    result=report(a.gemma,a.output_dir,modal_hourly=a.modal_hourly)
    print(json.dumps({'models':result['models'],'projections':result['projections']},indent=2))


if __name__=='__main__':
    main()
