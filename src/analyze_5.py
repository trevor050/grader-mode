"""Strict Phase 5 grids and final-only paired problem-cluster inference.

Default main analysis requires both models and the original 200 Phase 1 problems,
with two templates per retained cell. Explicit model_scope validates one complete
family independently for sequential reporting, with the same sample/audit gates. Only the registered whole-cell drop
stages are accepted, and only with --allow-cell-drops. --pilot allows partial
collection for descriptive checks, never hypothesis decisions. Missing human
attempt/numeric or human final-boundary labels keep inference pending; no
model labels are substituted.

Generation pins live on each row (sampling may instead be a nested object):
model_revision, temperature=.7, top_p=.95, top_k=-1, max_tokens=4096,
dtype/quantization/reasoning_effort from the pinned model registry.
All factors share the same integer seed for each problem/template pair.
The original sample is independently checked against the locally cached,
pinned GSM8K dataset. This module never downloads data or calls a provider.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.build_prompts import DATASET_REVISION, sample_problems, request_seed
from src.models_5 import MODELS_5, PACKAGE_VERSIONS, RENDER_DATE, GEMMA, NEMOTRON, GLM, REGISTRATION_COMMIT, ORIGINAL_REGISTRATION_COMMIT
from src.score import read_rows
from src.score_5 import METRICS, REVISIONS, PREFILLS, score_rows, active_models
from src.phase5_parser_agreement import agreement, load_labels, write_audit_sample, write_splitter_sample
from src.analyze_4 import _interval, _sort_problem

MAIN = [('main', t, f, 'none') for t,f in itertools.product((False,True), ('plain','explicit'))]
PREFILL = [('prefill', True, 'explicit', p) for p in PREFILLS]


def _cell(row):
    return row['arm'], row['thinking'], row['format'], row['prefill']


@lru_cache(maxsize=1)
def _original_sample():
    from datasets import Dataset
    from datasets.config import HF_DATASETS_CACHE
    path = Path(HF_DATASETS_CACHE)/'openai___gsm8k'/'main'/'0.0.0'/DATASET_REVISION/'gsm8k-test.arrow'
    if not path.is_file():
        raise ValueError('original pinned GSM8K sample unavailable locally; build the Phase 5 requests first')
    return {p['problem_idx']:(p['gold'],p['question']) for p in sample_problems(Dataset.from_file(str(path)))}


def _manifest_identities(manifest):
    sample = manifest.get('sample') if isinstance(manifest, dict) else manifest
    if not isinstance(sample, list):
        raise ValueError('problem manifest requires sample list')
    result = {}
    for p in sample:
        if not isinstance(p, dict):
            raise ValueError('malformed problem manifest sample')
        pid = p.get('problem_id', p.get('problem_idx'))
        question = p.get('question', p.get('problem'))
        if (type(pid) not in (int,str) or (isinstance(pid,str) and not pid.strip())
                or not isinstance(question,str) or not question.strip()
                or type(p.get('gold')) is not int or pid in result):
            raise ValueError('malformed or duplicate problem manifest sample')
        if ('problem_id' in p and 'problem_idx' in p and
                (type(p['problem_id']) is not type(p['problem_idx']) or p['problem_id'] != p['problem_idx'])):
            raise ValueError('conflicting problem manifest identity aliases')
        result[pid] = p['gold'],question
    return result


def _pin(row, key):
    sampling = row.get('sampling', {})
    if not isinstance(sampling, dict):
        raise ValueError('sampling pins must be an object')
    values = [obj[key] for obj in (row, sampling) if key in obj]
    if not values:
        raise ValueError(f'missing generation pin: {key}')
    if any(type(v) is not type(values[0]) or v != values[0] for v in values[1:]):
        raise ValueError(f'conflicting generation pin: {key}')
    return values[0]


def _validate_pins(rows, *, pilot):
    seeds = {}
    provenance = {}
    has_tokenizer_pin = any('tokenizer_revision' in item for item in rows)
    for r in rows:
        if not isinstance(r.get('run_id'),str) or not r['run_id'].strip():
            raise ValueError('run_id required')
        stage = r.get('stage', 'main')
        if stage not in ('main','full','pilot') or (stage == 'pilot' and not pilot):
            raise ValueError('pilot/stage data requires explicit pilot analysis')
        if r.get('pilot', False) is not False and not pilot:
            raise ValueError('pilot data requires explicit pilot analysis')
        for name in ('run_id', 'phase', 'stage', 'pilot', 'package_versions',
                     'render_date', 'source_sha256', 'implementation_sha256', 'registration_commit'):
            value = r.get(name)
            if name in provenance and value != provenance[name]:
                raise ValueError(f'mixed or missing {name} within run')
            provenance[name] = value
        for name in ('budget_shortened', 'budget_stopped', 'partial'):
            if r.get(name, False) is not False:
                raise ValueError(f'{name} data is not a complete registered cell')
        registration = r.get('registration_commit')
        if registration is not None and registration != REGISTRATION_COMMIT:
            if not (pilot and r['model'] == GEMMA and registration == ORIGINAL_REGISTRATION_COMMIT):
                raise ValueError('unregistered registration_commit')
        spec = MODELS_5[r['model']]
        expected = {'model_revision': REVISIONS[r['model']], 'temperature':.7, 'top_p':.95,
            'top_k':-1, 'max_tokens':4096, 'dtype':spec['dtype'],
            'quantization':spec['quantization'], 'reasoning_effort':spec['reasoning_effort']}
        for name, expected_value in expected.items():
            value = _pin(r,name)
            if (isinstance(value,bool) or value != expected_value
                    or (type(expected_value) is int and type(value) is not int)):
                raise ValueError(f'unregistered generation pin {name}: {value!r}')
        if (has_tokenizer_pin or not pilot) and r.get('tokenizer_revision') != REVISIONS[r['model']]:
            raise ValueError('unregistered or missing tokenizer_revision')
        sampling = r.get('sampling', {})
        if 'skip_special_tokens' in sampling and sampling['skip_special_tokens'] is not False:
            raise ValueError('special tokens must be preserved')
        if 'stop_token_ids' in sampling and sampling['stop_token_ids'] != MODELS_5[r['model']]['stop_token_ids']:
            raise ValueError('unregistered stop_token_ids')
        if 'package_versions' in r:
            packages = r['package_versions']
            if not isinstance(packages,dict) or any(packages.get(k) != v for k,v in PACKAGE_VERSIONS.items()):
                raise ValueError('unregistered package_versions')
        if 'render_date' in r and r['render_date'] != RENDER_DATE:
            raise ValueError('unregistered render_date')
        seed = r.get('seed')
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('seed must be uint32 integer')
        key = r['problem_id'], r['template']
        if key in seeds and seeds[key] != seed:
            raise ValueError(f'unmatched seeds across arms/models for {key}')
        seeds[key] = seed
        if not pilot:
            if type(r['problem_id']) is not int or seed != request_seed(f"phase5:p{r['problem_id']:04d}:t{r['template']}"):
                raise ValueError('seed must match the original Phase 5 problem/template seed')


def validate_fallback_provenance(proof, *, allow_fixture=False):
    """Validate an outcome-blind, prospective runtime fallback record.

    Runtime proof names the pinned Nemotron, RTX 5090, and exactly two failed
    runtime_smoke attempts (startup then fix), without violation scoring. Each
    attempt has distinct attempt_id, started_unix, ended_unix and evidence text.
    decision_unix follows the failures and precedes first_generation_unix.
    Unit fixtures explicitly declare runtime failure; they never enable inference.
    """
    if not isinstance(proof,dict) or proof.get('outcomes_consulted') is not False:
        raise ValueError('fallback provenance requires outcomes_consulted=false')
    if proof.get('kind') == 'unit_fixture':
        if not allow_fixture or proof.get('explicit_runtime_failure') is not True:
            raise ValueError('explicit runtime failure fixture provenance required; not full inference evidence')
        return 'unit_fixture'
    if (proof.get('kind') != 'runtime_startup_failure'
            or proof.get('model') != NEMOTRON
            or proof.get('model_revision') != REVISIONS[NEMOTRON]
            or proof.get('registration_commit') != REGISTRATION_COMMIT
            or proof.get('gpu') != 'RTX 5090'
            or proof.get('selected_before_generation') is not True):
        raise ValueError('fallback provenance must pin prospective Nemotron RTX 5090 runtime failure')
    def timestamp(value):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError('fallback provenance requires finite positive timestamps')
        return value
    attempts = proof.get('attempts')
    if not isinstance(attempts,list) or len(attempts)!=2:
        raise ValueError('fallback provenance requires startup and one fix attempt')
    ids, ends, starts = set(), [], []
    for attempt,stage in zip(attempts,('startup','fix')):
        if (not isinstance(attempt,dict) or attempt.get('stage')!=stage
                or attempt.get('status')!='failed' or attempt.get('purpose')!='runtime_smoke'
                or attempt.get('violation_scored') is not False
                or not isinstance(attempt.get('attempt_id'),str) or not attempt['attempt_id'].strip()
                or attempt['attempt_id'] in ids
                or not isinstance(attempt.get('evidence'),str) or not attempt['evidence'].strip()):
            raise ValueError('fallback provenance requires two distinct failed, unscored runtime attempts')
        start,end=timestamp(attempt.get('started_unix')),timestamp(attempt.get('ended_unix'))
        if end<start:
            raise ValueError('fallback provenance timestamps are not prospective')
        starts.append(start);ends.append(end);ids.add(attempt['attempt_id'])
    decision=timestamp(proof.get('decision_unix'))
    first=timestamp(proof.get('first_generation_unix'))
    if starts[1]<ends[0] or decision<ends[1] or first<=decision:
        raise ValueError('fallback provenance requires failures then decision before generation')
    return 'runtime_startup_failure'


def validate_grid(rows, *, expected_problems=200, allow_cell_drops=False,
                  problem_manifest=None, pilot=False, allow_glm_fallback=False,
                  fallback_provenance=None, model_scope=None):
    if type(expected_problems) is not int or expected_problems<1:
        raise ValueError('expected_problems must be positive integer')
    pair=active_models(rows)
    if model_scope is not None and (model_scope not in MODELS_5 or {r['model'] for r in rows}!={model_scope}):
        raise ValueError('model_scope requires exactly the requested registered family')
    models=(model_scope,) if model_scope else pair
    fallback_kind=None
    if GLM in models:
        if not allow_glm_fallback:
            raise ValueError('GLM fallback requires explicit allow_glm_fallback')
        fallback_kind=validate_fallback_provenance(fallback_provenance,
            allow_fixture=pilot or expected_problems!=200)
        if fallback_kind=='runtime_startup_failure':
            for r in rows:
                if r['model'] != GLM:
                    continue
                started=r.get('generation_started_unix')
                if started is None and not pilot and expected_problems == 200:
                    raise ValueError('full GLM fallback requires raw generation_started_unix')
                if started is not None and (isinstance(started,bool) or not isinstance(started,(int,float))
                        or not math.isfinite(started) or started<fallback_provenance['first_generation_unix']):
                    raise ValueError('fallback decision must precede GLM full generation')
    elif fallback_provenance is not None:
        raise ValueError('fallback provenance supplied without GLM fallback rows')
    _validate_pins(rows,pilot=pilot)
    identities,actual={},defaultdict(set)
    for r in rows:
        p=r['problem_id'];identity=r['gold'],r['question']
        if p in identities and identities[p]!=identity:
            raise ValueError(f'inconsistent gold or question for problem {p}')
        identities[p]=identity
        key,observation=(r['model'],p),(_cell(r),r['template'])
        if observation in actual[key]:
            raise ValueError(f'duplicate factor tuple in grid: {key} {observation}')
        actual[key].add(observation)
    if len(identities)!=expected_problems:
        raise ValueError(f'grid requires exactly {expected_problems} problems; got {len(identities)}')
    drop_stage=None
    if not pilot:
        if {r['model'] for r in rows}!=set(models):
            raise ValueError('complete grid requires both registered models')
        for stage in range(4):
            expected={}
            for model in models:
                controls_dropped=stage >= (1 if model==GEMMA else 2)
                cells=[c for c in PREFILL if not controls_dropped or c[3] in ('empty','reminder')]
                if model==GEMMA or stage<3:
                    cells=MAIN+cells
                expected[model]=set(itertools.product(cells,(0,1)))
            if all(actual[(m,p)]==expected[m] for m in models for p in identities):
                drop_stage=stage;break
        if drop_stage is None or (drop_stage and not allow_cell_drops):
            raise ValueError('incomplete/mixed grid; only registered drop order with explicit allow_cell_drops is accepted')
    original_checked=expected_problems==200 and not pilot
    if original_checked and identities!=_original_sample():
        raise ValueError('grid must match original 200 Phase 1 problem identities/gold/questions')
    manifest_checked=problem_manifest is not None
    if manifest_checked:
        if not isinstance(problem_manifest,dict):
            raise ValueError('problem manifest requires registration_commit and sample')
        registration=problem_manifest.get('registration_commit')
        if registration!=REGISTRATION_COMMIT and not (pilot and set(r['model'] for r in rows)=={GEMMA}
                and registration==ORIGINAL_REGISTRATION_COMMIT):
            raise ValueError('problem manifest must pin amendment registration_commit')
        for key,value in (('dataset_revision',DATASET_REVISION),('sample_seed',0),('sample_size',200)):
            if key in problem_manifest and (type(problem_manifest[key]) is not type(value) or problem_manifest[key]!=value):
                raise ValueError(f'unregistered problem manifest {key}')
        manifest_ids=_manifest_identities(problem_manifest)
        if pilot:
            if any(manifest_ids.get(p)!=identity for p,identity in identities.items()):
                raise ValueError('pilot identities do not match problem manifest')
        elif identities!=manifest_ids:
            raise ValueError('grid identities/gold/questions do not match problem manifest')
        if expected_problems==200 and manifest_ids!=_original_sample():
            raise ValueError('manifest must match original 200 Phase 1 sample')
    dropped=[]
    if drop_stage:
        for model in models[:min(drop_stage,2)]:
            dropped.extend({'model':model,'arm':'prefill','thinking':True,'format':'explicit','prefill':p}
                for p in ('neutral','irrelevant'))
        if drop_stage==3:
            dropped.extend({'model':pair[1],'arm':a,'thinking':t,'format':f,'prefill':p} for a,t,f,p in MAIN)
    if model_scope and not pilot:
        present_cells={_cell(r) for r in rows}
        dropped=[dict(model=model_scope,arm=a,thinking=t,format=f,prefill=p)
                 for a,t,f,p in MAIN+PREFILL if (a,t,f,p) not in present_cells]
    return {'mode':'pilot' if pilot else 'full' if not drop_stage else f'cell_drop_stage_{drop_stage}',
        'drop_stage':drop_stage,'dropped_cells':dropped,'n_problems':len(identities),
        'models':list(models),'second_model':pair[1],'model_scope':model_scope,'fallback':fallback_kind,
        'fallback_provenance':fallback_provenance,
        'expected_problems':expected_problems,'templates':[0,1],'n_responses':len(rows),
        'problem_manifest_checked':manifest_checked,
        'original_sample_checked':original_checked or (manifest_checked and expected_problems==200)}


def _unavailable(reason):
    return {'status':'unavailable', 'reason':reason, 'estimate':None, 'ci95':[None,None],
            'tested':False, 'holds':None}


def _summarize(rows, *, eligible, pilot=False):
    problems = sorted({r['problem_id'] for r in rows}, key=_sort_problem)
    result = {'n_problems':len(problems), 'n_responses':len(rows), 'models':{}}
    if not problems:
        return result
    indices = np.random.default_rng(0).integers(0,len(problems),size=(2000,len(problems)))
    for model in active_models(rows):
        selected = [r for r in rows if r['model'] == model]
        if not selected:
            continue
        cells = sorted({_cell(r) for r in selected})
        by_key = defaultdict(list)
        for r in selected:
            by_key[(r['problem_id'],_cell(r))].append(r)
        # Partial pilot cells have no CIs or contrasts; cell counts still describe
        # every collected reply. Never silently bootstrap unpaired available cases.
        complete = all(len(by_key[(p,c)]) == 2 for p in problems for c in cells)
        values = np.array([[[np.mean([r[m] for r in by_key[(p,c)]]) for m in METRICS]
            for c in cells] for p in problems],dtype=float) if complete else None
        point = values.mean(axis=0) if complete else None
        boot = np.concatenate([values[b].mean(axis=1) for b in np.array_split(indices,16)]) if complete else None
        cell_results = []
        for j,c in enumerate(cells):
            cell_rows = [r for r in selected if _cell(r) == c]
            metrics = {}
            for k,m in enumerate(METRICS):
                metric = _interval(point[j,k],boot[:,j,k]) if complete else {
                    'estimate':float(np.mean([r[m] for r in cell_rows])), 'ci95':[None,None],
                    'tested':False, 'holds':None, 'status':'descriptive_partial_pilot'}
                if m == 'final_reply_length':
                    metric['unit'] = 'characters'
                else:
                    metric['count'] = sum(r[m] for r in cell_rows)
                metrics[m] = metric
            cell_results.append({'arm':c[0], 'thinking':c[1], 'thinking_mode':'on' if c[1] else 'off',
                'request_type':c[2], 'format':c[2], 'prefill':c[3], 'n_responses':len(cell_rows), 'metrics':metrics})
        contrasts = {}
        for metric in ('violation','leak'):
            k = METRICS.index(metric)
            def linear(terms, tested):
                if not complete or any(c not in cells for c,_ in terms):
                    return _unavailable('partial_pilot_or_registered_cell_drop')
                paired = sum(w*values[:,cells.index(c),k] for c,w in terms)
                return _interval(paired.mean(),paired[indices].mean(axis=1),
                    tested=tested and metric=='violation' and not pilot, eligible=eligible)
            out = {}
            on_plain=('main',True,'plain','none'); off_plain=('main',False,'plain','none')
            on_exp=('main',True,'explicit','none'); off_exp=('main',False,'explicit','none')
            out['H18' if model == GEMMA else 'H21'] = linear([(on_exp,1),(off_exp,-1),(on_plain,-1),(off_plain,1)],True)
            out['on_plain_minus_off_plain'] = linear([(on_plain,1),(off_plain,-1)],False)
            h = 'H19' if model == GEMMA else 'H20'
            for comparator in ('empty','neutral','irrelevant'):
                out[f'{h}_{comparator}_minus_reminder'] = linear([
                    (('prefill',True,'explicit',comparator),1),
                    (('prefill',True,'explicit','reminder'),-1)],True)
            full = all(out[f'{h}_{c}_minus_reminder'].get('status') != 'unavailable' for c in ('empty','neutral','irrelevant'))
            tested = metric == 'violation' and not pilot
            out[f'{h}_joint'] = {'tested':tested and full,
                'holds':all(out[f'{h}_{c}_minus_reminder']['holds'] for c in ('empty','neutral','irrelevant')) if eligible and tested and full else None,
                'rule':'all_three_directional_lower_bounds_above_zero',
                'status':'available' if full else 'unavailable'}
            if not full:
                out[f'{h}_reduced'] = dict(out[f'{h}_empty_minus_reminder'],
                    reason='registered_control_cell_drop', scope='reminder_vs_empty_only')
            contrasts[metric] = out
        result['models'][model] = {'n_problems':len({r['problem_id'] for r in selected}),
            'n_responses':len(selected), 'cells':cell_results, 'contrasts':contrasts}
    return result


def analyze(rows, *, expected_problems=200, allow_cell_drops=False,
            problem_manifest=None, pilot=False, labels=(), allow_glm_fallback=False,
            fallback_provenance=None, model_scope=None):
    if hasattr(rows,'to_dict'):
        rows = rows.to_dict('records')
    rows = score_rows(rows)
    grid = validate_grid(rows, expected_problems=expected_problems, allow_cell_drops=allow_cell_drops,
        problem_manifest=problem_manifest, pilot=pilot, allow_glm_fallback=allow_glm_fallback,
        fallback_provenance=fallback_provenance, model_scope=model_scope)
    audit = agreement(rows, labels, model_scope=model_scope)
    reasons = []
    if pilot:
        reasons.append('pilot')
    if expected_problems != 200:
        reasons.append('synthetic_or_smoke_problem_count')
    if not grid['problem_manifest_checked']:
        reasons.append('problem_manifest_not_checked')
    if any(r['protocol_failure'] for r in rows):
        reasons.append('protocol_failures')
    if audit['status'] != 'complete' or audit['committed_number']['status'] != 'complete':
        reasons.append('parser_audit_pending')
    if audit['splitter']['status'] != 'complete':
        reasons.append('splitter_audit_pending')
    eligible = not reasons
    report = _summarize(rows,eligible=eligible,pilot=pilot)
    report.update(phase='phase5', grid=grid,
        bootstrap={'resamples':2000,'seed':0,'unit':'problem_id','interval':'percentile 95%',
                   'paired_across':'templates,cells,arms,metrics,models'},
        inference={'eligible':eligible,'reasons':reasons,'parser_audit_required_by_preregistration':True,
                   'splitter_audit_required_by_preregistration':True,
                   'analysis_before_hand_audit_authorized':False},
        parser_agreement=audit,
        methods={'denominator':'all_requests_including_empty_finals', 'final_reply_length_unit':'characters',
            'correct_attempt':'any_detected_value_equals_gold','wrong_attempt':'attempt_and_not_correct_attempt',
            'attempt_window':'numeric_token_start_offset_0_through_12_after_marker',
            'cap_hit_no_final':'finish_reason_length_and_empty_final', 'multiplicity_adjustment':None})
    report['gold_ge_10'] = _summarize([r for r in rows if r['gold'] >= 10],eligible=eligible,pilot=pilot)
    return report


def _number(value):
    return 'NA' if value is None else f'{value:.6g}'


def render_markdown(report):
    lines = ['# Phase 5', '', f"Grid: `{report['grid']['mode']}`; {report['n_problems']} problems; {report['n_responses']} responses.",
        '', 'Bootstrap: 2,000 paired problem resamples; seed 0; percentile 95% intervals.',
        '', 'Denominator: all requests, including cap-hit replies with no final. Length: characters.',
        '', f"Inference eligible: {report['inference']['eligible']}; reasons: {', '.join(report['inference']['reasons']) or 'none'}.",
        '', f"Parser audit: {report['parser_agreement']['status']}; {report['parser_agreement']['n_labeled']}/{report['parser_agreement']['n_required']} labeled.",
        '', f"Splitter audit: {report['parser_agreement']['splitter']['status']}; {report['parser_agreement']['splitter']['n_labeled']}/{report['parser_agreement']['splitter']['n_required']} human finals; {report['parser_agreement']['splitter']['n_pending']} pending; {report['parser_agreement']['splitter']['n_disagreements']} disagreements.",
        '', 'Dropped cells: '+json.dumps(report['grid']['dropped_cells'])]
    for model,audit in report['parser_agreement']['models'].items():
        split = audit['splitter']
        lines += ['',f"{model}: {audit['n_labeled']}/40 labels; attempt agreement {_number(audit['agreement'])}; numeric agreement {_number(audit['committed_number']['agreement'])}.",
            f"Splitter: {split['status']}; {split['n_labeled']}/40 human finals; {split['n_pending']} pending; agreement {_number(split['agreement'])}; {split['n_disagreements']} disagreements."]
    for title,data in (('All gold',report),('Gold >= 10',report['gold_ge_10'])):
        lines += ['',f'## {title}', '',f"Problems: {data['n_problems']}; responses: {data['n_responses']}."]
        for model,item in data['models'].items():
            lines += ['',f'### {model}', '', '| Arm | Thinking | Request | Prefill | N | Metric | Count | Estimate | CI 95% |',
                '| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |']
            for cell in item['cells']:
                for metric,value in cell['metrics'].items():
                    lines.append(f"| {cell['arm']} | {cell['thinking_mode']} | {cell['format']} | {cell['prefill']} | {cell['n_responses']} | {metric} | {value.get('count','')} | {_number(value['estimate'])} | [{', '.join(_number(x) for x in value['ci95'])}] |")
            lines += ['', '| Outcome | Contrast | Estimate | CI 95% | Tested | Holds |',
                '| --- | --- | ---: | --- | --- | --- |']
            for metric,contrasts in item['contrasts'].items():
                for name,value in contrasts.items():
                    lines.append(f"| {metric} | {name} | {_number(value.get('estimate'))} | [{', '.join(_number(x) for x in value.get('ci95',[None,None]))}] | {value['tested']} | {value['holds']} |")
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('results/phase5'))
    parser.add_argument('--problem-manifest',type=Path)
    parser.add_argument('--allow-cell-drops',action='store_true')
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--allow-glm-fallback',action='store_true')
    parser.add_argument('--fallback-provenance',type=Path)
    parser.add_argument('--expected-problems',type=int,default=200,help='Non-200 counts are descriptive only')
    parser.add_argument('--labels',type=Path)
    parser.add_argument('--model-scope',choices=tuple(MODELS_5),help='Validate one complete family independently')
    args = parser.parse_args()
    try:
        rows = score_rows(read_rows(args.input))
        manifest = json.loads(args.problem_manifest.read_text()) if args.problem_manifest else None
        report = analyze(rows, expected_problems=args.expected_problems, problem_manifest=manifest,
            allow_cell_drops=args.allow_cell_drops,pilot=args.pilot,
            allow_glm_fallback=args.allow_glm_fallback,
            fallback_provenance=json.loads(args.fallback_provenance.read_text()) if args.fallback_provenance else None,
            labels=load_labels(args.labels) if args.labels else (), model_scope=args.model_scope)
        write_audit_sample(rows,args.output_dir/'parser-audit-sample.jsonl')
        write_splitter_sample(rows,args.output_dir/'splitter-audit-sample.jsonl')
        report['parser_agreement']['sample_path'] = str(args.output_dir/'parser-audit-sample.jsonl')
        report['parser_agreement']['splitter']['sample_path'] = str(args.output_dir/'splitter-audit-sample.jsonl')
        args.output_dir.mkdir(parents=True,exist_ok=True)
        (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        (args.output_dir/'summary.md').write_text(render_markdown(report))
    except (ValueError,OSError,TypeError) as error:
        parser.error(str(error))
    print(f'Analyzed {len(rows)} responses -> {args.output_dir}/summary.{{json,md}}')


if __name__ == '__main__':
    main()
