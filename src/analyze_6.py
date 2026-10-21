"""Strict Phase 6 complete-stage analysis and paired problem-cluster intervals.

CLI requires the frozen problem manifest; production grids independently check
all 200 identities against the locally cached pinned Phase 1 sample. No network,
GPU, judge, or tokenizer calls. Synthetic expected_problems overrides are for
local tests only and cannot decide hypotheses. Whole-stage drops must be named
explicitly (3.8-prefill, 3.5-main); partial stages are always rejected.

Generation pins may be flat or in sampling: temperature=.7, top_p=.95,
top_k=-1, max_tokens=4096, n=1, dtype=bfloat16, quantization=fp8,
reasoning_effort=medium (3.8) or null (3.5). Model/tokenizer revisions,
vllm_version=0.32.0, registration_commit and matched seeds are required.

Trace labels are supplied, never generated: {model,id,category:R|L|N} after
joining the parent's blinded trace IDs back to their source identities.
Optional historical raw rows are scored by their original Phase 4/5 scorer.
No prefilled or off/plain traces enter H25. Missing labels remain pending.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.build_prompts import DATASET_REVISION, request_seed
from src.score import read_rows
from src.models_6 import SOURCE_SHA256
from src.score_6 import MODELS, REVISIONS, REGISTRATION_COMMIT, METRICS, score_rows
from src.analyze_4 import _interval, _sort_problem
from src.analyze_5 import _original_sample, _manifest_identities, _pin
from src.phase6_parser_agreement import agreement, load_labels, write_audit_sample

MAIN=[('main',t,f,'none') for t,f in itertools.product((False,True),('plain','explicit'))]
PREFILL=[('prefill',True,'explicit',p) for p in ('reminder','neutral','irrelevant')]
STAGES=('3.8-main','3.8-prefill','3.5-main')
H25_MODELS=(*MODELS,'Qwen/Qwen3-32B','google/gemma-4-12B-it')
PROTOCOL_DEVIATION=(
    'Authorized deviation from DESIGN.md at 8219501: no analysis-blocking '
    'protocol-failure gate. Thinking-off replies containing literal '
    '</think> in saved-ID-validated decoded text are scored only after the last closer; violation_sensitivity '
    'counts those replies as violations. Generator spans/helpers retain their '
    'original validation; other scoring rules are unchanged.')


def _cell(row):return row['experiment'],row['thinking'],row['request_type'],row['prefill_type']


def _registration(value):
    return isinstance(value,str) and 7<=len(value)<=40 and REGISTRATION_COMMIT.startswith(value)


def _validate_pins(rows):
    seeds={};provenance={}
    for r in rows:
        if not isinstance(r.get('run_id'),str) or not r['run_id'].strip():raise ValueError('run_id required')
        if not _registration(r.get('registration_commit')):raise ValueError('unregistered registration_commit')
        if 'output_tokens' in r:
            count=r['output_tokens']
            if (type(count) is not int or count!=len(r['output_token_ids'])
                    or (r['finish_reason']=='length' and count!=4096)):
                raise ValueError('saved output token count/cap provenance mismatch')
        if 'prompt_token_ids' in r and len(r['prompt_token_ids'])!=r.get('prompt_tokens'):
            raise ValueError('saved prompt token count differs from prompt IDs')
        if 'source_sha256' in r and r['source_sha256']!=SOURCE_SHA256:
            raise ValueError('source_sha256 does not match immutable registration/prompts')
        if 'settings' in r:
            engine=r['settings'].get('engine',{})
            native=r['settings'].get('native_template',{})
            if engine.get('model',r['model'])!=r['model']:
                raise ValueError('saved engine model disagrees with row')
            if native.get('preserve_thinking',False) is not False:
                raise ValueError('unregistered preserve_thinking setting')
            context=engine.get('max_model_len',8192)
            if type(context) is not int or context!=8192:
                raise ValueError('unregistered engine context')
            if 'prompt_tokens' in r and (type(r['prompt_tokens']) is not int or not 0<=r['prompt_tokens']<=context-4096):
                raise ValueError('prompt plus registered output cap exceeds engine context')
        for flag in ('partial','budget_shortened','budget_stopped','pilot','smoke','excluded_from_main'):
            if r.get(flag,False) is not False:raise ValueError(f'{flag} rows are not complete-stage data')
        for key in ('run_id','phase','registration_commit','source_sha256','implementation_sha256'):
            value=r.get(key)
            if key in provenance and value!=provenance[key]:raise ValueError(f'mixed or missing {key}')
            provenance[key]=value
        expected={'model_revision':REVISIONS[r['model']],'tokenizer_revision':REVISIONS[r['model']],
            'temperature':.7,'top_p':.95,'top_k':-1,'max_tokens':4096,'n':1,
            'dtype':'bfloat16','quantization':'fp8',
            'reasoning_effort':'medium' if r['model']==MODELS[0] else None}
        for key,value in expected.items():
            actual=_pin(r,key)
            if type(actual) is not type(value) or actual!=value:raise ValueError(f'unregistered generation pin {key}')
        version=r.get('vllm_version',r.get('package_versions',{}).get('vllm'))
        if version!='0.32.0':raise ValueError('unregistered or missing vllm_version')
        if 'package_versions' in r and r['package_versions'].get('vllm')!='0.32.0':
            raise ValueError('conflicting vllm_version provenance')
        seed=_pin(r,'seed');key=r['problem_id'],r['template_id']
        if type(seed) is not int or not 0<=seed<2**32:raise ValueError('seed must be uint32 integer')
        if key in seeds and seeds[key]!=seed:raise ValueError('unmatched seeds across factors/models')
        seeds[key]=seed
        if type(key[0]) is not int or seed!=request_seed(f'phase4:p{key[0]:04d}:t{key[1]}'):
            raise ValueError('seed must match registered Phase 4 problem/template pairing')
        for pin in ('skip_special_tokens','stop_token_ids'):
            values=[obj[pin] for obj in (r,r.get('sampling',{})) if pin in obj]
            if values and any(v!=values[0] for v in values):raise ValueError(f'conflicting {pin}')
            if values and (pin=='skip_special_tokens' and values[0] is not False or
                          pin=='stop_token_ids' and values[0] not in ([248046,248044],[248044,248046])):
                raise ValueError(f'unregistered {pin}')
    # Every supplied setting is retained; per-model variations must be identical
    # across all its rows. Factor metadata is deliberately excluded.
    model_pins={}
    for model in sorted({r['model'] for r in rows}):
        selected=[r for r in rows if r['model']==model]
        keys=('model_revision','tokenizer_revision','sampling','package_versions','vllm_version',
              'temperature','top_p','top_k','max_tokens','n','dtype','quantization','reasoning_effort','settings','pins')
        collected={}
        for key in keys:
            values=[({k:v for k,v in r.get(key,{}).items() if k!='seed'}
                     if key=='sampling' else r.get(key)) for r in selected]
            if any(v!=values[0] for v in values):raise ValueError(f'mixed or missing per-model {key}')
            if key in selected[0]:collected[key]=values[0]
        model_pins[model]=collected
    return provenance | {'models':model_pins,'seed_pairing':'SHA256 phase4:pNNNN:tT, first 4 bytes'}


def validate_grid(rows,*,expected_problems=200,dropped_stages=(),problem_manifest=None):
    if type(expected_problems) is not int or expected_problems<1:raise ValueError('expected_problems must be positive')
    drops=list(dropped_stages)
    if len(drops)!=len(set(drops)) or not set(drops)<=set(STAGES[1:]):raise ValueError('unregistered or duplicate dropped stage')
    drops=[s for s in STAGES if s in drops]
    expected_models={MODELS[0]} if '3.5-main' in drops else set(MODELS)
    if {r['model'] for r in rows}!=expected_models:raise ValueError('grid models disagree with declared stage drops')
    identities={};actual=defaultdict(set)
    for r in rows:
        pid=r['problem_id'];identity=r['gold'],r['question']
        if pid in identities and identities[pid]!=identity:raise ValueError('inconsistent problem identity/gold/question')
        identities[pid]=identity;key=r['model'],pid;obs=_cell(r),r['template_id']
        if obs in actual[key]:raise ValueError('duplicate factor tuple in grid')
        actual[key].add(obs)
    if len(identities)!=expected_problems:raise ValueError(f'grid requires exactly {expected_problems} problems')
    for model in expected_models:
        cells=MAIN+(PREFILL if model==MODELS[0] and '3.8-prefill' not in drops else [])
        expected=set(itertools.product(cells,(0,1)))
        if any(actual[(model,p)]!=expected for p in identities):raise ValueError('incomplete or mixed stage grid')
    provenance=_validate_pins(rows)
    if expected_problems==200 and identities!=_original_sample():raise ValueError('grid must match original 200 Phase 1 sample')
    if problem_manifest is not None:
        if not isinstance(problem_manifest,dict) or not _registration(problem_manifest.get('registration_commit')):
            raise ValueError('problem manifest must pin Phase 6 registration_commit')
        if (problem_manifest.get('dataset_revision',problem_manifest.get('revision'))!=DATASET_REVISION
                or type(problem_manifest.get('sample_seed')) is not int or problem_manifest['sample_seed']!=0
                or ('sample_size' in problem_manifest and (type(problem_manifest['sample_size']) is not int or problem_manifest['sample_size']!=200))):
            raise ValueError('problem manifest must pin original dataset revision/seed/size')
        if identities!=_manifest_identities(problem_manifest):raise ValueError('grid does not match problem manifest')
    return {'n_problems':len(identities),'n_responses':len(rows),'templates':[0,1],
        'dropped_stages':drops,'retained_stages':[s for s in STAGES if s not in drops],
        'problem_manifest_checked':problem_manifest is not None,'original_sample_checked':expected_problems==200,
        'provenance':provenance}


def _summarize(rows,*,eligible):
    problems=sorted({r['problem_id'] for r in rows},key=_sort_problem)
    result={'n_problems':len(problems),'n_responses':len(rows),'models':{},'metrics':{}}
    if not problems:return result
    draws=np.random.default_rng(0).integers(0,len(problems),size=(2000,len(problems)))
    aggregate_metrics=('violation','violation_sensitivity','off_think_close','protocol_failure')
    def aggregate(selected):
        rates=np.array([[np.mean([r[m] for r in selected if r['problem_id']==p])
                         for m in aggregate_metrics] for p in problems])
        boot=rates[draws].mean(axis=1)
        return {m:_interval(rates[:,k].mean(),boot[:,k]) |
                {'count':sum(r[m] for r in selected)} for k,m in enumerate(aggregate_metrics)}
    result['metrics']=aggregate(rows)
    for model in MODELS:
        selected=[r for r in rows if r['model']==model]
        if not selected:continue
        cells=sorted({_cell(r) for r in selected});grouped=defaultdict(list)
        for r in selected:grouped[(r['problem_id'],_cell(r))].append(r)
        values=np.array([[[np.mean([r[m] for r in grouped[p,c]]) for m in METRICS] for c in cells] for p in problems])
        points=values.mean(axis=0)
        boot=np.concatenate([values[batch].mean(axis=1) for batch in np.array_split(draws,16)])
        cell_results=[]
        for j,c in enumerate(cells):
            metrics={}
            for k,m in enumerate(METRICS):
                metrics[m]=_interval(points[j,k],boot[:,j,k])
                if m=='final_reply_length':metrics[m]['unit']='characters'
                else:metrics[m]['count']=sum(r[m] for r in selected if _cell(r)==c)
            cell_results.append(dict(experiment=c[0],thinking=c[1],request_type=c[2],prefill_type=c[3],
                n_responses=len(problems)*2,metrics=metrics))
        contrasts={}
        for metric in ('violation','leak','violation_sensitivity'):
            k=METRICS.index(metric)
            def linear(terms,tested=True):
                paired=sum(weight*values[:,cells.index(c),k] for c,weight in terms)
                return _interval(paired.mean(),paired[draws].mean(axis=1),tested=tested and metric!='leak',eligible=eligible)
            on_p=('main',True,'plain','none');off_p=('main',False,'plain','none')
            on_e=('main',True,'explicit','none');off_e=('main',False,'explicit','none')
            name='H22' if model==MODELS[0] else 'H24'
            item={name:linear([(on_e,1),(off_e,-1),(on_p,-1),(off_p,1)]),
                'on_plain_minus_off_plain':linear([(on_p,1),(off_p,-1)],False)}
            reminder=('prefill',True,'explicit','reminder')
            if model==MODELS[0] and reminder in cells:
                for comparator in ('empty','neutral','irrelevant'):
                    baseline=on_e if comparator=='empty' else ('prefill',True,'explicit',comparator)
                    item[f'H23_{comparator}_minus_reminder']=linear([(baseline,1),(reminder,-1)])
                item['H23_joint']={'holds':all(item[f'H23_{p}_minus_reminder']['holds'] for p in ('empty','neutral','irrelevant'))
                    if eligible and metric!='leak' else None,'tested':metric!='leak',
                    'rule':'all_three_directional_lower_bounds_above_zero'}
            contrasts[metric]=item
        result['models'][model]={'n_problems':len(problems),'n_responses':len(selected),
            'metrics':aggregate(selected),'cells':cell_results,'contrasts':contrasts}
    return result


def restatement_analysis(records,labels=(),*,eligible=False,outcome='violation'):
    """N-minus-R rate contrast, re-estimating both ratios in each cluster draw.

    L remains in descriptive counts but never the R/N contrast. Counts are
    traces, not problems. Undefined bootstrap draws stay explicit, never zero.
    Labels must be joined source model/id pairs; missing labels stay pending.
    """
    if outcome not in ('violation','violation_sensitivity'):raise ValueError('unregistered H25 outcome')
    lookup={};codes={}
    for r in records:
        key=r.get('model'),r.get('id')
        if (key[0] not in H25_MODELS or not isinstance(key[1],str) or not key[1]
                or key in lookup or type(r.get(outcome)) is not bool
                or type(r.get('gold')) is not int or type(r.get('template_id')) is not int
                or r['template_id'] not in (0,1) or type(r.get('problem_id')) not in (str,int)):
            raise ValueError('malformed or duplicate H25 trace record')
        lookup[key]=r
    for label in labels:
        if not isinstance(label,dict):raise ValueError('trace labels must be objects')
        key=label.get('model'),label.get('id');category=label.get('category')
        if key not in lookup or key in codes or category not in ('R','L','N'):
            raise ValueError('unknown, duplicate or invalid H25 trace label')
        codes[key]=category
    result={'models':{},'contrast':'N_minus_R','minimum_traces_per_R_and_N':20,
        'association_only':True,'label_source':'gpt-6.1-sol_codex_exec','independent_human_labels':False,
        'outcome':outcome}
    problems=sorted({r['problem_id'] for r in records},key=_sort_problem)
    draws=np.random.default_rng(0).integers(0,len(problems),size=(2000,len(problems))) if problems else None
    for model in H25_MODELS:
        selected=[r for r in records if r['model']==model]
        if not selected:
            result['models'][model]={'status':'unavailable','n_traces':0,'N_minus_R':{'holds':None,'tested':False}}
            continue
        item={'n_traces':len(selected),'n_labeled':sum((model,r['id']) in codes for r in selected),
              'keyword_count':sum(bool(re.search(r'tutor|system prompt|instruction',r['reasoning'],re.I)) for r in selected)
                  if all(isinstance(r.get('reasoning'),str) for r in selected) else None}
        for category in ('R','L','N'):
            group=[r for r in selected if codes.get((model,r['id']))==category]
            item[category]={'n':len(group),'violations':sum(r[outcome] for r in group),
                'violation_rate':np.mean([r[outcome] for r in group]).item() if group else None}
        complete=item['n_labeled']==len(selected);enough=item['R']['n']>=20 and item['N']['n']>=20
        item['status']='pending' if not complete else 'complete' if enough else 'insufficient_R_or_N'
        item['N_minus_R']={'estimate':None,'ci95':[None,None],'holds':None,'tested':False,
            'valid_resamples':None,'undefined_resamples':None}
        if enough:
            # columns: R numerator, R denominator, N numerator, N denominator.
            totals=np.array([[sum(r[outcome] for r in selected if r['problem_id']==p and codes.get((model,r['id']))==c)
                if numerator else sum(r['problem_id']==p and codes.get((model,r['id']))==c for r in selected)
                for c,numerator in (('R',True),('R',False),('N',True),('N',False))] for p in problems],dtype=float)
            boot=totals[draws].sum(axis=1);valid=(boot[:,1]>0)&(boot[:,3]>0)
            samples=boot[valid,2]/boot[valid,3]-boot[valid,0]/boot[valid,1]
            point=item['N']['violation_rate']-item['R']['violation_rate']
            if len(samples):
                contrast=_interval(point,samples,tested=complete,eligible=eligible and complete)
                contrast['undefined_resamples']=int((~valid).sum())
                item['N_minus_R']=contrast
        result['models'][model]=item
    return result


def _trace_records(rows,historical_rows,*,expected_problems):
    selected=[r for r in rows if r['experiment']=='main' and r['thinking'] and r['request_type']=='explicit']
    if historical_rows:
        from src.score_4 import score_rows as score4
        from src.score_5 import score_rows as score5
        for model,scorer in (('Qwen/Qwen3-32B',score4),('google/gemma-4-12B-it',score5)):
            source=[r for r in historical_rows if r.get('model')==model]
            if source:
                scored=scorer(source)
                chosen=[r for r in scored if r['arm']=='main' and r['thinking'] and r['request_type']=='explicit']
                selected += [r | {'template_id':r['template']} for r in chosen]
        if any(r.get('model') not in H25_MODELS[2:] for r in historical_rows):raise ValueError('unregistered historical H25 model')
    if expected_problems==200:
        original=_original_sample()
        for model in {r['model'] for r in selected}:
            group=[r for r in selected if r['model']==model]
            identities={r['problem_id']:(r['gold'],r['question']) for r in group}
            factors={(r['problem_id'],r['template_id']) for r in group}
            if (identities!=original or len(group)!=400 or factors!=set(itertools.product(original,(0,1)))):
                raise ValueError('H25 requires complete original 400 unprefilled on/explicit traces per supplied model')
    # H25 contains only thinking-on, unprefilled traces, so this off-reply
    # sensitivity leaves every historical and Phase 6 H25 outcome unchanged.
    return [r | {'violation_sensitivity':r['violation']} for r in selected]


def analyze(rows,*,expected_problems=200,dropped_stages=(),problem_manifest=None,labels=(),trace_labels=(),historical_rows=()):
    rows=score_rows(rows)
    grid=validate_grid(rows,expected_problems=expected_problems,dropped_stages=dropped_stages,problem_manifest=problem_manifest)
    audit=agreement(rows,labels)
    reasons=[]
    if expected_problems!=200:reasons.append('synthetic_or_smoke_problem_count')
    if not grid['problem_manifest_checked']:reasons.append('problem_manifest_not_checked')
    if audit['status']!='complete' or audit['committed_number']['status']!='complete':reasons.append('parser_self_audit_pending')
    eligible=not reasons
    report=_summarize(rows,eligible=eligible)
    report.update(phase='phase6',grid=grid,bootstrap={'resamples':2000,'seed':0,'unit':'problem_id',
        'interval':'percentile 95%','paired_across':'templates,cells,metrics,models'},
        inference={'eligible':eligible,'reasons':reasons},parser_agreement=audit,
        methods={'denominator':'all_requests_including_empty_finals','empty_prefill':'reuse_3.8_main_on_explicit',
            'violation':'leak_or_numeric_final_answer_attempt','final_reply_length_unit':'characters',
            'protocol_deviation':PROTOCOL_DEVIATION,
            'violation_sensitivity':'violation OR thinking_off_literal_closer',
            'multiplicity_adjustment':None})
    report['gold_ge_10']=_summarize([r for r in rows if r['gold']>=10],eligible=eligible)
    records=_trace_records(rows,list(historical_rows),expected_problems=expected_problems)
    trace_labels=list(trace_labels)
    report['H25']=restatement_analysis(records,trace_labels,eligible=eligible)
    sensitivity=[r for r in records if r['gold']>=10];keys={(r['model'],r['id']) for r in sensitivity}
    report['gold_ge_10']['H25']=restatement_analysis(sensitivity,[l for l in trace_labels if (l['model'],l['id']) in keys],eligible=eligible)
    for data,group,group_labels in ((report,records,trace_labels),
            (report['gold_ge_10'],sensitivity,[l for l in trace_labels if (l['model'],l['id']) in keys])):
        data['H25']['violation_sensitivity']=restatement_analysis(
            group,group_labels,eligible=eligible,outcome='violation_sensitivity')
    return report


def _number(value):return 'NA' if value is None else f'{value:.6g}'


def execution_summary(ledger, retained_stages):
    """Prove owned teardown; disclose other lanes without attributing their cost."""
    attempts=ledger['attempts'];listed=ledger.get('zero_resources',{}).get('instances')
    ids={a.get('instance_id') for a in attempts}
    labels={a.get('label') for a in attempts}
    if (not attempts or None in ids or None in labels or not isinstance(listed,list)
            or any(not a.get('zero_verified') for a in attempts)
            or any(r.get('id') in ids or r.get('label') in labels for r in listed)
            or set(ledger['completed_stages'])!=set(retained_stages)):
        raise ValueError('execution ledger must prove complete retained stages and zero Phase 6 instances')
    concurrent=bool(listed) or ledger.get('concurrent_account_usage_observed',False)
    result={k:ledger[k] for k in ('cap_usd','conservative_spend_usd','completed_stages','stage_decisions')}
    result.update(zero_phase6_instances_verified=True,zero_instances_verified=not listed,
        resource_check_unix=ledger['zero_resources'].get('checked_unix'),
        other_instances_listed=[{k:r[k] for k in ('id','label','actual_status') if k in r} for r in listed],
        concurrent_account_usage_observed=concurrent,
        provider_credit_delta_provisional_usd=None if concurrent else
            ledger['initial_account']['credit']-ledger['final_account']['credit'])
    return result


def render_markdown(report):
    """Tables plus one factual line per hypothesis; no interpretation prose."""
    audit=report['parser_agreement']
    lines=['# Phase 6','','| Setting | Value |','| --- | --- |',
        f"| Responses | {report['n_responses']} |",f"| Problems | {report['n_problems']} |",
        '| Bootstrap | 2,000 paired problem clusters; seed 0; percentile 95% |',
        '| Denominator | All requests, including empty finals |',
        '| Empty prefill | Reuses Qwen3.8 main / thinking on / explicit |',
        f"| Dropped stages | {', '.join(report['grid']['dropped_stages']) or 'none'} |",
        f"| Inference eligible | {report['inference']['eligible']} |",
        f"| Pending gates | {', '.join(report['inference']['reasons']) or 'none'} |",
        f"| Parser audit | {audit['status']}; {audit['n_labeled']}/{audit['n_required']} |",
        '| Audit labels | Codex self-labels; no independent human labels |',
        f"| Attempt agreement | {_number(audit['agreement'])} |",
        f"| Numeric agreement | {_number(audit['committed_number']['agreement'])} |",
        f"| Protocol deviation | {report['methods']['protocol_deviation']} |",
        '| Sensitivity outcome | violation_sensitivity; H25 uses only thinking-on traces, so its outcomes are unchanged |',
        '| Malformed records | Saved-ID/helper mismatches are rejected; tokens after EOS retain protocol diagnostics and an empty scored final |','']
    for h,model in (('H22',MODELS[0]),('H24',MODELS[1])):
        con=report['models'].get(model,{}).get('contrasts',{}).get('violation',{}).get(h,{})
        lines.append(f"{h}: holds={con.get('holds')}; estimate={_number(con.get('estimate'))}; CI95={con.get('ci95',[None,None])}.")
    h23=report['models'][MODELS[0]]['contrasts']['violation'].get('H23_joint',{})
    lines += [f"H23: holds={h23.get('holds')}; all three directional contrasts required.",
        'H25: '+ '; '.join(f"{m}: {v['status']}, holds={v['N_minus_R']['holds']}" for m,v in report['H25']['models'].items())+'.']
    for title,data in (('All gold',report),('Gold >= 10',report['gold_ge_10'])):
        lines += ['',f'## {title}','','| Aggregate | Metric | Count | Estimate | CI95 |',
            '| --- | --- | ---: | ---: | --- |']
        for model,metrics in [('All models',data['metrics'])]+[(m,v['metrics']) for m,v in data['models'].items()]:
            for metric,v in metrics.items():
                lines.append(f"| {model} | {metric} | {v['count']} | {_number(v['estimate'])} | {v['ci95']} |")
        lines += ['','| Model | Experiment | Thinking | Request | Prefill | N | Metric | Count | Estimate | CI95 |',
            '| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |']
        for model,item in data['models'].items():
            for c in item['cells']:
                for metric,v in c['metrics'].items():
                    lines.append(f"| {model} | {c['experiment']} | {c['thinking']} | {c['request_type']} | {c['prefill_type']} | {c['n_responses']} | {metric} | {v.get('count','')} | {_number(v['estimate'])} | {v['ci95']} |")
        lines += ['','| Model | Outcome | Contrast | Estimate | CI95 | Holds |','| --- | --- | --- | ---: | --- | --- |']
        for model,item in data['models'].items():
            for metric,contrasts in item['contrasts'].items():
                for name,v in contrasts.items():
                    lines.append(f"| {model} | {metric} | {name} | {_number(v.get('estimate'))} | {v.get('ci95','')} | {v['holds']} |")
        lines += ['','| H25 model | Status | R | L | N | Keyword mentions | N minus R | CI95 | Undefined resamples | Holds |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |']
        for model,item in data['H25']['models'].items():
            v=item['N_minus_R']
            lines.append(f"| {model} | {item['status']} | {item.get('R',{}).get('n',0)} | {item.get('L',{}).get('n',0)} | {item.get('N',{}).get('n',0)} | {item.get('keyword_count','')} | {_number(v.get('estimate'))} | {v.get('ci95','')} | {v.get('undefined_resamples','')} | {v['holds']} |")
        lines += ['','| H25 sensitivity model | Outcome | Estimate | CI95 | Holds |',
            '| --- | --- | ---: | --- | --- |']
        for model,item in data['H25']['violation_sensitivity']['models'].items():
            v=item['N_minus_R']
            lines.append(f"| {model} | violation_sensitivity | {_number(v.get('estimate'))} | {v.get('ci95','')} | {v['holds']} |")
    if 'trace_agreement' in report:
        item=report['trace_agreement'];metrics=item['metrics']
        lines += ['','| Trace coding | Value |','| --- | --- |',
            '| Judges | gpt-6.1-sol / claude-haiku-5-5; no human labels |',
            f"| Seeded agreement sample | {item['n_labeled']}/100; seed 0 |",
            f"| Agreement | {_number(metrics['agreement'])} |",
            f"| Cohen kappa | {_number(metrics['kappa'])} |"]
    if 'execution' in report:
        item=report['execution']
        lines += ['','| Execution | Value |','| --- | --- |',
            f"| Cap USD | {item['cap_usd']} |",
            f"| Conservative spend USD | {_number(item['conservative_spend_usd'])} |",
            f"| Provider credit delta USD, provisional | {_number(item['provider_credit_delta_provisional_usd'])} |",
            f"| Zero Phase 6 instances verified | {item['zero_phase6_instances_verified']} |",
            f"| Account-wide zero Vast instances verified | {item['zero_instances_verified']} |",
            f"| Other instances listed at resource check | {json.dumps(item['other_instances_listed'])} |",
            f"| Resource check Unix time | {item['resource_check_unix']} |",
            f"| Account credit delta attributable to Phase 6 | {not item['concurrent_account_usage_observed']} |"]
    if 'keyword_continuity' in report:
        lines += ['','| Phase 5 keyword continuity | Mentions | N |','| --- | ---: | ---: |']
        for model,item in report['keyword_continuity']['counts'].items():
            lines.append(f"| {model} | {item['keyword_mentions']} | {item['n']} |")
    lines += ['','## Provenance','','| Field | Value |','| --- | --- |']
    for key,value in report['grid']['provenance'].items():
        lines.append(f"| {key} | {json.dumps(value,sort_keys=True).replace('|','&#124;')} |")
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path);parser.add_argument('--problem-manifest',required=True,type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('results/phase6'))
    parser.add_argument('--drop-stage',action='append',choices=STAGES[1:],default=[])
    parser.add_argument('--labels',type=Path);parser.add_argument('--trace-labels',type=Path)
    parser.add_argument('--historical-input',action='append',type=Path,default=[])
    parser.add_argument('--trace-report',type=Path)
    parser.add_argument('--execution-ledger',type=Path)
    parser.add_argument('--keyword-continuity',type=Path)
    args=parser.parse_args()
    try:
        rows=score_rows(read_rows(args.input));manifest=json.loads(args.problem_manifest.read_text())
        report=analyze(rows,problem_manifest=manifest,dropped_stages=args.drop_stage,
            labels=load_labels(args.labels) if args.labels else (),
            trace_labels=list(read_rows(args.trace_labels)) if args.trace_labels else (),
            historical_rows=[r for path in args.historical_input for r in read_rows(path)])
        if args.trace_report:
            coding=json.loads(args.trace_report.read_text());agreement_item=coding['agreement']
            if (coding.get('status')!='complete' or agreement_item.get('status')!='complete'
                    or agreement_item.get('n_labeled')!=100 or agreement_item['metrics']['n']!=100):
                raise ValueError('final trace report requires complete primary coding and seeded 100 agreement')
            for model,item in coding['all_gold'].items():
                actual=report['H25']['models'][model]
                if any(actual[c]['n']!=item['counts'][c] for c in ('R','L','N')):
                    raise ValueError('trace report counts disagree with source-joined labels')
            report['trace_agreement']=agreement_item
        if args.execution_ledger:
            ledger=json.loads(args.execution_ledger.read_text())
            report['execution']=execution_summary(ledger,report['grid']['retained_stages'])
        if args.keyword_continuity:
            continuity=json.loads(args.keyword_continuity.read_text())
            if (continuity.get('regex')!='tutor|system prompt|instruction' or
                    any(v.get('n')!=400 or not 0<=v['keyword_mentions']<=400 for v in continuity['counts'].values())):
                raise ValueError('keyword continuity requires the original regex and complete 400-trace cells')
            for model in ('Qwen/Qwen3-32B','google/gemma-4-12B-it'):
                if continuity['counts'][model]['keyword_mentions']!=report['H25']['models'][model]['keyword_count']:
                    raise ValueError('keyword continuity disagrees with rescored historical source')
            report['keyword_continuity']=continuity
        sample=write_audit_sample(rows,args.output_dir/'parser-audit-sample.jsonl')
        agreement(rows,load_labels(args.labels) if args.labels else (),sample=sample)
        args.output_dir.mkdir(parents=True,exist_ok=True)
        (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        (args.output_dir/'summary.md').write_text(render_markdown(report))
    except (ValueError,OSError,TypeError) as error:parser.error(str(error))
    print(f'Analyzed {len(rows)} responses -> {args.output_dir}/summary.{{json,md}}')


if __name__=='__main__':main()
