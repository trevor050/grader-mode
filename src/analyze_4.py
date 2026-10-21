"""Phase 4 preregistered cell rates and paired problem-cluster bootstrap.

Only a complete two-model grid is accepted: 200 shared problems, two templates,
6 main cells/model plus 8 prefill cells at 14B (8,000 rows). Explicitly allowing
the registered 32B H15 fallback permits exactly its four plain/explicit cells
(7,200 rows total), never a partially filled boxed cell. Every response stays
in the denominator. Optional problem-manifest sample checks bind identities,
questions and gold to a previously frozen Phase 1 sample; without it grid
validation is structural and does not certify dataset provenance.

Each cell averages its two templates within each problem. All cells, outcomes
and models share seed-0 draws of whole problems; percentile 95% CIs use 2,000
resamples. H17 equally pools explicit/boxed and tests empty-minus-reminder,
neutral-minus-reminder, and irrelevant-minus-reminder; these latter two are
the differences in lowering from the common empty baseline. Positive lower
bounds support the directional hypotheses. Leak repeats are descriptive.
Final length is Unicode characters after whitespace trimming, not tokens.

The user explicitly authorized analysis before the hand audit. Summaries
report parser agreement pending until lead labels are passed via --labels.
No interpretation prose, external API requests, or paid compute are used.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.score import read_rows
from src.score_4 import METRICS, MODELS, PREFILLS, REQUESTS, score_rows
from src.phase4_parser_agreement import agreement, load_labels, write_audit_sample


MAIN = [('main', think, request, 'none') for think, request in itertools.product((False, True), REQUESTS)]
PREFILL = [('prefill', True, request, prefill) for request, prefill in itertools.product(('explicit','boxed'), PREFILLS)]


def _cell(row):
    return (row['arm'], row['thinking'], row['request_type'], row['prefill'])


def _sort_problem(pid):
    return (type(pid).__name__, pid)


def validate_grid(rows, *, expected_problems=200, allow_32b_fallback=False, problem_manifest=None):
    if type(expected_problems) is not int or expected_problems < 1:
        raise ValueError('expected_problems must be a positive integer')
    if {r['model'] for r in rows} != set(MODELS):
        raise ValueError('complete grid requires both registered models')
    # Missing provenance is tolerated only when absent from every row.
    for field in ('run_id', 'phase'):
        values = {json.dumps(r.get(field), sort_keys=True) for r in rows}
        if len(values) != 1:
            raise ValueError(f'mixed or missing {field} within run')
    problems, identities = set(), {}
    actual = defaultdict(set)
    for r in rows:
        p = r['problem_id']
        problems.add(p)
        signature = (r['gold'], r['question'])
        if p in identities and identities[p] != signature:
            raise ValueError(f'inconsistent gold or question for problem {p}')
        identities[p] = signature
        key = (r['model'], p)
        observation = (_cell(r), r['template'])
        if observation in actual[key]:
            raise ValueError(f'duplicate factor tuple in grid: {key} {observation}')
        actual[key].add(observation)
    if len(problems) != expected_problems:
        raise ValueError(f'grid requires exactly {expected_problems} problems; got {len(problems)}')
    full32 = set(itertools.product(MAIN, (0,1)))
    fallback32 = {c for c in full32 if c[0][2] != 'boxed'}
    sets32 = [actual[(MODELS[1],p)] for p in problems]
    if all(c == full32 for c in sets32):
        mode = 'full'
    elif allow_32b_fallback and all(c == fallback32 for c in sets32):
        mode = '32b_h15_fallback'
    else:
        raise ValueError('incomplete or mixed 32B grid; registered fallback requires explicit allow_32b_fallback')
    expected14 = set(itertools.product(MAIN+PREFILL, (0,1)))
    if any(actual[(MODELS[0],p)] != expected14 for p in problems):
        raise ValueError('incomplete 14B main/prefill grid')
    if problem_manifest is not None:
        sample = problem_manifest.get('sample') if isinstance(problem_manifest, dict) else problem_manifest
        if not isinstance(sample, list):
            raise ValueError('problem manifest requires sample list')
        expected = {}
        for p in sample:
            pid = p.get('problem_id', p.get('problem_idx'))
            question = p.get('question', p.get('problem'))
            if type(pid) not in (int,str) or not isinstance(question,str) or type(p.get('gold')) is not int or pid in expected:
                raise ValueError('malformed or duplicate problem manifest sample')
            expected[pid] = (p['gold'],question)
        if identities != expected:
            raise ValueError('grid identities/gold/questions do not match problem manifest')
    return {'mode':mode, 'n_problems':len(problems), 'expected_problems':expected_problems,
            'templates':[0,1], 'n_responses':len(rows),
            'problem_manifest_checked':problem_manifest is not None}


def _interval(point, samples, *, tested=False, eligible=False):
    lo, hi = (float(x) for x in np.quantile(samples,[.025,.975]))
    return {'estimate':float(point), 'ci95':[lo,hi], 'valid_resamples':len(samples),
            'undefined_resamples':0, 'tested':tested,
            'holds':bool(lo > 0) if tested and eligible else None,
            'ci_excludes_zero':bool(lo > 0 or hi < 0)}


def _summarize(rows, *, eligible):
    problems = sorted({r['problem_id'] for r in rows}, key=_sort_problem)
    if not problems:
        return {'n_problems':0,'n_responses':0,'models':{}}
    indices = np.random.default_rng(0).integers(0,len(problems),size=(2000,len(problems)))
    result = {'n_problems':len(problems), 'n_responses':len(rows), 'models':{}}
    for model in MODELS:
        selected = [r for r in rows if r['model'] == model]
        cells = sorted({_cell(r) for r in selected})
        by_key = defaultdict(list)
        for r in selected:
            by_key[(r['problem_id'],_cell(r))].append(r)
        values = np.array([[[np.mean([r[m] for r in by_key[(p,c)]]) for m in METRICS]
                           for c in cells] for p in problems], dtype=float)
        point = values.mean(axis=0)
        # Limit peak memory; samples remain paired across every cell/metric.
        boot = np.concatenate([values[batch].mean(axis=1) for batch in np.array_split(indices,16)])
        cell_results = []
        for j,c in enumerate(cells):
            metrics = {}
            for k,m in enumerate(METRICS):
                metric = _interval(point[j,k],boot[:,j,k])
                if m != 'final_reply_length':
                    metric['count'] = sum(r[m] for r in selected if _cell(r)==c)
                else:
                    metric['unit'] = 'characters'
                metrics[m] = metric
            cell_results.append({'arm':c[0], 'thinking':c[1], 'request_type':c[2], 'prefill':c[3],
                                 'n_responses':len(problems)*2, 'metrics':metrics})
        contrasts = {}
        for metric in ('violation','leak'):
            k = METRICS.index(metric)
            def linear(terms, tested):
                # Contrast within each problem before averaging. Apart from
                # preserving pairing, this prevents rounding cell means from
                # turning an identically zero interaction into a tiny effect.
                paired = sum(w*values[:,cells.index(c),k] for c,w in terms)
                draws = paired[indices].mean(axis=1)
                return _interval(paired.mean(),draws,
                                 tested=tested and metric=='violation', eligible=eligible)
            on_plain=('main',True,'plain','none'); off_plain=('main',False,'plain','none')
            on_exp=('main',True,'explicit','none'); off_exp=('main',False,'explicit','none')
            out = {'H15':linear([(on_exp,1),(off_exp,-1),(on_plain,-1),(off_plain,1)],True),
                   'on_plain_minus_off_plain':linear([(on_plain,1),(off_plain,-1)],False)}
            boxed=('main',True,'boxed','none')
            out['H16'] = (linear([(boxed,1),(on_exp,-1)],True) if boxed in cells else
                          {'status':'unavailable','reason':'registered_32b_h15_fallback',
                           'estimate':None,'ci95':[None,None],'tested':False,'holds':None})
            if model == MODELS[0]:
                for comparator in ('empty','neutral','irrelevant'):
                    terms = [(('prefill',True,r,comparator),.5) for r in ('explicit','boxed')]
                    terms += [(('prefill',True,r,'reminder'),-.5) for r in ('explicit','boxed')]
                    out[f'H17_{comparator}_minus_reminder'] = linear(terms,True)
                h17 = [out[f'H17_{c}_minus_reminder']['holds'] for c in ('empty','neutral','irrelevant')]
                out['H17_joint'] = {'holds':all(h17) if eligible and metric=='violation' else None,
                                    'tested':metric=='violation', 'rule':'all_three_directional_lower_bounds_above_zero'}
            contrasts[metric] = out
        result['models'][model] = {'n_problems':len(problems),'n_responses':len(selected),
                                   'cells':cell_results,'contrasts':contrasts}
    return result


def analyze(rows, *, expected_problems=200, allow_32b_fallback=False, problem_manifest=None):
    if hasattr(rows,'to_dict'):
        rows = rows.to_dict('records')
    rows = score_rows(rows)
    grid = validate_grid(rows, expected_problems=expected_problems,
                         allow_32b_fallback=allow_32b_fallback, problem_manifest=problem_manifest)
    eligible = expected_problems == 200 and grid['problem_manifest_checked']
    report = _summarize(rows, eligible=eligible)
    report.update(phase='phase4',grid=grid,
                  bootstrap={'resamples':2000,'seed':0,'unit':'problem_id','interval':'percentile 95%',
                             'paired_across':'templates,cells,arms,metrics,models'},
                  inference={'eligible':eligible,'reasons':([] if eligible else ([
                        'synthetic_or_smoke_problem_count'] if expected_problems != 200 else []) + (
                        ['problem_manifest_not_checked'] if not grid['problem_manifest_checked'] else [])),
                             'parser_audit_required_by_preregistration':True,
                             'analysis_before_hand_audit_authorized':True},
                  methods={'denominator':'all_requests_including_empty_finals',
                           'final_reply_length_unit':'characters',
                           'correct_attempt':'any_detected_value_equals_gold',
                           'wrong_attempt':'attempt_and_not_correct_attempt',
                           'attempt_window':'numeric_token_start_offset_0_through_12_after_marker',
                           'multiplicity_adjustment':None},
                  parser_agreement={'status':'pending','n_sampled':60,'n_labeled':0,'agreement':None})
    report['gold_ge_10'] = _summarize([r for r in rows if r['gold'] >= 10],eligible=eligible)
    return report


def _number(value):
    return 'NA' if value is None else f'{value:.6g}'


def render_markdown(report):
    """Tables and factual status metadata only; no scientific interpretation."""
    lines = ['# Phase 4', '', f"Grid: `{report['grid']['mode']}`; {report['n_problems']} problems; {report['n_responses']} responses.",
             '', 'Bootstrap: 2,000 paired problem resamples; seed 0; percentile 95% intervals.',
             '', 'Denominator: all requests, including empty finals. Length: characters.',
             '', f"Parser agreement: `{report['parser_agreement']['status']}`; {report['parser_agreement']['n_labeled']}/60 labeled."]
    if report['parser_agreement'].get('agreement') is not None:
        lines += [f"Agreement: {_number(report['parser_agreement']['agreement'])}; Cohen kappa: {_number(report['parser_agreement']['cohen_kappa'])}."]
    for title,data in (('All gold',report),('Gold >= 10',report['gold_ge_10'])):
        lines += ['',f'## {title}', '', f"Problems: {data['n_problems']}; responses: {data['n_responses']}."]
        for model,item in data['models'].items():
            lines += ['',f'### {model}', '', '| Arm | Thinking | Request | Prefill | N | Metric | Count | Estimate | CI 95% |',
                      '| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |']
            for cell in item['cells']:
                for metric,value in cell['metrics'].items():
                    lines.append(f"| {cell['arm']} | {cell['thinking']} | {cell['request_type']} | {cell['prefill']} | {cell['n_responses']} | {metric} | {value.get('count','')} | {_number(value['estimate'])} | [{', '.join(_number(x) for x in value['ci95'])}] |")
            lines += ['', '| Outcome | Contrast | Estimate | CI 95% | Tested | Holds |',
                      '| --- | --- | ---: | --- | --- | --- |']
            for metric,contrasts in item['contrasts'].items():
                for name,value in contrasts.items():
                    lines.append(f"| {metric} | {name} | {_number(value.get('estimate'))} | [{', '.join(_number(x) for x in value.get('ci95',[None,None]))}] | {value['tested']} | {value['holds']} |")
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('results/phase4'))
    parser.add_argument('--allow-32b-fallback',action='store_true')
    parser.add_argument('--problem-manifest',type=Path)
    parser.add_argument('--labels',type=Path,help='Lead attempt labels, JSONL or CSV; absent means pending')
    args = parser.parse_args()
    try:
        rows = score_rows(read_rows(args.input))
        manifest = json.loads(args.problem_manifest.read_text()) if args.problem_manifest else None
        report = analyze(rows,allow_32b_fallback=args.allow_32b_fallback,problem_manifest=manifest)
        sample = write_audit_sample(rows,args.output_dir/'parser-audit-sample.jsonl')
        report['parser_agreement'] = agreement(rows,load_labels(args.labels) if args.labels else (),sample=sample)
        report['parser_agreement']['sample_path'] = str(args.output_dir/'parser-audit-sample.jsonl')
        args.output_dir.mkdir(parents=True,exist_ok=True)
        (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        (args.output_dir/'summary.md').write_text(render_markdown(report))
    except (ValueError,OSError,TypeError) as error:
        parser.error(str(error))
    print(f"Analyzed {len(rows)} responses -> {args.output_dir}/summary.{{json,md}}")


if __name__ == '__main__':
    main()
