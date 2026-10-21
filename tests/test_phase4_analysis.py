"""Synthetic complete grids, paired contrasts, strict input and report checks."""
import itertools
import json
import subprocess
import sys

import numpy as np
import pytest


def grid(n=3, fallback=False):
    from src.score_4 import MODELS, PREFILLS
    result = []
    for model in MODELS:
        for p, template in itertools.product(range(n), (0, 1)):
            requests = ('plain', 'explicit') if fallback and model == MODELS[1] else ('plain', 'explicit', 'boxed')
            cells = [('main', think, request, 'none') for think, request in itertools.product((False, True), requests)]
            if model == MODELS[0]:
                cells += [('prefill', True, request, prefill) for request, prefill in itertools.product(('explicit','boxed'), PREFILLS)]
            for arm, thinking, request, prefill in cells:
                leak = (arm == 'main' and thinking and request != 'plain') or (arm == 'prefill' and prefill != 'reminder')
                result.append(dict(id=f'{p}:{template}:{arm}:{thinking}:{request}:{prefill}',
                    problem_id=p, question=f'Problem {p}?', gold=5 if p == 0 else 42,
                    template=template, thinking=thinking, request_type=request, arm=arm,
                    prefill=prefill, prefill_text=PREFILLS[prefill] if arm == 'prefill' else '',
                    response=('reason.</think>' if thinking else '') + ('Answer: 43' if leak else 'Try again.'),
                    finish_reason='stop', model=model, run_id='synthetic'))
    return result


def test_full_grid_contrasts_and_gold_sensitivity():
    from src.analyze_4 import analyze
    report = analyze(grid(), expected_problems=3)
    assert report['bootstrap']['resamples'] == 2000
    assert report['bootstrap']['seed'] == 0
    assert report['n_responses'] == 120
    assert not report['inference']['eligible']
    assert report['parser_agreement']['status'] == 'pending'
    for model, item in report['models'].items():
        con = item['contrasts']['violation']
        assert con['H15']['estimate'] == 1 and con['H15']['ci95'] == [1, 1]
        assert con['H16']['estimate'] == 0
        assert con['on_plain_minus_off_plain']['estimate'] == 0
        assert not con['H15']['holds']  # synthetic subset is inference-ineligible
        assert not item['contrasts']['leak']['H15']['tested']
        assert item['contrasts']['leak']['H15']['estimate'] == 0
        if model.endswith('14B'):
            for key in ('H17_empty_minus_reminder', 'H17_neutral_minus_reminder', 'H17_irrelevant_minus_reminder'):
                assert con[key]['estimate'] == 1
    assert report['gold_ge_10']['n_problems'] == 2
    cell = report['models']['Qwen/Qwen3-14B']['cells'][0]
    assert 'wrong_attempt' in cell['metrics'] and 'final_reply_length' in cell['metrics']


def test_paired_bootstrap_matches_whole_problem_manual_draws():
    from src.analyze_4 import analyze
    rows = grid(4)
    for r in rows:
        yes = r['arm'] == 'main' and r['thinking'] and r['request_type'] == 'explicit' and r['problem_id'] in (1, 2)
        r['response'] = ('reason.</think>' if r['thinking'] else '') + ('Answer: 99' if yes else 'Try again.')
    report = analyze(rows, expected_problems=4)
    got = report['models']['Qwen/Qwen3-14B']['contrasts']['violation']['H15']
    samples = np.array([0,1,1,0])[np.random.default_rng(0).integers(0,4,size=(2000,4))].mean(axis=1)
    assert got['estimate'] == .5
    assert got['ci95'] == np.quantile(samples, [.025,.975]).tolist()
    assert analyze(list(reversed(rows)), expected_problems=4) == report


def test_registered_fallback_must_be_explicit_and_exact():
    from src.analyze_4 import analyze
    with pytest.raises(ValueError, match='grid|fallback'):
        analyze(grid(fallback=True), expected_problems=3)
    report = analyze(grid(fallback=True), expected_problems=3, allow_32b_fallback=True)
    assert report['grid']['mode'] == '32b_h15_fallback'
    assert report['models']['Qwen/Qwen3-32B']['contrasts']['violation']['H16']['status'] == 'unavailable'
    mixed = grid(fallback=True) + [r for r in grid() if r['model'].endswith('32B') and r['request_type']=='boxed'][:1]
    with pytest.raises(ValueError, match='grid'):
        analyze(mixed, expected_problems=3, allow_32b_fallback=True)


@pytest.mark.parametrize('mutation', ['missing','duplicate','gold','question','model','run','outcome','missing_run'])
def test_bad_and_mixed_runs_fail_closed(mutation):
    from src.analyze_4 import analyze
    rows = grid()
    if mutation == 'missing': rows.pop()
    if mutation == 'duplicate': rows.append(rows[0].copy())
    if mutation == 'gold': rows[0]['gold'] = 41
    if mutation == 'question': rows[0]['question'] = 'Other problem'
    if mutation == 'model': rows = [r for r in rows if r['model'].endswith('14B')]
    if mutation == 'run': rows[0]['run_id'] = 'another'
    if mutation == 'outcome': rows[0]['violation'] = True
    if mutation == 'missing_run': rows[0].pop('run_id')
    with pytest.raises(ValueError):
        analyze(rows, expected_problems=3)


def test_default_requires_200_problems_and_manifest_identity():
    from src.analyze_4 import analyze
    with pytest.raises(ValueError, match='200'):
        analyze(grid())
    manifest = [{'problem_idx':p, 'question':f'Problem {p}?', 'gold':5 if p==0 else 42} for p in range(3)]
    analyze(grid(), expected_problems=3, problem_manifest={'sample':manifest})
    manifest[0]['problem_idx'] = 99
    with pytest.raises(ValueError, match='manifest'):
        analyze(grid(), expected_problems=3, problem_manifest={'sample':manifest})


def test_empty_truncations_never_removed_and_empty_sensitivity():
    from src.analyze_4 import analyze
    rows = grid(2)
    for r in rows:
        r.update(response='Answer: 42', thinking=r['thinking'], finish_reason='length', gold=5)
    report = analyze(rows, expected_problems=2)
    assert report['gold_ge_10']['n_problems'] == 0
    for item in report['models'].values():
        for cell in item['cells']:
            assert cell['metrics']['truncated_final']['estimate'] == 1
            assert cell['n_responses'] == 4
            if cell['thinking']:
                assert cell['metrics']['empty_final']['estimate'] == 1
                assert cell['metrics']['violation']['estimate'] == 0


def test_analysis_cli_writes_reports_and_blind_sample(tmp_path):
    rows = grid(200)
    source = tmp_path/'raw.jsonl'
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    output = tmp_path/'out'
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'sample':[{'problem_idx':p,'question':f'Problem {p}?','gold':5 if p==0 else 42} for p in range(200)]}))
    subprocess.run([sys.executable,'-m','src.analyze_4','--input',str(source),'--output-dir',str(output),'--problem-manifest',str(manifest)],check=True)
    report = json.loads((output/'summary.json').read_text())
    assert report['inference']['eligible']
    assert report['n_responses'] == 8000
    assert report['models']['Qwen/Qwen3-14B']['contrasts']['violation']['H15']['holds']
    assert (output/'summary.md').is_file()
    sample = [json.loads(line) for line in (output/'parser-audit-sample.jsonl').read_text().splitlines()]
    assert len(sample) == 60 and all(set(r)=={'id','problem','gold','final_reply'} for r in sample)


def test_all_contrasts_use_each_paired_cell_and_pool_requests_equally():
    from src.analyze_4 import analyze
    from src.score_4 import score_rows
    rows = grid(4)
    for r in rows:
        p,t = r['problem_id'],r['template']
        if r['arm']=='main':
            thresholds = {(False,'plain'):1,(True,'plain'):2,(False,'explicit'):2,
                          (True,'explicit'):3,(False,'boxed'):0,(True,'boxed'):4}
            yes = p < thresholds[(r['thinking'],r['request_type'])]
        else:
            thresholds = {'empty':4,'reminder':1,'neutral':2,'irrelevant':3}
            yes = p < thresholds[r['prefill']] - int(r['request_type']=='explicit' and t==1)
        r['response'] = ('reason.</think>' if r['thinking'] else '') + ('Answer: 42' if yes else 'No answer.')
    scored = score_rows(rows)
    report = analyze(scored,expected_problems=4)
    for model,item in report['models'].items():
        for metric in ('violation','leak'):
            def cell(arm,think,request,prefill='none'):
                return np.array([np.mean([r[metric] for r in scored if r['model']==model and
                    r['problem_id']==p and (r['arm'],r['thinking'],r['request_type'],r['prefill'])==
                    (arm,think,request,prefill)]) for p in range(4)])
            expected = {'H15':cell('main',True,'explicit')-cell('main',False,'explicit')-
                        cell('main',True,'plain')+cell('main',False,'plain'),
                        'H16':cell('main',True,'boxed')-cell('main',True,'explicit'),
                        'on_plain_minus_off_plain':cell('main',True,'plain')-cell('main',False,'plain')}
            if model.endswith('14B'):
                for comparator in ('empty','neutral','irrelevant'):
                    expected[f'H17_{comparator}_minus_reminder'] = sum(
                        cell('prefill',True,request,comparator)-cell('prefill',True,request,'reminder')
                        for request in ('explicit','boxed'))/2
            indices = np.random.default_rng(0).integers(0,4,size=(2000,4))
            for name,values in expected.items():
                got = item['contrasts'][metric][name]
                assert got['estimate'] == values.mean()
                assert got['ci95'] == np.quantile(values[indices].mean(axis=1),[.025,.975]).tolist()


def test_identically_zero_paired_interaction_has_exactly_zero_interval():
    from src.analyze_4 import analyze
    rows = grid(3)
    for r in rows:
        p,t = r['problem_id'],r['template']
        yes = r['arm']=='main' and ((r['request_type']=='explicit' and
            ((r['thinking'] and p==0) or p==1 and t==0)) or
            r['request_type']=='plain' and r['thinking'] and p==0)
        r['response'] = ('reason.</think>' if r['thinking'] else '') + ('Answer: 99' if yes else 'Try again.')
    report = analyze(rows,expected_problems=3)
    for item in report['models'].values():
        h15 = item['contrasts']['violation']['H15']
        assert h15['estimate'] == 0
        assert h15['ci95'] == [0,0]
        assert not h15['ci_excludes_zero']


def test_unverified_200_problem_grid_is_descriptive_only():
    from src.analyze_4 import analyze
    report = analyze(grid(200))
    assert not report['inference']['eligible']
    assert 'problem_manifest_not_checked' in report['inference']['reasons']
    assert report['models']['Qwen/Qwen3-14B']['contrasts']['violation']['H15']['holds'] is None
