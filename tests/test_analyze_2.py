"""Synthetic Phase 2 grids; no generations or activation data are needed."""
import json

import numpy as np
import pandas as pd
import pytest

from src.analyze import bootstrap_estimate
from src.analyze_2 import analyze, render_summary


HELDOUT = ('boxed_alt_1', 'boxed_alt_2', 'final')
CELLS = (
    [('boxed', 'baseline'), ('boxed', 'ablate_d')]
    + [('boxed', f'ablate_r{i}') for i in range(5)]
    + [('none', 'baseline'), ('none', 'add_d')]
    + [('none', f'add_r{i}') for i in range(5)]
    + [(fmt, arm) for fmt in HELDOUT for arm in ('baseline', 'ablate_d')]
)
MODEL = 'Qwen/Qwen3-8B'


def grid(templates=2):
    rows = []
    for rank in range(101, 201):
        for fmt, arm in CELLS:
            for template in range(templates):
                # Random directions have deliberately unequal effects. The
                # expected superiority is their mean, not any one direction.
                if fmt == 'boxed':
                    leak = arm == 'baseline' or (
                        arm.startswith('ablate_r') and int(arm[-1]) < 3)
                elif fmt == 'none':
                    leak = arm == 'add_d' or (
                        arm.startswith('add_r') and int(arm[-1]) < 2)
                else:
                    leak = arm == 'baseline' and rank - 100 <= 20 * (HELDOUT.index(fmt) + 1)
                rows.append(dict(
                    model=MODEL, id=f'{rank}-{fmt}-{arm}-{template}',
                    problem_idx=1000 + 3 * rank, selection_rank=rank, split='test',
                    format=fmt, template_idx=template, intervention=arm,
                    gold=5 if rank <= 120 else 42, leak=leak, commit=leak,
                    correct_commit=leak, empty=False, repetition=False,
                    low_number_count=False, degenerate=False,
                ))
    return pd.DataFrame(rows)


def cell(result, fmt, arm):
    return next(c for c in result['cells'] if (c['format'], c['intervention']) == (fmt, arm))


def test_known_signed_contrasts_and_complete_rates_without_response_text():
    df = grid(4)
    report = analyze(df, resamples=31, seed=9)
    result = report['models'][MODEL]
    assert report['inference']['eligible']
    assert report['bootstrap']['unit'] == 'problem_idx'
    assert report['expected_templates'] == 4
    assert result['n_problems'] == 100
    assert result['n_responses'] == 8000
    assert len(result['cells']) == 20
    assert result['h9']['necessity']['estimate'] == 1
    assert result['h9']['superiority']['estimate'] == pytest.approx(.6)
    assert result['h10']['sufficiency']['estimate'] == 1
    assert result['h10']['superiority']['estimate'] == pytest.approx(.6)
    for fmt, expected in zip(HELDOUT, (.2, .4, .6)):
        assert result['h11'][fmt]['estimate'] == pytest.approx(expected)
    assert result['h11']['pooled']['estimate'] == pytest.approx(.4)
    baseline = cell(result, 'boxed', 'baseline')
    for metric in ('leak', 'commit', 'correct_commit'):
        assert baseline[metric]['estimate'] == 1
        assert baseline[metric]['ci95'] == [1, 1]
        assert baseline[metric]['n_positive'] == 400
        assert baseline[metric]['n_responses'] == 400
        assert baseline[metric]['valid_resamples'] == 31
    assert len([c for c in result['cells'] if c['steered']]) == 15
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_paired_bootstrap_keeps_problem_and_template_dependence():
    df = grid()
    shared = df.selection_rank.le(150)
    # All arms and templates share the same problem-level outcome. A row
    # bootstrap, or subtracting independently bootstrapped means, has width.
    df[['leak', 'commit', 'correct_commit']] = np.repeat(shared.to_numpy()[:, None], 3, axis=1)
    result = analyze(df, resamples=2000, seed=7, expected_templates=2)['models'][MODEL]
    for hypothesis in ('h9', 'h10', 'h11'):
        for contrast in result[hypothesis].values():
            assert contrast['estimate'] == 0
            assert contrast['ci95'] == [0, 0]
    rate = cell(result, 'boxed', 'baseline')['leak']
    reference = bootstrap_estimate(np.r_[np.ones(50), np.zeros(50)][:, None],
                                   lambda x: x[0], resamples=2000, seed=7)
    assert rate['ci95'] == reference['ci95']
    assert rate['ci95'][0] < .5 < rate['ci95'][1]


def test_nonconstant_pooled_and_random_contrast_intervals_match_cluster_reference():
    df = grid()
    mask = df['format'].eq('boxed') & df.intervention.eq('ablate_d')
    df.loc[mask, ['leak', 'commit', 'correct_commit']] = np.repeat(
        df.loc[mask].selection_rank.le(140).to_numpy()[:, None], 3, axis=1)
    result = analyze(df, resamples=211, seed=5, expected_templates=2)['models'][MODEL]
    reference = bootstrap_estimate(
        (.6 - np.r_[np.ones(40), np.zeros(60)])[:, None], lambda x: x[0],
        resamples=211, seed=5)
    actual = result['h9']['superiority']
    assert actual['estimate'] == pytest.approx(reference['estimate'])
    assert actual['ci95'] == pytest.approx(reference['ci95'])
    effects = np.array([[rank <= limit for limit in (20, 40, 60)]
                        for rank in range(1, 101)], dtype=float)
    reference = bootstrap_estimate(effects, lambda x: x.mean(), resamples=211, seed=5)
    assert result['h11']['pooled']['ci95'] == pytest.approx(reference['ci95'])


def test_templates_are_averaged_before_resampling_and_metrics_stay_distinct():
    df = grid()
    df['leak'] = df.template_idx.eq(0)
    df['commit'] = True
    df['correct_commit'] = False
    result = analyze(df, resamples=17, expected_templates=2)['models'][MODEL]
    c = cell(result, 'none', 'add_d')
    assert c['leak']['ci95'] == [.5, .5]
    assert c['commit']['ci95'] == [1, 1]
    assert c['correct_commit']['ci95'] == [0, 0]


def test_development_rows_do_not_change_estimates_or_drive_selection():
    df = grid()
    dev = df.iloc[:3].copy()
    dev['selection_rank'] = 1
    dev['problem_idx'] = 19
    dev['split'] = 'dev'
    dev['id'] = ['dev-0', 'dev-1', 'dev-2']
    dev[['leak', 'commit', 'correct_commit']] = True
    kwargs = dict(resamples=13, seed=8, expected_templates=2)
    first = analyze(df, **kwargs)
    combined = analyze(pd.concat([df, dev]), **kwargs)
    assert first['models'] == combined['models']
    assert first['gold_ge_10'] == combined['gold_ge_10']
    assert combined['n_dev_responses_excluded'] == 3
    assert combined['selection_performed'] is False
    assert analyze(df.sample(frac=1, random_state=29).to_dict('records'), **kwargs) == first


def test_degeneration_checks_all_steered_cells_strict_threshold_and_dependency_flags():
    df = grid()
    # Baseline degeneration does not trigger the registered steering flag.
    baseline = df.intervention.eq('baseline')
    df.loc[baseline, ['empty', 'degenerate']] = True
    at_threshold = df['format'].eq('boxed') & df.intervention.eq('ablate_d') & df.selection_rank.le(110)
    df.loc[at_threshold, ['repetition', 'degenerate']] = True
    over_threshold = df['format'].eq('boxed') & df.intervention.eq('ablate_r4') & df.selection_rank.le(111)
    df.loc[over_threshold, ['low_number_count', 'degenerate']] = True
    heldout = df['format'].eq('boxed_alt_2') & df.intervention.eq('ablate_d') & df.selection_rank.le(111)
    df.loc[heldout, ['empty', 'degenerate']] = True
    result = analyze(df, resamples=19, expected_templates=2)['models'][MODEL]
    assert not cell(result, 'boxed', 'ablate_d')['not_interpretable']
    random = cell(result, 'boxed', 'ablate_r4')
    assert random['not_interpretable'] is True
    assert random['degeneration']['degenerate']['estimate'] == .11
    assert random['degeneration']['low_number_count']['n_positive'] == 22
    assert not result['h9']['necessity']['not_interpretable']
    assert result['h9']['superiority']['not_interpretable']
    assert not result['h10']['superiority']['not_interpretable']
    assert result['h11']['boxed_alt_2']['not_interpretable']
    assert result['h11']['pooled']['not_interpretable']
    assert not result['h11']['final']['not_interpretable']
    for c in result['cells']:
        if c['steered']:
            assert set(c['degeneration']) == {'empty', 'repetition', 'low_number_count', 'degenerate'}
            assert all(len(v['ci95']) == 2 for v in c['degeneration'].values())


def test_gold_sensitivity_recomputes_counts_effects_and_empty_subset():
    df = grid()
    report = analyze(df, resamples=23, expected_templates=2)
    subset = report['gold_ge_10']['models'][MODEL]
    assert subset['n_problems'] == 80
    assert subset['n_responses'] == 3200
    assert subset['h11']['boxed_alt_1']['ci95'] == [0, 0]
    assert subset['h11']['boxed_alt_2']['estimate'] == .25
    assert subset['h11']['final']['estimate'] == .5
    df['gold'] = 5
    report = analyze(df, resamples=3, expected_templates=2)
    subset = report['gold_ge_10']['models'][MODEL]
    assert subset['n_problems'] == subset['n_responses'] == 0
    assert subset['h9']['necessity']['estimate'] is None
    assert subset['h9']['necessity']['ci95'] == [None, None]
    assert subset['h9']['necessity']['undefined_resamples'] == 3
    assert subset['h9']['necessity']['not_interpretable'] is None
    json.dumps(report, allow_nan=False)
    assert 'undefined' in render_summary(report)


def test_gold_boundary_and_sensitivity_degeneration_use_subset_denominator():
    df = grid()
    df.loc[df.gold.eq(42), 'gold'] = 10
    steered = df['format'].eq('none') & df.intervention.eq('add_d')
    # Nine of 100 problems is below the cutoff, but nine of the 80 retained
    # problems is above it. All gold == 10 rows must remain in sensitivity.
    df.loc[steered & df.selection_rank.between(121, 129), ['empty', 'degenerate']] = True
    report = analyze(df, resamples=7, expected_templates=2)
    assert not report['models'][MODEL]['h10']['sufficiency']['not_interpretable']
    sensitivity = report['gold_ge_10']['models'][MODEL]
    assert sensitivity['n_problems'] == 80
    assert sensitivity['h10']['sufficiency']['not_interpretable']
    c = cell(sensitivity, 'none', 'add_d')
    assert c['degeneration']['degenerate']['n_positive'] == 18
    assert c['degeneration']['degenerate']['n_responses'] == 160


def test_multiple_models_are_analyzed_separately_with_default_resamples():
    df = grid(4)
    other = df.assign(model='other', leak=False, commit=False, correct_commit=False)
    report = analyze(pd.concat([df, other]))
    assert report['bootstrap']['resamples'] == 2000
    assert report['models'][MODEL]['h9']['necessity']['ci95'] == [1, 1]
    assert report['models']['other']['h9']['necessity']['ci95'] == [0, 0]


def test_summary_contains_counts_intervals_checks_and_sensitivity_only():
    report = analyze(grid(), resamples=7, expected_templates=2)
    text = render_summary(report)
    for token in ('H9', 'H10', 'H11', '95%', '200', '100', 'gold >= 10',
                  'correct_commit', 'low_number_count', 'not_interpretable', 'boxed_alt_1'):
        assert token in text
    assert not any(word in text.lower() for word in ('supports', 'significant', 'causally', 'proves'))


@pytest.mark.parametrize('mutation, message', [
    (lambda d: d.drop(columns='leak'), 'missing'),
    (lambda d: d.iloc[0:0], 'empty'),
    (lambda d: pd.concat([d, d.iloc[:1]]), 'duplicate'),
    (lambda d: d.assign(id='same'), 'duplicate'),
    (lambda d: d.iloc[1:], 'grid'),
    (lambda d: d[d.selection_rank != 101], '100|rank'),
    (lambda d: d.assign(split='dev'), 'split|rank'),
    (lambda d: d.assign(selection_rank=d.selection_rank - 1), 'split|rank'),
    (lambda d: d.assign(split='unknown'), 'split'),
    (lambda d: d.assign(selection_rank=True), 'selection_rank'),
    (lambda d: d.assign(problem_idx=4), 'problem|rank'),
    (lambda d: d.assign(problem_idx=-1), 'problem'),
    (lambda d: d.assign(template_idx=9), 'template|grid'),
    (lambda d: d.assign(intervention='ablate_r5'), 'intervention|grid'),
    (lambda d: d.assign(format='heldout'), 'format|grid'),
    (lambda d: d.assign(format='none'), 'format|grid'),
    (lambda d: d.assign(gold=4.5), 'gold'),
    (lambda d: d.assign(leak='false'), 'leak'),
    (lambda d: d.assign(empty='false'), 'empty'),
    (lambda d: d.assign(empty=0), 'empty'),
    (lambda d: d.assign(degenerate=True), 'degenerate'),
    (lambda d: d.assign(leak=False, correct_commit=True), 'correct_commit'),
    (lambda d: d.assign(commit=False, correct_commit=True), 'correct_commit'),
    (lambda d: d.assign(model=' '), 'model'),
    (lambda d: d.assign(selection_rank=None), 'null'),
])
def test_rejects_malformed_inputs(mutation, message):
    with pytest.raises(ValueError, match=message):
        analyze(mutation(grid()), resamples=1, expected_templates=2)


def test_rejects_duplicate_cells_with_distinct_ids_and_inconsistent_metadata():
    df = grid()
    extra = df.iloc[:1].assign(id='different')
    with pytest.raises(ValueError, match='duplicate|grid'):
        analyze(pd.concat([df, extra]), resamples=1, expected_templates=2)
    for column, value, message in [('gold', 23, 'gold'), ('selection_rank', 102, 'problem|rank')]:
        broken = df.copy()
        broken.loc[0, column] = value
        with pytest.raises(ValueError, match=message):
            analyze(broken, resamples=1, expected_templates=2)
    other = df.assign(model='other', problem_idx=df.problem_idx + 1)
    with pytest.raises(ValueError, match='problem|rank'):
        analyze(pd.concat([df, other]), resamples=1, expected_templates=2)


@pytest.mark.parametrize('kwargs', [
    {'resamples': 0}, {'resamples': True}, {'resamples': 1.5},
    {'seed': -1}, {'seed': True}, {'expected_templates': 3}, {'expected_templates': True},
])
def test_rejects_invalid_analysis_options(kwargs):
    with pytest.raises(ValueError):
        analyze(grid(), **kwargs)


def test_two_templates_require_explicit_option_and_missing_grid_is_not_hidden_by_sensitivity():
    df = grid()
    with pytest.raises(ValueError, match='grid|template'):
        analyze(df, resamples=1)
    broken = df.drop(df.index[(df.gold < 10)][0])
    with pytest.raises(ValueError, match='grid'):
        analyze(broken, resamples=1, expected_templates=2)


def test_real_development_screen_metadata_is_excluded_before_test_cell_validation():
    df=grid()
    dev=df.iloc[:2].copy()
    dev['selection_rank']=1;dev['problem_idx']=19;dev['split']='dev'
    dev['intervention']='screen_L7';dev['id']=['screen-0','screen-1']
    kwargs=dict(resamples=3,expected_templates=2)
    report=analyze(pd.concat([df,dev]),**kwargs)
    assert report['models']==analyze(df,**kwargs)['models']
    assert report['n_dev_responses_excluded']==2
