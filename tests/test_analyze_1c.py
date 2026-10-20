"""Phase 1c paired risk contrasts, structural validation, and CLI contract."""
import importlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

FORMATS = ('none', 'boxed', 'dapo', 'answer_tag', 'result_tag', 'boxed_step')
STAGES = ('SFT', 'DPO', 'RLVR')
MODELS = {
    ladder: {stage: 'allenai/Olmo-3-7B-' + ladder.title() + suffix
             for stage, suffix in zip(STAGES, ('-SFT', '-DPO', ''))}
    for ladder in ('instruct', 'think')
}


def module():
    from src import analyze_1c
    return analyze_1c


def grid(n=4, ladder='instruct', stages=STAGES):
    rows = []
    for stage in stages:
        for p in range(n):
            for arm, prose, fmt in [('tutor', c, f) for c in ('plain', 'env') for f in FORMATS] + [('ceiling', 'plain', 'none')]:
                for t in range(4):
                    counts = {
                        'SFT': (0, 0, 0, 0, 0, 0),
                        'DPO': (1, 2, 2, 2, 2, 1),
                        'RLVR': (1, 4, 3, 2, 2, 2),
                    }
                    count = 4 if arm == 'ceiling' else counts[stage][FORMATS.index(fmt)]
                    if prose == 'env' and arm == 'tutor':
                        count = min(4, count + (fmt == 'boxed'))
                    success = t < count
                    rows.append(dict(phase='phase1c', model=MODELS[ladder][stage], ladder=ladder,
                        checkpoint=stage, problem_idx=p, gold=5 if p == 0 else 42,
                        id=f'{p}-{arm}-{prose}-{fmt}-{t}', cue=prose, prose=prose,
                        format=fmt, arm=arm, template_idx=t, thinking=ladder == 'think',
                        response='raw', reasoning='work' if ladder == 'think' else '',
                        final_reply='hint', truncated=False, excluded_from_rates=False,
                        leak=success, commit=success, correct_commit=success))
    return pd.DataFrame(rows)


def cell(result, stage, prose='plain', fmt='boxed', arm='tutor'):
    return next(c for c in result['checkpoints'][stage]['cells']
                if (c['prose'], c['format'], c['arm']) == (prose, fmt, arm))


def test_module_exposes_analysis_contract():
    spec = importlib.util.find_spec('src.analyze_1c')
    assert spec is not None, 'Phase 1c analysis module is missing'
    assert callable(module().analyze) and callable(module().validate_grid)


def test_known_paired_hypotheses_and_exploratory_effects():
    report = module().analyze(grid(), resamples=60)
    result = report['ladders']['instruct']
    c = result['contrasts']
    assert report['phase'] == 'phase1c'
    assert report['bootstrap']['unit'] == 'problem_idx'
    assert len(result['checkpoints']['RLVR']['cells']) == 13
    assert cell(result, 'RLVR')['leak']['estimate'] == 1
    assert c['plain_format_effects']['DPO']['boxed']['estimate'] == .25
    assert c['plain_format_effects']['RLVR']['pooled_rl']['estimate'] == .5
    assert c['h6']['estimate'] == .25 and c['h6']['ci95'] == [.25, .25]
    assert c['h7']['rlvr_specificity']['estimate'] == .25
    assert c['h7']['difference_in_differences']['estimate'] == .25
    assert c['h8']['estimate'] == .25
    assert c['per_format_h6']['boxed']['estimate'] == .5
    assert c['per_format_h6']['dapo']['estimate'] == .25
    assert c['per_format_h6']['answer_tag']['estimate'] == 0
    assert c['env_format_interactions']['DPO']['boxed']['estimate'] == .25
    assert c['sft_dpo_format_changes']['pooled_rl']['estimate'] == .25
    assert result['common_ceiling_solved']['n_problems'] == 4
    assert result['common_ceiling_solved']['contrasts'] == c
    assert report['gold_ge_10']['ladders']['instruct']['n_problems'] == 3
    assert not result['inference']['eligible']
    json.dumps(report, allow_nan=False)


def test_common_ceiling_requires_any_valid_correct_commit_at_every_stage():
    df = grid()
    df.loc[(df.arm == 'ceiling') & (df.checkpoint == 'SFT') & (df.problem_idx == 0), 'correct_commit'] = False
    df.loc[(df.arm == 'ceiling') & (df.checkpoint == 'DPO') & (df.problem_idx == 1), 'correct_commit'] = False
    # One correct commitment is enough, not four; containment alone is insufficient.
    selected = (df.arm == 'ceiling') & (df.checkpoint == 'RLVR') & (df.problem_idx == 2) & (df.template_idx != 3)
    df.loc[selected, 'correct_commit'] = False
    df.loc[(df.problem_idx < 2) & (df.arm == 'tutor'), ['leak', 'commit', 'correct_commit']] = False
    result = module().analyze(df, resamples=80)['ladders']['instruct']
    common = result['common_ceiling_solved']
    assert common['problem_indices'] == [2, 3]
    assert common['contrasts']['h6']['estimate'] == .25
    assert result['contrasts']['h6']['estimate'] == .125
    assert set(common['contrasts']) == set(result['contrasts'])


def test_bootstrap_preserves_checkpoint_and_cell_pairing():
    df = grid(5)
    # Identical problem-varying levels in every checkpoint: every stage contrast is zero.
    success = df.problem_idx.ge(2) & df['format'].ne('none')
    df['leak'] = df['commit'] = df['correct_commit'] = success
    result = module().analyze(df, resamples=2000, seed=11)['ladders']['instruct']
    assert result['contrasts']['h6']['ci95'] == [0, 0]
    assert result['contrasts']['h7']['difference_in_differences']['ci95'] == [0, 0]
    assert cell(result, 'RLVR')['leak']['ci95'] == pytest.approx([.2, 1])


def test_empty_thinking_final_exclusions_recompute_ratio_denominators():
    df = grid(ladder='think')
    selected = (df.checkpoint == 'RLVR') & (df.prose == 'plain') & (df['format'] == 'boxed') & (df.arm == 'tutor')
    # p0: all excluded; other problems retain all successes. Keep invalid outcomes
    # true to prove exclusions also mask numerators, not just denominators.
    empty = selected & df.problem_idx.eq(0)
    df.loc[empty, ['truncated', 'excluded_from_rates']] = True
    df.loc[empty, 'final_reply'] = ''
    report = module().analyze(df, resamples=1000, seed=0)['ladders']['think']
    rate = cell(report, 'RLVR')
    assert rate['n_valid'] == 12 and rate['n_excluded'] == 4
    assert rate['n_truncated_empty'] == 4 and rate['leak']['estimate'] == 1
    assert rate['leak']['ci95'] == [1, 1]
    assert rate['leak']['undefined_resamples'] > 0
    df.loc[selected, 'excluded_from_rates'] = True
    df.loc[selected, 'final_reply'] = ''
    result = module().analyze(df, resamples=30)['ladders']['think']
    assert result['contrasts']['h6']['estimate'] is None
    assert result['contrasts']['h6']['undefined_resamples'] == 30
    json.dumps(result, allow_nan=False)


def test_partial_checkpoints_are_annotated_without_fabricated_contrasts():
    report = module().analyze(grid(stages=('RLVR',)), resamples=30)
    result = report['ladders']['instruct']
    assert result['checkpoint_status'] == 'partial'
    assert result['available_checkpoints'] == ['RLVR']
    assert result['missing_checkpoints'] == ['SFT', 'DPO']
    assert result['contrasts']['h6']['estimate'] is None
    assert result['contrasts']['h6']['status'] == 'unavailable'
    assert result['contrasts']['h8']['estimate'] == .25
    assert result['contrasts']['h7']['rlvr_specificity']['estimate'] == .25
    assert result['contrasts']['h7']['difference_in_differences']['estimate'] is None
    assert result['common_ceiling_solved']['status'] == 'unavailable'
    assert result['common_ceiling_solved']['contrasts'] == {}


def test_ladders_can_have_different_sample_sizes_and_empty_sensitivity():
    df = pd.concat([grid(4), grid(2, ladder='think')], ignore_index=True)
    result = module().analyze(df, resamples=20)
    assert result['ladders']['instruct']['n_problems'] == 4
    assert result['ladders']['think']['n_problems'] == 2
    df['gold'] = 5
    result = module().analyze(df, resamples=20)
    assert result['gold_ge_10']['n_responses'] == 0
    assert result['gold_ge_10']['ladders']['think']['contrasts']['h6']['estimate'] is None
    assert result['gold_ge_10']['ladders']['think']['common_ceiling_solved']['n_problems'] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('mutation,match', [
    (lambda d: pd.concat([d, d.iloc[:1]]), 'duplicate'),
    (lambda d: d.iloc[1:], 'grid'),
    (lambda d: d.assign(phase='phase1c_calibration'), 'phase1c'),
    (lambda d: d.assign(is_calibration=True), 'calibration'),
    (lambda d: d.assign(cue='env'), 'cue.*prose'),
    (lambda d: d.assign(thinking=True), 'thinking'),
    (lambda d: d.assign(model='wrong/model'), 'model'),
    (lambda d: d.assign(problem_idx=-1), 'problem'),
    (lambda d: d.assign(template_idx=True), 'integer'),
    (lambda d: d.assign(gold=42.0), 'integer'),
    (lambda d: d.assign(leak=2), 'binary'),
    (lambda d: d.assign(commit=False), 'correct_commit'),
    (lambda d: d.assign(excluded_from_rates=True), 'excluded'),
    (lambda d: d.assign(final_reply=None), 'null'),
    (lambda d: d.assign(ladder='other'), 'ladder'),
    (lambda d: d.assign(checkpoint='OTHER'), 'checkpoint'),
    (lambda d: d.assign(format='unknown'), 'grid'),
    (lambda d: d.drop(columns='prose'), 'missing'),
])
def test_validation_rejects_invalid_identity_or_design(mutation, match):
    with pytest.raises(ValueError, match=match):
        module().validate_grid(mutation(grid()))


def test_validation_rejects_unpaired_problem_sets_or_gold():
    df = grid()
    with pytest.raises(ValueError, match='problem sets'):
        module().validate_grid(df[~((df.checkpoint == 'SFT') & df.problem_idx.eq(0))])
    df.loc[df.checkpoint == 'SFT', 'gold'] = 99
    with pytest.raises(ValueError, match='gold'):
        module().validate_grid(df)


def test_empty_common_subset_is_undefined_not_zero():
    df = grid()
    df.loc[df.arm == 'ceiling', 'correct_commit'] = False
    result = module().analyze(df, resamples=40)['ladders']['instruct']['common_ceiling_solved']
    assert result['status'] == 'empty'
    assert result['n_problems'] == 0
    assert result['contrasts']['h6']['estimate'] is None
    assert result['contrasts']['h6']['ci95'] == [None, None]


def test_cli_writes_primary_and_sensitivity_number_reports(tmp_path):
    source = tmp_path / 'scored.jsonl'
    grid().to_json(source, orient='records', lines=True)
    target = tmp_path / 'analysis'
    subprocess.run([sys.executable, '-m', 'src.analyze_1c', '--input', str(source),
                    '--output-dir', str(target), '--resamples', '30'], check=True)
    assert len(list(target.glob('handcheck_*.csv'))) == 3
    assert {p.name for p in target.glob('summary*')} == {
        'summary.json', 'summary.md', 'summary_gold_ge_10.json', 'summary_gold_ge_10.md'}
    report = json.loads((target / 'summary.json').read_text())
    sensitivity = json.loads((target / 'summary_gold_ge_10.json').read_text())
    assert report['n_responses'] == 624
    assert sensitivity['min_gold'] == 10 and sensitivity['n_responses'] == 468
    markdown = (target / 'summary.md').read_text()
    assert all(s in markdown for s in ('H6', 'H7', 'H8', 'common ceiling', 'SFT', 'DPO', 'RLVR', 'Correct commit'))
    assert '95%' in markdown and 'partial' not in markdown
    assert 'holds' not in markdown and 'significant' not in markdown


@pytest.mark.parametrize('resamples', [0, -1, True, 1.5])
def test_invalid_resample_counts(resamples):
    with pytest.raises(ValueError, match='resamples'):
        module().analyze(grid(), resamples=resamples)


def test_h7_cancels_the_shared_none_baseline_even_when_it_is_all_excluded():
    df = grid(ladder='think')
    empty = (df.arm == 'tutor') & df.prose.eq('plain') & df['format'].eq('none')
    df.loc[empty, 'excluded_from_rates'] = True
    df.loc[empty, 'final_reply'] = ''
    result = module().analyze(df, resamples=100)['ladders']['think']['contrasts']
    assert result['h6']['estimate'] is None
    # (RL - none) - (result_tag - none) simplifies to RL - result_tag;
    # an unavailable baseline must not destroy an otherwise estimable contrast.
    assert result['h7']['rlvr_specificity']['estimate'] == .25
    assert result['h7']['difference_in_differences']['ci95'] == [.25, .25]


def test_bootstrap_matches_direct_cluster_ratio_oracle_with_uneven_exclusions():
    df = grid(3, ladder='think')
    selected = df.arm.eq('tutor') & df.prose.eq('plain') & df['format'].eq('boxed')
    # Valid denominators 1, 2, 4 with success counts 1, 0, 4.
    empty = selected & (((df.problem_idx == 0) & (df.template_idx > 0))
                        | ((df.problem_idx == 1) & (df.template_idx > 1)))
    df.loc[empty, 'excluded_from_rates'] = True
    df.loc[empty, 'final_reply'] = ''
    success = selected & df.problem_idx.ne(1)
    df.loc[selected, ['leak', 'commit', 'correct_commit']] = np.repeat(
        success[selected].to_numpy()[:, None], 3, axis=1)
    result = module().analyze(df, resamples=500, seed=17)['ladders']['think']
    draws = np.random.default_rng(17).integers(0, 3, size=(500, 3))
    oracle = np.array([1, 0, 4])[draws].sum(axis=1) / np.array([1, 2, 4])[draws].sum(axis=1)
    value = cell(result, 'RLVR')['leak']
    assert value['estimate'] == pytest.approx(5 / 7)
    assert value['ci95'] == pytest.approx(np.quantile(oracle, [.025, .975]))
    # The identical format rates across stages still cancel under common draws.
    assert result['contrasts']['per_format_h6']['boxed']['ci95'] == [0, 0]


def test_rl_format_pool_uses_equal_format_weights_after_exclusions():
    df = grid(ladder='think')
    selected = df.arm.eq('tutor') & df.checkpoint.eq('RLVR') & df.prose.eq('plain') & df['format'].eq('boxed')
    empty = selected & df.template_idx.ne(0)
    df.loc[empty, 'excluded_from_rates'] = True
    df.loc[empty, 'final_reply'] = ''
    result = module().analyze(df, resamples=30)['ladders']['think']['contrasts']
    assert result['plain_format_effects']['RLVR']['pooled_rl']['estimate'] == .5
    assert result['h6']['estimate'] == .25


def test_excluded_ceiling_success_does_not_qualify_problem():
    df = grid(ladder='think')
    empty = df.arm.eq('ceiling') & df.problem_idx.eq(0) & df.checkpoint.eq('SFT')
    df.loc[empty, 'excluded_from_rates'] = True
    df.loc[empty, 'final_reply'] = ''
    result = module().analyze(df, resamples=30)['ladders']['think']['common_ceiling_solved']
    assert result['problem_indices'] == [1, 2, 3]


def test_real_scored_rows_flow_into_analysis_without_reasoning_leaks():
    from src.score import score_row
    df = grid(2, ladder='think')
    rows = []
    for row in df.to_dict('records'):
        answer = rf'\boxed{{{row["gold"]}}}' if row['leak'] else 'Try another step.'
        raw = {k: v for k, v in row.items() if k not in (
            'reasoning', 'final_reply', 'truncated', 'excluded_from_rates',
            'leak', 'commit', 'correct_commit')}
        raw.update(response=rf'<think>\boxed{{{row["gold"]}}}</think>' + answer, finish_reason='stop')
        rows.append(score_row(raw))
    result = module().analyze(pd.DataFrame(rows), resamples=40)['ladders']['think']
    assert cell(result, 'SFT')['leak']['estimate'] == 0
    assert result['contrasts']['h6']['estimate'] == .25
    assert result['common_ceiling_solved']['n_problems'] == 2


def test_instruct_only_marks_think_deferred_without_hypotheses():
    report=module().analyze(grid(),resamples=10)
    assert report['deferred_ladders']==['think']
    assert set(report['ladders'])=={'instruct'}
    assert report['ladders']['instruct']['contrasts']['h6']['status']=='ok'
    assert report['gold_ge_10']['deferred_ladders']==['think']
    assert 'Think ladder: deferred' in module().render_summary(report)
