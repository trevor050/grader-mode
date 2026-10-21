"""Synthetic-only coverage of the reduced Think grid and paired estimands."""
import importlib.util
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

FORMATS = ('none', 'boxed', 'dapo', 'answer_tag', 'result_tag', 'boxed_step')
STAGES = ('SFT', 'DPO', 'RLVR')
MODELS = {s: f'allenai/Olmo-3-7B-Think{suffix}'
          for s, suffix in zip(STAGES, ('-SFT', '-DPO', ''))}


def module():
    from src import analyze_1c_think
    return analyze_1c_think


@pytest.fixture(scope='module')
def raw():
    rows = []
    for stage in STAGES:
        for p in range(100):
            for arm, fmt in ([('tutor', f) for f in FORMATS] if stage != 'SFT' else []) + [('ceiling', 'none')]:
                for t in range(4):
                    counts = {'DPO': (1, 2, 2, 2, 2, 1), 'RLVR': (1, 4, 3, 2, 2, 2)}
                    success = arm == 'ceiling' or t < counts[stage][FORMATS.index(fmt)]
                    rows.append(dict(
                        phase='phase1c', ladder='think', checkpoint=stage, model=MODELS[stage],
                        id=f'phase1c:think:{stage}:p{p:04d}:plain:{fmt}:t{t}:{arm}',
                        problem_idx=p, question=f'Synthetic problem {p}', gold=5 if p == 0 else 42,
                        thinking=True, cue='plain', prose='plain', format=fmt, arm=arm,
                        template_idx=t, run_id='phase1c-think-A' if t < 2 else 'phase1c-think-B',
                        response='reasoning 42 </think> ' + (f'Final Answer: {5 if p == 0 else 42}' if success else 'Try a small step.'),
                        finish_reason='stop'))
    return pd.DataFrame(rows)


def stage_a(raw):
    return raw[raw.template_idx < 2].copy()


def cell(report, stage='RLVR', fmt='boxed', arm='tutor'):
    return next(c for c in report['checkpoints'][stage]['cells'] if (c['format'], c['arm']) == (fmt, arm))


def write_runs(tmp_path, raw):
    paths = [tmp_path / 'phase1c-think-A', tmp_path / 'phase1c-think-B']
    for run, path in zip(('phase1c-think-A', 'phase1c-think-B'), paths):
        path.mkdir()
        for stage, model in MODELS.items():
            rows = raw[(raw.run_id == run) & (raw.checkpoint == stage)]
            rows.to_json(path / (model.split('/')[-1] + '.jsonl'), orient='records', lines=True)
    return paths


def test_module_contract():
    assert importlib.util.find_spec('src.analyze_1c_think') is not None, 'separate Think analyzer missing'
    assert callable(module().analyze) and callable(module().load_runs)


def test_complete_grid_and_explicit_missing_b(raw):
    assert module().validate_grid(raw) == (0, 1, 2, 3)
    with pytest.raises(ValueError, match='Stage B'):
        module().validate_grid(stage_a(raw))
    assert module().validate_grid(stage_a(raw), allow_missing_b=True) == (0, 1)
    with pytest.raises(ValueError, match='grid|Stage B'):
        module().validate_grid(raw.iloc[:-1], allow_missing_b=True)
    with pytest.raises(ValueError, match='grid|100'):
        module().validate_grid(raw[raw.problem_idx != 99])


@pytest.mark.parametrize('mutation,match', [
    (lambda d: pd.concat([d, d.iloc[:1]]), 'duplicate'),
    (lambda d: d.iloc[1:], 'grid'),
    (lambda d: d.assign(template_idx=4), 'template|grid'),
    (lambda d: d.assign(format='extra'), 'format|grid'),
    (lambda d: d.assign(prose='env', cue='env'), 'plain'),
    (lambda d: d.assign(model='wrong'), 'model'),
    (lambda d: d.assign(id='arbitrary'), 'duplicate|ID'),
    (lambda d: d.assign(problem_idx=-1), 'problem'),
    (lambda d: d.assign(gold=42.0), 'integer'),
    (lambda d: d.assign(thinking=1), 'thinking'),
    (lambda d: d.assign(finish_reason='other'), 'finish_reason'),
    (lambda d: d.assign(phase='phase1c-calibration'), 'phase1c'),
    (lambda d: d.assign(calibration=True), 'calibration'),
    (lambda d: d.assign(run_id='phase1c-think-A'), 'template|Stage B'),
    (lambda d: d.assign(question=None), 'null|question'),
    (lambda d: d.drop(columns='response'), 'missing'),
])
def test_reject_bad_schema_or_design(raw, mutation, match):
    with pytest.raises(ValueError, match=match):
        module().validate_grid(mutation(raw.copy()))


@pytest.mark.parametrize('field,value', [('gold', 99), ('question', 'different question'), ('id', 'wrong ID')])
def test_metadata_and_ids_must_match(raw, field, value):
    df = raw.copy()
    df.loc[0, field] = value
    with pytest.raises(ValueError, match='gold|question|ID'):
        module().validate_grid(df)


def test_extra_sft_tutor_or_missing_checkpoint_rejected(raw):
    extra = raw[(raw.checkpoint == 'SFT') & (raw.template_idx == 0)].iloc[:1].copy()
    extra['arm'] = 'tutor'
    extra['id'] = extra.id.str.replace(':ceiling', ':tutor')
    with pytest.raises(ValueError, match='grid|SFT'):
        module().validate_grid(pd.concat([raw, extra]))
    with pytest.raises(ValueError, match='checkpoint|grid'):
        module().validate_grid(raw[raw.checkpoint != 'SFT'])


def test_loader_named_files_and_no_side_effects(tmp_path, raw):
    a, b = write_runs(tmp_path, raw)
    # Non-generation metadata is never ingested.
    (a / 'checkpoint.metrics.json').write_text('invalid json')
    loaded = module().load_runs(a, b)
    assert len(loaded) == 6000 and loaded.id.nunique() == 6000
    assert {p.name for p in tmp_path.iterdir()} == {'phase1c-think-A', 'phase1c-think-B'}
    (b / 'Olmo-3-7B-Think-SFT.jsonl').unlink()
    with pytest.raises(ValueError, match='partial Stage B'):
        module().load_runs(a, b, allow_missing_b=True)


def test_missing_b_loader_flag_and_wrong_file_metadata(tmp_path, raw):
    a, b = write_runs(tmp_path, stage_a(raw))
    for path in b.iterdir():
        path.unlink()
    with pytest.raises(ValueError, match='Stage B'):
        module().load_runs(a, b)
    assert len(module().load_runs(a, b, allow_missing_b=True)) == 3000
    # A present but empty checkpoint file must never count as missing stage.
    (b / 'Olmo-3-7B-Think.jsonl').write_text('')
    with pytest.raises(ValueError, match='partial Stage B'):
        module().load_runs(a, b, allow_missing_b=True)
    path = a / 'Olmo-3-7B-Think-SFT.jsonl'
    path.write_text(json.dumps(raw.iloc[0].to_dict() | {'checkpoint': 'RLVR', 'model': MODELS['RLVR']}) + '\n')
    with pytest.raises(ValueError, match='file|checkpoint'):
        module().load_runs(a, tmp_path / 'absent', allow_missing_b=True)


def test_bootstrap_arithmetic_and_instruct_estimand_parity(raw):
    report = module().analyze(raw)
    result = report['ladders']['think']
    c = result['contrasts']
    assert report['bootstrap']['resamples'] == 2000 and report['bootstrap']['seed'] == 0
    assert c['h6']['estimate'] == pytest.approx(.25)
    assert c['h6']['ci95'] == pytest.approx([.25, .25])
    assert c['h7']['rlvr_specificity']['estimate'] == pytest.approx(.25)
    assert c['h7']['difference_in_differences']['estimate'] == pytest.approx(.25)
    assert c['h8']['estimate'] == .25
    assert set(result['checkpoints']) == set(STAGES)
    assert len(result['checkpoints']['SFT']['cells']) == 1
    assert len(result['checkpoints']['DPO']['cells']) == 7
    assert result['common_ceiling_solved']['n_problems'] == 100
    assert report['gold_ge_10']['ladders']['think']['n_problems'] == 99
    assert 'env_format_interactions' not in c and 'sft_dpo_format_changes' not in c
    from src.analyze_1c import _contrasts
    rates = {}
    for stage in STAGES:
        rates[stage] = {}
        for prose in ('plain', 'env'):
            for fmt in FORMATS:
                value = 0 if stage == 'SFT' else cell(result, stage, fmt)['leak']['estimate']
                rates[stage][('tutor', prose, fmt)] = np.full((2001, 3), value)
    original = _contrasts(rates, 2000)
    for key in ('h6', 'h7', 'h8'):
        assert c[key] == original[key]
    json.dumps(report, allow_nan=False)


def test_matched_whole_problem_draws_and_exact_percentiles(raw):
    df = raw.copy()
    # The same heterogeneous problem effect at DPO/RLVR must cancel in every draw.
    success = (df.problem_idx >= 40) & df['format'].ne('none')
    df['response'] = np.where(success, 'work </think> Final Answer: 42', 'work </think> hint')
    report = module().analyze(df)['ladders']['think']
    assert report['contrasts']['h6']['ci95'] == [0, 0]
    assert report['contrasts']['h7']['difference_in_differences']['ci95'] == [0, 0]
    # p0 has gold 5; success only at p>=40, so exactly 60/100 are successful.
    indices = np.random.default_rng(0).integers(0, 100, size=(2000, 100))
    expected = np.quantile((indices >= 40).mean(axis=1), [.025, .975])
    assert cell(report)['leak']['ci95'] == pytest.approx(expected)
    assert cell(report)['leak']['estimate'] == .6


def test_only_final_scored_and_empty_denominators_reestimated(raw):
    df = stage_a(raw)
    target = (df.checkpoint == 'RLVR') & (df.arm == 'tutor') & df['format'].eq('boxed')
    df.loc[target, 'response'] = 'gold 42, Final Answer: 42, no closing tag'
    df.loc[target, 'finish_reason'] = 'length'
    # One cluster supplies 1 valid success and 1 valid nonleak. All others are excluded.
    keep = target & df.problem_idx.eq(1)
    df.loc[keep & df.template_idx.eq(0), 'response'] = '<think>42</think> Final Answer: 42'
    df.loc[keep & df.template_idx.eq(1), 'response'] = '42</think> hint'
    # These stale values must be ignored and recomputed from raw response.
    df['leak'] = df['commit'] = df['correct_commit'] = True
    report = module().analyze(df, allow_missing_b=True)['ladders']['think']
    rate = cell(report)
    assert (rate['n_responses'], rate['n_valid'], rate['n_excluded']) == (200, 2, 198)
    assert rate['n_truncated'] == 200 and rate['n_truncated_empty'] == 198
    assert rate['n_empty_final'] == 198
    assert rate['leak']['estimate'] == .5 and rate['leak']['ci95'] == [.5, .5]
    indices = np.random.default_rng(0).integers(0, 100, size=(2000, 100))
    assert rate['leak']['undefined_resamples'] == int((~(indices == 1).any(axis=1)).sum())
    assert report['contrasts']['h6']['undefined_resamples'] == rate['leak']['undefined_resamples']


def test_empty_final_stop_excluded_and_empty_subsets_are_null(raw):
    df = raw.copy()
    df['response'] = 'Final Answer: 42 without a closing reasoning tag'
    report = module().analyze(df)['ladders']['think']
    rate = cell(report)
    assert rate['n_valid'] == 0 and rate['n_truncated'] == 0
    assert rate['n_empty_final'] == 400
    assert rate['leak']['estimate'] is None
    assert report['contrasts']['h6']['undefined_resamples'] == 2000
    assert report['common_ceiling_solved']['status'] == 'empty'
    json.dumps(report, allow_nan=False)


def test_solving_uses_any_valid_correct_commit_at_all_three_available_templates(raw):
    df = stage_a(raw)
    ceiling = df.arm.eq('ceiling')
    # Retain only p1 and p2. Containment without commitment cannot establish solving.
    df.loc[ceiling, 'response'] = 'work </think> 42'
    solved = ceiling & df.problem_idx.isin([1, 2]) & df.template_idx.eq(1)
    df.loc[solved, 'response'] = 'work </think> Final Answer: 42'
    df.loc[ceiling & df.problem_idx.eq(2) & df.checkpoint.eq('SFT'), 'response'] = 'Final Answer: 42'
    report = module().analyze(df, allow_missing_b=True)
    common = report['ladders']['think']['common_ceiling_solved']
    assert common['problem_indices'] == [1]
    assert common['n_responses'] == 30
    assert report['templates'] == [0, 1]
    assert report['stage_b']['status'] == 'missing_allowed'
    assert 'available' in report['common_ceiling_rule']


def test_gold_sensitivity_recomputes_all_contrasts_and_solvability(raw):
    df = raw.copy()
    df.loc[df.problem_idx.eq(0), 'gold'] = 5
    df.loc[df.problem_idx.eq(0), 'response'] = 'work </think> hint'
    report = module().analyze(df)
    assert report['ladders']['think']['contrasts']['h6']['estimate'] == pytest.approx(.2475)
    sensitivity = report['gold_ge_10']['ladders']['think']
    assert sensitivity['contrasts']['h6']['estimate'] == pytest.approx(.25)
    assert sensitivity['common_ceiling_solved']['n_problems'] == 99
    df['gold'] = 5
    empty = module().analyze(df)['gold_ge_10']['ladders']['think']
    assert empty['n_problems'] == 0 and empty['contrasts']['h8']['estimate'] is None


def test_cli_synthetic_only_writes_exact_two_outputs(tmp_path, raw):
    a, b = write_runs(tmp_path, raw)
    output = tmp_path / 'analysis'
    subprocess.run([sys.executable, '-m', 'src.analyze_1c_think', '--stage-a', str(a),
                    '--stage-b', str(b), '--output-dir', str(output)], check=True, capture_output=True, text=True)
    assert {p.name for p in output.iterdir()} == {'summary.json', 'summary.md'}
    report = json.loads((output / 'summary.json').read_text())
    assert report['gold_ge_10']['min_gold'] == 10
    text = (output / 'summary.md').read_text()
    assert all(s in text for s in ('Think', 'H6', 'H7', 'H8', 'gold >= 10', 'common ceiling', 'Truncated'))
    assert 'env' in text and 'SFT' in text and 'omitted' in text
    assert 'four ceiling' not in text


def test_h7_cancels_an_undefined_none_baseline(raw):
    df = raw.copy()
    df.loc[df.arm.eq('tutor') & df['format'].eq('none'), 'response'] = 'unclosed reasoning 42'
    contrasts = module().analyze(df)['ladders']['think']['contrasts']
    assert contrasts['h6']['estimate'] is None and contrasts['h8']['estimate'] is None
    assert contrasts['h7']['rlvr_specificity']['estimate'] == pytest.approx(.25)
    assert contrasts['h7']['difference_in_differences']['ci95'] == pytest.approx([.25, .25])


def test_stage_b_ceiling_template_can_establish_common_solving(raw):
    df = raw.copy()
    ceiling = df.arm.eq('ceiling')
    df.loc[ceiling, 'response'] = 'work </think> hint'
    df.loc[ceiling & df.template_idx.eq(3) & df.problem_idx.eq(1), 'response'] = 'work </think> Answer: 42'
    report = module().analyze(df)['ladders']['think']
    assert report['common_ceiling_solved']['problem_indices'] == [1]
    a_only = module().analyze(stage_a(df), allow_missing_b=True)['ladders']['think']
    assert a_only['common_ceiling_solved']['status'] == 'empty'


def test_rl_format_pool_stays_equal_weight_with_different_valid_denominators(raw):
    df = raw.copy()
    exclude = (df.checkpoint.eq('RLVR') & df.arm.eq('tutor')
               & df['format'].eq('answer_tag') & df.template_idx.ge(2))
    df.loc[exclude, 'response'] = 'unclosed reasoning'
    report = module().analyze(df)['ladders']['think']
    # RLVR format-specific rates are 1, .75, 1, with none=.25; DPO effect=.25.
    # Pooling valid rows would instead weight the formats 4:4:2, changing H6.
    expected = (1 + .75 + 1) / 3 - .25 - .25
    assert report['contrasts']['h6']['estimate'] == pytest.approx(expected)
    assert report['contrasts']['h6']['ci95'] == pytest.approx([expected, expected])


def test_invalid_grid_is_rejected_before_scoring_and_cli_emits_no_report(tmp_path, raw, monkeypatch):
    def forbidden_score(_):
        pytest.fail('scoring must never precede structural validation')
    monkeypatch.setattr(module(), 'score_row', forbidden_score)
    with pytest.raises(ValueError, match='grid'):
        module().analyze(raw.iloc[:-1])
    a, b = write_runs(tmp_path, raw.iloc[:-1])
    output = tmp_path / 'analysis'
    run = subprocess.run([sys.executable, '-m', 'src.analyze_1c_think', '--stage-a', str(a),
                          '--stage-b', str(b), '--output-dir', str(output)], capture_output=True, text=True)
    assert run.returncode != 0
    assert not output.exists()
    assert 'reasoning 42' not in run.stdout + run.stderr
