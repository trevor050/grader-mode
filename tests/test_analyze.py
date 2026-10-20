import csv
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.analyze import analyze, bootstrap_estimate, fit_logistic, handcheck, validate_grid


def grid(n=8, model='Qwen/Qwen3-0.6B', variant='constant'):
    rows = []
    rng = np.random.default_rng(191)
    for p in range(n):
        for cue in ('none', 'human', 'auto'):
            for fmt in ('none', 'boxed', 'final'):
                for t in range(4):
                    leak = (cue == 'auto' and fmt != 'none') if variant == 'constant' else (rng.random() < .15 + .12 * (cue == 'auto') + .12 * (fmt != 'none') + .15 * (p % 3))
                    rows.append(dict(model=model, id=f'{p}-{cue}-{fmt}-{t}', problem_idx=p,
                                     cue=cue, format=fmt, template_idx=t, arm='tutor', gold=42,
                                     response='42' if leak else 'hint', leak=leak, commit=leak, correct_commit=leak))
        for t in range(4):
            rows.append(dict(model=model, id=f'{p}-ceiling-{t}', problem_idx=p, cue='none',
                             format='none', template_idx=t, arm='ceiling', gold=42,
                             response='Final Answer:42', leak=True, commit=True, correct_commit=True))
    return pd.DataFrame(rows)


def test_validation_rejects_duplicates_and_unbalanced_grids():
    df = grid()
    validate_grid(df)
    with pytest.raises(ValueError, match='duplicate'):
        validate_grid(pd.concat([df, df.iloc[:1]]))
    with pytest.raises(ValueError, match='grid'):
        validate_grid(df.iloc[1:])
    with pytest.raises(ValueError, match='problem'):
        validate_grid(pd.concat([df, grid(7, 'other')]))


def test_cluster_bootstrap_resamples_whole_problems_and_known_contrast():
    # Four observations within each problem are identical: row bootstrap is wrong.
    values = np.array([[0., 0.], [0., 0.], [1., 1.], [1., 1.]])
    result = bootstrap_estimate(values, lambda means: means[0] - means[1], resamples=2000, seed=0)
    assert result['estimate'] == 0
    assert result['ci95'] == [0, 0]
    mean = bootstrap_estimate(values, lambda means: means[0], resamples=2000, seed=0)
    assert mean['ci95'] == [0, 1]
    assert mean == bootstrap_estimate(values, lambda means: means[0], resamples=2000, seed=0)


def test_known_risk_contrasts_ratio_and_smoke_gate():
    result = analyze(grid(), resamples=100, seed=0)
    model = result['models']['Qwen/Qwen3-0.6B']
    assert len(model['cells']) == 9
    assert model['h1']['boxed']['estimate'] == 1
    assert model['h1']['final']['ci95'] == [1, 1]
    assert model['h1']['pooled_demand']['estimate'] == 1
    assert model['h2']['none']['estimate'] == 0
    assert model['h2']['boxed']['estimate'] == 1
    assert model['h2']['pooled']['estimate'] == pytest.approx(2 / 3)
    assert model['leak_ceiling_ratio']['estimate'] == pytest.approx(2 / 9)
    assert model['ceiling']['primary_correct_commit']['estimate'] == 1
    assert not result['inference']['eligible']
    assert not result['phase2_gate']['eligible']
    assert model['logistic']['status'] == 'separated'


def test_zero_denominator_is_null_and_counted():
    df = grid()
    df.loc[df.arm == 'ceiling', ['leak', 'commit', 'correct_commit']] = False
    result = analyze(df, resamples=50)['models'][df.model[0]]
    assert result['leak_ceiling_ratio']['estimate'] is None
    assert result['leak_ceiling_ratio']['ci95'] == [None, None]
    assert result['leak_ceiling_ratio']['zero_denominator_resamples'] == 50


def test_logistic_cluster_se_and_interactions():
    df = grid(30, variant='mixed')
    fit = fit_logistic(df[df.arm == 'tutor'])
    assert fit['status'] == 'ok'
    assert fit['covariance'] == 'cluster(problem_idx)'
    interactions = [row for row in fit['coefficients'] if ':' in row['term']]
    assert len(interactions) == 4
    assert all(len(row['ci95']) == 2 for row in interactions)
    df['leak'] = False
    assert fit_logistic(df[df.arm == 'tutor'])['status'] == 'separated'


def test_handcheck_blank_then_existing_real_labels(tmp_path):
    df = grid()
    first = handcheck(df, tmp_path, seed=0)
    assert first[df.model[0]]['n_labeled'] == 0
    path = next(tmp_path.glob('handcheck_*.csv'))
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 100
    assert all(row['manual_leak'] == '' for row in rows)
    rows[0]['manual_leak'] = rows[0]['leak']
    rows[1]['manual_leak'] = str(1 - int(rows[1]['leak']))
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    second = handcheck(df, tmp_path, seed=0)
    assert second[df.model[0]]['n_labeled'] == 2
    assert second[df.model[0]]['leak_agreement'] == .5
    assert second[df.model[0]]['commit_agreement'] is None


def test_h3_complete_models_but_smoke_ineligible():
    df = pd.concat([grid(model=f'Qwen/Qwen3-{size}B') for size in (.6, 1.7, 4, 8, 14, 32)])
    result = analyze(df, resamples=50)
    assert result['h3']['status'] == 'descriptive_only'
    assert result['h3']['pooled']['estimate'] == pytest.approx(0, abs=1e-14)
    assert not result['inference']['eligible']


def test_analysis_cli_writes_json_and_handcheck(tmp_path):
    source = tmp_path / 'scored.jsonl'
    grid().to_json(source, orient='records', lines=True)
    target = tmp_path / 'analysis'
    subprocess.run([sys.executable, '-m', 'src.analyze', '--input', str(source), '--output-dir', str(target), '--resamples', '30'], check=True)
    assert json.loads((target / 'summary.json').read_text())['bootstrap']['seed'] == 0
    assert len(list(target.glob('handcheck_*.csv'))) == 1
    markdown = (target / 'summary.md').read_text()
    assert 'H1' in markdown and 'H2' in markdown and '95%' in markdown
    assert 'separated' in markdown and 'manual' in markdown


def test_paired_ceiling_ratio_preserves_cluster_correlation():
    df = grid(2)
    for metric in ('leak', 'commit', 'correct_commit'):
        df[metric] = df.problem_idx.astype(bool)
    ratio = analyze(df, resamples=200)['models'][df.model[0]]['leak_ceiling_ratio']
    assert ratio['estimate'] == 1
    assert ratio['ci95'] == [1, 1]
    assert ratio['zero_denominator_resamples'] > 0
    assert ratio['valid_resamples'] + ratio['zero_denominator_resamples'] == 200


def test_full_ladder_gate_and_known_positive_h3_slope():
    sizes = (.6, 1.7, 4, 8, 14, 32)
    frames = []
    for k, size in enumerate(sizes):
        df = grid(200, model=f'Qwen/Qwen3-{size}B')
        selected = (df.arm == 'tutor') & (df.cue == 'auto') & (df['format'] != 'none') & (df.problem_idx < k * 40)
        for metric in ('leak', 'commit', 'correct_commit'):
            df.loc[df.arm == 'tutor', metric] = selected[df.arm == 'tutor']
        frames.append(df)
    result = analyze(pd.concat(frames), resamples=80)
    expected = np.polyfit(np.log(sizes), np.arange(6) * .2 * 2 / 3, 1)[0]
    assert result['inference']['eligible']
    assert result['phase2_gate']['passed']
    assert result['h3']['status'] == 'ok'
    assert result['h3']['pooled']['estimate'] == pytest.approx(expected)
    assert result['h3']['pooled']['ci95'][0] > 0


def test_quasi_separation_in_one_cell_is_not_reported_as_finite_fit():
    df = grid(30, variant='mixed')
    df.loc[(df.cue == 'auto') & (df['format'] == 'boxed'), 'leak'] = True
    fit = fit_logistic(df[df.arm == 'tutor'])
    assert fit['status'] == 'separated'
    assert fit['separated_cells'] == ['auto/boxed']


def test_fit_exception_is_reported_without_coefficients(monkeypatch):
    import src.analyze as module
    df = grid(30, variant='mixed')
    def broken_fit(*args, **kwargs):
        raise RuntimeError('synthetic fitting failure')
    monkeypatch.setattr(module.smf, 'glm', broken_fit)
    fit = fit_logistic(df[df.arm == 'tutor'])
    assert fit['status'] == 'failed'
    assert fit['coefficients'] == []
    assert 'synthetic fitting failure' in fit['reason']


def test_handcheck_refuses_stale_labels(tmp_path):
    df = grid()
    handcheck(df, tmp_path)
    df['response'] = 'changed response'
    with pytest.raises(ValueError, match='changed'):
        handcheck(df, tmp_path)
