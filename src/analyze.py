"""Problem-clustered Phase 1 analysis. See docs/analysis-methods.md."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import warnings
from itertools import product
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.score import read_rows

CUES = ('none', 'human', 'auto')
FORMATS = ('none', 'boxed', 'final')
METRICS = ('leak', 'commit', 'correct_commit')
CELLS = tuple(product(CUES, FORMATS))
LADDER = (.6, 1.7, 4., 8., 14., 32.)
FORMULA = "leak ~ C(cue, Treatment(reference='none')) * C(format, Treatment(reference='none'))"


def validate_grid(df: pd.DataFrame) -> None:
    """Reject partial, repeated, or incomparable problem/template/arm grids."""
    required = {'model', 'id', 'problem_idx', 'cue', 'format', 'template_idx', 'arm',
                'gold', 'response', *METRICS}
    if required - set(df.columns):
        raise ValueError(f'missing scored fields: {sorted(required - set(df.columns))}')
    if df.empty or df[list(required)].isna().any().any():
        raise ValueError('empty input or null scored fields')
    if df.duplicated(['model', 'id']).any():
        raise ValueError('duplicate IDs per model')
    for name in ('problem_idx', 'template_idx', 'gold'):
        if not df[name].map(lambda x: isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))).all():
            raise ValueError(f'{name} must contain integers')
    if (df.problem_idx < 0).any():
        raise ValueError('problem_idx must retain a nonnegative original dataset index')
    for name in ('model', 'id', 'response'):
        if not df[name].map(lambda x: isinstance(x, str)).all():
            raise ValueError(f'{name} must contain strings')
    if (df.model.str.len() == 0).any() or (df.id.str.len() == 0).any():
        raise ValueError('model and id must be nonempty')
    for metric in METRICS:
        if not df[metric].isin([0, 1, False, True]).all():
            raise ValueError(f'{metric} must be binary')
    if ((df.correct_commit > df.commit) | (df.correct_commit > df.leak)).any():
        raise ValueError('correct_commit requires both leak and commit')
    expected = {('tutor', cue, fmt, t) for cue, fmt, t in product(CUES, FORMATS, range(4))}
    expected |= {('ceiling', 'none', 'none', t) for t in range(4)}
    reference_problems = None
    for model, model_df in df.groupby('model', sort=True):
        problems = set(model_df.problem_idx)
        if reference_problems is None:
            reference_problems = problems
        elif reference_problems != problems:
            raise ValueError('models have unbalanced problem sets')
        for p, rows in model_df.groupby('problem_idx'):
            actual = list(rows[['arm', 'cue', 'format', 'template_idx']].itertuples(index=False, name=None))
            if len(actual) != len(expected) or set(actual) != expected:
                raise ValueError(f'incomplete/unbalanced grid for model={model}, problem_idx={p}; expected 40 unique rows')
    if (df.groupby('problem_idx').gold.nunique() != 1).any():
        raise ValueError('inconsistent gold for a problem across templates/arms/models')


def _cluster_means(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    # Batching keeps default 2,000 x 200 x all outcomes below a large allocation.
    return np.concatenate([values[batch].mean(axis=1) for batch in np.array_split(indices, max(1, (len(indices) + 127) // 128))])


def _interval(estimate: float, samples: np.ndarray) -> dict:
    finite = samples[np.isfinite(samples)]
    return {
        'estimate': float(estimate) if np.isfinite(estimate) else None,
        'ci95': [float(x) for x in np.quantile(finite, [.025, .975])] if len(finite) else [None, None],
        'valid_resamples': int(len(finite)),
        'undefined_resamples': int(len(samples) - len(finite)),
    }


def bootstrap_estimate(values: np.ndarray, statistic: Callable, *, resamples: int = 2000, seed: int = 0) -> dict:
    """Percentile CI from paired whole-problem means (rows are clusters)."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not len(values) or resamples < 1 or not np.isfinite(values).all():
        raise ValueError('finite nonempty 2D problem values and positive resamples required')
    indices = np.random.default_rng(seed).integers(0, len(values), size=(resamples, len(values)))
    samples = np.array([statistic(x) for x in _cluster_means(values, indices)], dtype=float)
    return _interval(statistic(values.mean(axis=0)), samples)


def fit_logistic(tutor: pd.DataFrame) -> dict:
    """Saturated binomial GLM; do not regularize separated cells into inference."""
    info = {'formula': FORMULA, 'covariance': 'cluster(problem_idx)', 'coefficients': []}
    # Nine cells / nine parameters: any all-zero/all-one cell has an infinite
    # cell logit (complete or quasi separation), even if IRLS reports convergence.
    groups = tutor.groupby(['cue', 'format'], observed=True).leak.agg(['min', 'max'])
    separated = [f'{cue}/{fmt}' for (cue, fmt), row in groups.iterrows() if row['min'] == row['max']]
    if separated:
        return info | {'status': 'separated', 'reason': 'No finite unpenalized MLE: pure-outcome cells.', 'separated_cells': separated}
    if tutor.problem_idx.nunique() < 2:
        return info | {'status': 'failed', 'reason': 'Cluster covariance requires at least two problems.'}
    try:
        data = tutor.copy()
        data['leak'] = data.leak.astype(int)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = smf.glm(FORMULA, data=data, family=sm.families.Binomial()).fit(
                cov_type='cluster', cov_kwds={'groups': data.problem_idx, 'use_correction': True})
        warning_text = [str(w.message) for w in caught]
        if not result.converged or not np.isfinite(result.params).all() or not np.isfinite(result.conf_int()).all().all():
            return info | {'status': 'failed', 'reason': 'Nonconvergence or nonfinite parameters/covariance.', 'warnings': warning_text}
        intervals = result.conf_int()
        coefficients = [dict(term=term, estimate=float(value), standard_error=float(result.bse[term]),
                             ci95=[float(x) for x in intervals.loc[term]], p_value=float(result.pvalues[term]))
                        for term, value in result.params.items()]
        return info | {'status': 'ok', 'coefficients': coefficients, 'n_clusters': int(data.problem_idx.nunique()),
                       'n_responses': len(data), 'warnings': warning_text}
    except Exception as error:
        return info | {'status': 'failed', 'reason': f'{type(error).__name__}: {error}'}


def _model_size(model: str) -> float | None:
    match = re.search(r'Qwen3-([0-9]*\.?[0-9]+)B(?:$|[^\w])', model, re.IGNORECASE)
    return float(match.group(1)) if match else None


def analyze(df: pd.DataFrame, *, resamples: int = 2000, seed: int = 0) -> dict:
    validate_grid(df)
    if resamples < 1:
        raise ValueError('resamples must be positive')
    problems = sorted(df.problem_idx.unique())
    models = sorted(df.model.unique())
    sizes = [_model_size(model) for model in models]
    complete_ladder = len(sizes) == len(LADDER) and set(sizes) == set(LADDER)
    reasons = []
    if len(problems) < 200:
        reasons.append('Fewer than 200 problems: smoke runs are ineligible for inference.')
    if not complete_ladder:
        reasons.append('Incomplete six-model Qwen3 ladder: ineligible for inference.')
    eligible = not reasons
    indices = np.random.default_rng(seed).integers(0, len(problems), size=(resamples, len(problems)))
    report = {'bootstrap': {'resamples': resamples, 'seed': seed, 'unit': 'problem_idx', 'interval': 'percentile 95%'},
              'inference': {'eligible': eligible, 'reasons': reasons, 'n_problems': len(problems), 'complete_ladder': complete_ladder},
              'models': {}}
    h2_samples, h2_points = {}, {}
    for model in models:
        rows = df[df.model == model]
        tutor = rows[rows.arm == 'tutor']
        ceiling = rows[rows.arm == 'ceiling']
        arrays = []
        for cue, fmt in CELLS:
            arrays.append(tutor[(tutor.cue == cue) & (tutor['format'] == fmt)].groupby('problem_idx')[list(METRICS)].mean().reindex(problems).to_numpy())
        arrays.append(ceiling.groupby('problem_idx')[['correct_commit', 'leak']].mean().reindex(problems).to_numpy())
        values = np.concatenate(arrays, axis=1)
        point, boot = values.mean(axis=0), _cluster_means(values, indices)
        def estimate(fn):
            return _interval(fn(point), np.asarray(fn(boot.T)))
        def column(cue, fmt, metric='leak'):
            return CELLS.index((cue, fmt)) * 3 + METRICS.index(metric)
        def ratio(numerator, denominator):
            def fn(x):
                num, den = numerator(x), x[denominator]
                return np.divide(num, den, out=np.full(np.shape(num), np.nan), where=den != 0)
            result = estimate(fn)
            result['zero_denominator_resamples'] = int(np.sum(boot[:, denominator] == 0))
            result['zero_denominator_point'] = bool(point[denominator] == 0)
            return result
        cells = []
        for cue, fmt in CELLS:
            cell = {'cue': cue, 'format': fmt, 'n_responses': len(problems) * 4}
            for metric in METRICS:
                i = column(cue, fmt, metric)
                cell[metric] = estimate(lambda x, i=i: x[i])
            cell['leak_ceiling_ratio'] = ratio(lambda x, i=column(cue, fmt): x[i], 27)
            cell['leak_containment_ceiling_ratio_sensitivity'] = ratio(lambda x, i=column(cue, fmt): x[i], 28)
            cells.append(cell)
        h1, h2 = {}, {}
        for fmt in ('boxed', 'final'):
            a, b, c, d = column('auto', fmt), column('none', fmt), column('auto', 'none'), column('none', 'none')
            h1[fmt] = estimate(lambda x, a=a, b=b, c=c, d=d: x[a] - x[b] - x[c] + x[d])
        h1['pooled_demand'] = estimate(lambda x: sum(x[column('auto', f)] - x[column('none', f)] for f in ('boxed', 'final')) / 2 - x[column('auto', 'none')] + x[column('none', 'none')])
        for fmt in FORMATS:
            a, b = column('auto', fmt), column('human', fmt)
            h2[fmt] = estimate(lambda x, a=a, b=b: x[a] - x[b])
        pooled_h2 = lambda x: sum(x[column('auto', f)] - x[column('human', f)] for f in FORMATS) / 3
        h2['pooled'] = estimate(pooled_h2)
        h2_samples[model] = {fmt: boot[:, column('auto', fmt)] - boot[:, column('human', fmt)] for fmt in FORMATS}
        h2_samples[model]['pooled'] = pooled_h2(boot.T)
        h2_points[model] = {key: value['estimate'] for key, value in h2.items()}
        pooled_leak = lambda x: sum(x[column(c, f)] for c, f in CELLS) / 9
        report['models'][model] = {
            'n_problems': len(problems), 'n_responses': len(rows), 'cells': cells, 'h1': h1, 'h2': h2,
            'pooled_tutor_leak': estimate(pooled_leak),
            'ceiling': {'primary_correct_commit': estimate(lambda x: x[27]), 'containment_sensitivity': estimate(lambda x: x[28])},
            'leak_ceiling_ratio': ratio(pooled_leak, 27),
            'leak_containment_ceiling_ratio_sensitivity': ratio(pooled_leak, 28),
            'logistic': fit_logistic(tutor),
        }
    if len(models) >= 2 and None not in sizes and len(set(sizes)) == len(sizes):
        log_parameters = np.log(np.array(sizes) * 1e9)
        weights = (log_parameters - log_parameters.mean()) / np.sum((log_parameters - log_parameters.mean()) ** 2)
        h3 = {'status': 'ok' if eligible else 'descriptive_only', 'predictor': 'natural log(parameter count)', 'model_parameters': dict(zip(models, [size * 1e9 for size in sizes]))}
        for key in (*FORMATS, 'pooled'):
            h3[key] = _interval(np.dot(weights, [h2_points[m][key] for m in models]),
                                weights @ np.array([h2_samples[m][key] for m in models]))
        report['h3'] = h3
    else:
        report['h3'] = {'status': 'unavailable', 'reason': 'Requires at least two complete models with distinct recognized Qwen3 sizes.'}
    passing = [model for model, result in report['models'].items()
               if any(contrast['ci95'][0] is not None and contrast['ci95'][0] > 0
                      for hypothesis in ('h1', 'h2') for contrast in result[hypothesis].values())]
    report['phase2_gate'] = {'eligible': eligible, 'passed': eligible and len(passing) >= 2,
                             'positive_interval_models': passing, 'required_models': 2,
                             'rule': 'H1 or H2 positive 95% interval in at least two models; unadjusted contrasts; inference eligibility required.'}
    return report


def handcheck(df: pd.DataFrame, output_dir: str | Path, *, seed: int = 0) -> dict:
    """Write deterministic audit samples; preserve and score only human-entered labels."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for model, rows in df.groupby('model', sort=True):
        name = re.sub(r'[^A-Za-z0-9_.-]+', '-', model)
        name += '-' + hashlib.sha256(model.encode()).hexdigest()[:8]
        path = output_dir / f'handcheck_{name}.csv'
        ordered = rows.sort_values('id')
        rng = np.random.default_rng(seed)
        positions = sorted(rng.choice(len(ordered), size=min(100, len(ordered)), replace=False))
        selected = ordered.iloc[positions]
        saved = {}
        if path.exists():
            with path.open(newline='', encoding='utf-8') as handle:
                for row in csv.DictReader(handle):
                    if row['id'] in saved:
                        raise ValueError(f'duplicate handcheck id in {path}')
                    saved[row['id']] = row
            if set(saved) != set(selected.id):
                raise ValueError(f'handcheck sample differs from existing labels in {path}; use a fresh output directory')
        records = []
        for row in selected.to_dict('records'):
            record = {key: row[key] for key in ('model', 'id', 'problem_idx', 'cue', 'format', 'template_idx', 'arm', 'gold', 'response')}
            for metric in METRICS:
                record[metric] = int(row[metric])
                manual = saved.get(row['id'], {}).get('manual_' + metric, '').strip()
                if manual not in ('', '0', '1'):
                    raise ValueError(f'{path}: manual_{metric} must be blank, 0, or 1')
                record['manual_' + metric] = manual
            previous = saved.get(row['id'])
            if previous and any(str(previous[key]) != str(record[key]) for key in ('model', 'response', 'gold')):
                raise ValueError(f'handcheck response/gold changed for {row["id"]}; refusing stale labels')
            record['manual_notes'] = saved.get(row['id'], {}).get('manual_notes', '')
            records.append(record)
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        summary = {'path': str(path), 'n_sampled': len(records), 'n_labeled': 0}
        for metric in METRICS:
            labeled = [row for row in records if row['manual_' + metric] != '']
            summary[metric + '_n_labeled'] = len(labeled)
            summary[metric + '_agreement'] = sum(int(row['manual_' + metric]) == row[metric] for row in labeled) / len(labeled) if labeled else None
        summary['n_labeled'] = summary['leak_n_labeled']
        summary['status'] = 'manually_labeled' if summary['n_labeled'] else 'awaiting_manual_labels'
        summaries[model] = summary
    return summaries


def render_summary(report: dict) -> str:
    """Human-readable companion to the full machine-readable statistical report."""
    def interval(value, percent=False):
        scale = 100 if percent else 1
        def number(x):
            return 'undefined' if x is None else f'{x * scale:.2f}' + ('%' if percent else '')
        return f"{number(value['estimate'])} [{number(value['ci95'][0])}, {number(value['ci95'][1])}]"
    lines = ['# Phase 1 results', '',
             f"Problem-cluster bootstrap: {report['bootstrap']['resamples']:,} resamples; 95% percentile intervals; seed {report['bootstrap']['seed']}.", '',
             'Inference eligibility: ' + str(report['inference']['eligible']) + '.']
    lines.extend('- ' + reason for reason in report['inference']['reasons'])
    for model, result in report['models'].items():
        lines += ['', f'## {model}', '', f"{result['n_problems']} problems; {result['n_responses']} responses.", '',
                  'Tutor leak: ' + interval(result['pooled_tutor_leak'], True) + '.',
                  'Ceiling solve (correct commit): ' + interval(result['ceiling']['primary_correct_commit'], True) + '.',
                  'Ceiling containment sensitivity: ' + interval(result['ceiling']['containment_sensitivity'], True) + '.',
                  'Tutor leak / ceiling solve: ' + interval(result['leak_ceiling_ratio']) + '.', '',
                  '| Cue | Format | Leak, 95% CI | Commit, 95% CI | Correct commit, 95% CI | Leak / ceiling, 95% CI |',
                  '|---|---|---|---|---|---|']
        for cell in result['cells']:
            lines.append('| ' + ' | '.join([cell['cue'], cell['format'],
                *(interval(cell[m], True) for m in METRICS), interval(cell['leak_ceiling_ratio'])]) + ' |')
        lines += ['', 'Risk contrasts (percentage-point differences; 95% CI):', '',
                  '| Hypothesis | Contrast | Estimate [95% CI], percentage points |', '|---|---|---|']
        for hypothesis in ('h1', 'h2'):
            for name, value in result[hypothesis].items():
                lines.append(f"| {hypothesis.upper()} | {name} | {interval(value, True).replace('%', '')} |")
        fit = result['logistic']
        lines += ['', f"Logistic regression: **{fit['status']}**; problem-clustered standard errors."]
        if fit['status'] != 'ok':
            lines.append(fit.get('reason', ''))
            if fit.get('separated_cells'):
                lines.append('Separated cells: ' + ', '.join(fit['separated_cells']) + '.')
        else:
            lines += ['', '| Term | Log-odds estimate | Cluster SE | 95% CI |', '|---|---|---|---|']
            for row in fit['coefficients']:
                lines.append(f"| {row['term']} | {row['estimate']:.4f} | {row['standard_error']:.4f} | [{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}] |")
        audit = report.get('handcheck', {}).get(model)
        if audit:
            lines += ['', f"Manual handcheck: {audit['n_sampled']} sampled, {audit['n_labeled']} leak labels entered; {audit['status']}.",
                      f"CSV: `{audit['path']}`."]
            for metric in METRICS:
                agreement = audit[metric + '_agreement']
                lines.append(f"- {metric}: {audit[metric + '_n_labeled']} manual labels; agreement " + ('unavailable.' if agreement is None else f'{agreement:.1%}.'))
    lines += ['', '## Across-model diagnostics', '', 'H3: ' + report['h3']['status'] + '.',
              'Phase 2 gate passed: ' + str(report['phase2_gate']['passed']) + '.', '',
              'Smoke results are descriptive pipeline checks. Handcheck agreement requires real manual labels. All contrasts are unadjusted for multiplicity. See docs/analysis-methods.md for estimands, ceiling sensitivity, and undefined-ratio handling. Full diagnostics and counts are in summary.json.', '']
    if 'pooled' in report['h3']:
        lines.insert(-1, 'H3 pooled slope per natural-log parameter count: ' + interval(report['h3']['pooled']) + '.')
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--resamples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--min-gold', type=int, default=None,
                        help='Sensitivity analysis: keep only problems whose gold answer is at least this value.')
    args = parser.parse_args()
    try:
        df = pd.DataFrame(list(read_rows(args.input)))
        if args.min_gold is not None:
            df = df[df['gold'] >= args.min_gold].reset_index(drop=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = analyze(df, resamples=args.resamples, seed=args.seed)
        report['min_gold'] = args.min_gold
        if args.min_gold is None:
            report['handcheck'] = handcheck(df, args.output_dir, seed=args.seed)
        destination = args.output_dir / 'summary.json'
        destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
        (args.output_dir / 'summary.md').write_text(render_summary(report), encoding='utf-8')
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f'Analysis -> {destination}; inference eligible: {report["inference"]["eligible"]}')


if __name__ == '__main__':
    main()
