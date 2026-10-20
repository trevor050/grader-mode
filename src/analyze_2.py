"""Phase 2 test-only paired problem-cluster analysis, without interpretation.

Accept a DataFrame or list of already-scored rows. Every supplied model must
have exactly ranks 101..200 and the registered 20 cells at template indices
0..3 (0..1 only with explicit expected_templates=2). Development rows may be
partial selection grids; their metadata is validated, but no development
outcomes enter estimates. This module never chooses layers or coefficients;
the caller remains responsible for development-only selection provenance.

All rates retain every response, including degenerate responses. Templates,
random directions, and held-out formats receive equal weight. Linear paired
contrasts are formed within problems before calling bootstrap_estimate, so
each resample retains all arms and templates of a sampled problem together.
The gold >= 10 subset is formed only after validating the entire input grid.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.analyze import METRICS, bootstrap_estimate

HELDOUT_FORMATS = ('boxed_alt_1', 'boxed_alt_2', 'final')
RANDOM_ABLATIONS = tuple(f'ablate_r{i}' for i in range(5))
RANDOM_ADDITIONS = tuple(f'add_r{i}' for i in range(5))
DEGENERATION = ('empty', 'repetition', 'low_number_count', 'degenerate')
CELLS = (
    tuple(('boxed', arm) for arm in ('baseline', 'ablate_d', *RANDOM_ABLATIONS))
    + tuple(('none', arm) for arm in ('baseline', 'add_d', *RANDOM_ADDITIONS))
    + tuple((fmt, arm) for fmt in HELDOUT_FORMATS for arm in ('baseline', 'ablate_d'))
)


def _is_integer(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _validate_grid(df: pd.DataFrame, expected_templates: int) -> None:
    required = {'model', 'id', 'problem_idx', 'selection_rank', 'split', 'format',
                'template_idx', 'intervention', 'gold', *METRICS, *DEGENERATION}
    if df.columns.duplicated().any():
        raise ValueError('duplicate column names')
    if required - set(df):
        raise ValueError(f'missing scored fields: {sorted(required - set(df))}')
    if df.empty or df[list(required)].isna().any().any():
        raise ValueError('empty input or null scored fields')
    for name in ('model', 'id', 'split', 'format', 'intervention'):
        if not df[name].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f'{name} must contain nonempty strings')
    for name in ('problem_idx', 'selection_rank', 'template_idx', 'gold'):
        if not df[name].map(_is_integer).all():
            raise ValueError(f'{name} must contain integers')
    if df.problem_idx.lt(0).any():
        raise ValueError('problem_idx must retain original nonnegative dataset indices')
    if not df.selection_rank.between(1, 200).all():
        raise ValueError('selection_rank must be in 1..200')
    if not df.split.isin(('dev', 'test')).all() or not df.split.eq(
            df.selection_rank.map(lambda r: 'dev' if r <= 100 else 'test')).all():
        raise ValueError('split must be dev for ranks 1..100 and test for ranks 101..200')
    if not df.template_idx.between(0, 3).all():
        raise ValueError('template_idx must be in 0..3')
    actual_cells = set(df.loc[df.split.eq('test'), ['format', 'intervention']].itertuples(index=False, name=None))
    if not actual_cells <= set(CELLS):
        raise ValueError('unknown format/intervention cell')
    for fmt, arm in df.loc[df.split.eq('dev'), ['format','intervention']].itertuples(index=False,name=None):
        if (fmt,arm) not in CELLS and not (
                fmt=='boxed' and re.fullmatch(r'(screen|confirm)_L[0-9]+',arm)
                or fmt=='none' and re.fullmatch(r'alpha_m(2|4|8|16)',arm)):
            raise ValueError('unknown development format/intervention cell')
    if df.duplicated(['model', 'id']).any():
        raise ValueError('duplicate IDs per model')
    if df.duplicated(['model', 'problem_idx', 'format', 'intervention', 'template_idx']).any():
        raise ValueError('duplicate problem/template/format/intervention cells per model')
    # Original dataset indices and selection ranks are distinct identities.
    # Enforce one shared bijection, including across models and dev/test rows.
    if (df.groupby('problem_idx').selection_rank.nunique().ne(1).any()
            or df.groupby('selection_rank').problem_idx.nunique().ne(1).any()):
        raise ValueError('problem_idx and selection_rank must have a one-to-one mapping')
    if df.groupby('problem_idx').gold.nunique().ne(1).any():
        raise ValueError('inconsistent gold for a problem across cells/templates/models')
    for metric in METRICS:
        if not df[metric].map(lambda x: isinstance(x, (bool, np.bool_)) or
                             (_is_integer(x) and x in (0, 1))).all():
            raise ValueError(f'{metric} must be binary')
    if ((df.correct_commit > df.leak) | (df.correct_commit > df.commit)).any():
        raise ValueError('correct_commit requires both leak and commit')
    for flag in DEGENERATION:
        if not df[flag].map(lambda x: isinstance(x, (bool, np.bool_))).all():
            raise ValueError(f'{flag} must be boolean')
    if not df.degenerate.eq(df[list(DEGENERATION[:-1])].any(axis=1)).all():
        raise ValueError('degenerate must equal empty OR repetition OR low_number_count')
    expected = {(fmt, arm, t) for fmt, arm in CELLS for t in range(expected_templates)}
    for model, rows in df.groupby('model', sort=True):
        test = rows[rows.split == 'test']
        if set(test.selection_rank) != set(range(101, 201)):
            raise ValueError(f'{model}: exactly 100 test problems with ranks 101..200 required')
        for problem, problem_rows in test.groupby('problem_idx', sort=True):
            actual = list(problem_rows[['format', 'intervention', 'template_idx']].itertuples(index=False, name=None))
            if len(actual) != len(expected) or set(actual) != expected:
                raise ValueError(f'incomplete/unbalanced grid: model={model}, problem_idx={problem}; '
                                 f'expected 20 cells x {expected_templates} templates')


def _estimate(values: np.ndarray, *, resamples: int, seed: int) -> dict:
    if not len(values):
        return {'estimate': None, 'ci95': [None, None],
                'valid_resamples': 0, 'undefined_resamples': resamples}
    return bootstrap_estimate(np.asarray(values, dtype=float)[:, None], lambda x: x[0],
                              resamples=resamples, seed=seed)


def _summarize(rows: pd.DataFrame, *, resamples: int, seed: int) -> dict:
    # Fixed rank order makes results invariant to input order or dataset index
    # numbering. The sampling unit is still each original problem_idx.
    problems = rows[['problem_idx', 'selection_rank']].drop_duplicates().sort_values('selection_rank').problem_idx.tolist()
    n = len(problems)
    cells, leaks, flags = [], {}, {}
    for fmt, arm in CELLS:
        subset = rows[(rows['format'] == fmt) & (rows.intervention == arm)]
        means = subset.groupby('problem_idx')[list(METRICS) + list(DEGENERATION)].mean().reindex(problems)

        def rate(name):
            return _estimate(means[name].to_numpy(), resamples=resamples, seed=seed) | {
                'n_positive': int(subset[name].sum()), 'n_responses': len(subset)}

        steered = arm != 'baseline'
        # Use counts for the strict > 10% threshold, without float rounding.
        flag = bool(steered and 10 * int(subset.degenerate.sum()) > len(subset)) if n else None
        result = {'format': fmt, 'intervention': arm, 'steered': steered,
                  'n_problems': n, 'n_responses': len(subset),
                  **{metric: rate(metric) for metric in METRICS},
                  'not_interpretable': flag}
        if steered:
            result['degeneration'] = {name: rate(name) for name in DEGENERATION}
        cells.append(result)
        leaks[(fmt, arm)] = means.leak.to_numpy()
        flags[(fmt, arm)] = flag

    def contrast(positive, negative):
        # Averaging inside each problem preserves paired random-direction and
        # format dependence. All grids are balanced, including sensitivity.
        values = (np.mean([leaks[c] for c in positive], axis=0)
                  - np.mean([leaks[c] for c in negative], axis=0))
        used = [c for c in (*positive, *negative) if c[1] != 'baseline']
        flagged = [dict(format=c[0], intervention=c[1]) for c in used if flags[c]]
        return _estimate(values, resamples=resamples, seed=seed) | {
            'n_problems': n, 'not_interpretable': bool(flagged) if n else None,
            'degenerate_cells': flagged}

    return {
        'n_problems': n, 'n_responses': len(rows), 'problem_indices': [int(p) for p in problems],
        'selection_ranks': sorted(int(r) for r in rows.selection_rank.unique()), 'cells': cells,
        'h9': {
            'necessity': contrast([('boxed', 'baseline')], [('boxed', 'ablate_d')]),
            'superiority': contrast([('boxed', r) for r in RANDOM_ABLATIONS], [('boxed', 'ablate_d')]),
        },
        'h10': {
            'sufficiency': contrast([('none', 'add_d')], [('none', 'baseline')]),
            'superiority': contrast([('none', 'add_d')], [('none', r) for r in RANDOM_ADDITIONS]),
        },
        'h11': {
            **{fmt: contrast([(fmt, 'baseline')], [(fmt, 'ablate_d')]) for fmt in HELDOUT_FORMATS},
            'pooled': contrast([(fmt, 'baseline') for fmt in HELDOUT_FORMATS],
                               [(fmt, 'ablate_d') for fmt in HELDOUT_FORMATS]),
        },
    }


def analyze(df: pd.DataFrame | list[dict], resamples: int = 2000,
            seed: int = 0, expected_templates: int = 4) -> dict:
    """Validate complete test grids, then report rates, H9–H11, and gold >= 10.

    Rates and contrasts are fractions; ci95 contains unadjusted percentile
    bounds. Each model/subset reports 20 cell rates with success/response
    counts, checks for all 15 steered cells, and eight signed leak contrasts.
    No rows are excluded because of degeneration. not_interpretable is true
    above 10% degeneration in a contributing steered cell, false otherwise,
    and null for the empty sensitivity subset. Empty estimates/CIs are null.
    expected_templates=2 explicitly selects the authorized reduced grid.
    """
    if not _is_integer(resamples) or resamples < 1:
        raise ValueError('resamples must be a positive integer')
    if not _is_integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not _is_integer(expected_templates) or expected_templates not in (2, 4):
        raise ValueError('expected_templates must be 4 or explicitly authorized 2')
    resamples, seed, expected_templates = int(resamples), int(seed), int(expected_templates)
    df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    _validate_grid(df, expected_templates)
    test = df[df.split == 'test']
    models = sorted(df.model.unique())

    def dataset(rows):
        return {'n_problems': int(rows.problem_idx.nunique()), 'n_responses': len(rows),
                'models': {model: _summarize(rows[rows.model == model], resamples=resamples, seed=seed)
                           for model in models}}

    return dataset(test) | {
        'phase': 'phase2', 'expected_templates': expected_templates,
        'bootstrap': {'resamples': resamples, 'seed': seed, 'unit': 'problem_idx',
                      'interval': 'percentile 95%', 'paired': True},
        'inference': {'eligible': True, 'split': 'test', 'required_problems': 100,
                      'selection_rank_min': 101, 'selection_rank_max': 200},
        'selection_performed': False, 'n_dev_responses_excluded': int(df.split.eq('dev').sum()),
        'rate_denominator': 'all responses, including degenerate responses',
        'random_direction_pooling': 'equal-weight mean over five directions within each problem',
        'heldout_format_pooling': 'equal-weight mean over boxed_alt_1, boxed_alt_2, final within each problem',
        'degeneration_threshold': .10,
        'gold_ge_10': dataset(test[test.gold >= 10]) | {'min_gold': 10},
    }


def render_summary(report: dict) -> str:
    """Render counts, rates, intervals, and registered degeneration flags only."""
    def interval(value):
        def number(x):
            return 'undefined' if x is None else f'{100 * x:.2f}'
        return f"{number(value['estimate'])} [{number(value['ci95'][0])}, {number(value['ci95'][1])}]"

    def flag(value):
        return 'undefined' if value is None else str(value).lower()

    lines = ['# Phase 2 results', '',
             f"Problem-cluster bootstrap: {report['bootstrap']['resamples']:,} resamples; "
             f"95% percentile intervals; seed {report['bootstrap']['seed']}.",
             'Rates: percent; contrasts: percentage points; intervals: unadjusted 95%.',
             f"Test ranks: 101..200; templates: {report['expected_templates']}; "
             f"development responses excluded: {report['n_dev_responses_excluded']}."]
    for label, dataset in (('All test problems', report), ('gold >= 10', report['gold_ge_10'])):
        lines += ['', f'## {label}', '']
        for model, result in dataset['models'].items():
            lines += [f'### {model}', '',
                      f"Problems: {result['n_problems']}; responses: {result['n_responses']}.", '',
                      '| Format | Intervention | Responses | leak n | leak [95% CI] | commit n | commit [95% CI] | correct_commit n | correct_commit [95% CI] |',
                      '|---|---|---|---|---|---|---|---|---|']
            for c in result['cells']:
                values = [c['format'], c['intervention'], str(c['n_responses'])]
                for metric in METRICS:
                    values.extend([str(c[metric]['n_positive']), interval(c[metric])])
                lines.append('| ' + ' | '.join(values) + ' |')
            lines += ['', '| Hypothesis | Contrast | Problems | Estimate [95% CI] | not_interpretable |',
                      '|---|---|---|---|---|']
            for hypothesis in ('h9', 'h10', 'h11'):
                for name, value in result[hypothesis].items():
                    lines.append(f"| {hypothesis.upper()} | {name} | {value['n_problems']} | "
                                 f"{interval(value)} | {flag(value['not_interpretable'])} |")
            lines += ['', 'Degeneration threshold: > 10%.', '',
                      '| Format | Intervention | Responses | Check | Count | Rate [95% CI] | not_interpretable |',
                      '|---|---|---|---|---|---|---|']
            for c in result['cells']:
                if c['steered']:
                    for name, value in c['degeneration'].items():
                        lines.append(f"| {c['format']} | {c['intervention']} | {c['n_responses']} | "
                                     f"{name} | {value['n_positive']} | {interval(value)} | "
                                     f"{flag(c['not_interpretable'])} |")
    return '\n'.join(lines) + '\n'


def main():
    import argparse
    import json
    from pathlib import Path
    from src.score import read_rows, score_response
    from src.phase2 import degeneration
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-dir', default='results/phase2', type=Path)
    args = parser.parse_args()
    selection = json.loads((args.output_dir/'selection.json').read_text())
    rows = list(read_rows(args.input))
    from src.phase2_validation import validate_execution
    manifest=json.loads((args.output_dir/'manifest.json').read_text())
    metrics=json.loads((args.output_dir/'metrics.json').read_text())
    verification=validate_execution(rows,selection,manifest,metrics)
    # Independently rescore persisted text with the original scorer and diagnostics.
    for row in rows:
        computed = {**score_response(row['response'],row['gold']),
                    **degeneration(row['response'],row['question'])}
        if any(row.get(k)!=v for k,v in computed.items()):
            raise ValueError(f"Persisted score/degeneration mismatch for {row['id']}")
    test = [r for r in rows if r['split']=='test']
    report = analyze(test, expected_templates=selection['test_templates'])
    report['n_dev_responses_excluded'] = len(rows)-len(test)
    report['execution_verification'] = verification
    report['selection'] = selection
    report['development_checks'] = []
    dev = pd.DataFrame([r for r in rows if r['split']=='dev'])
    for (fmt,arm), cell_rows in dev.groupby(['format','intervention'],sort=True):
        if arm=='baseline':continue
        checks = {key:{'count':int(cell_rows[key].sum()),'rate':float(cell_rows[key].mean())}
                  for key in DEGENERATION}
        report['development_checks'].append({'format':fmt,'intervention':arm,
                    'n_responses':len(cell_rows),'degeneration':checks,
                    'not_interpretable':10*int(cell_rows.degenerate.sum())>len(cell_rows)})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    text = render_summary(report)
    text += '\n## Development selections\n\n'
    text += f"Layer (zero-based): {selection['layer']}; alpha: {selection['alpha']:.6f}; multiplier: {selection['alpha_multiplier']}; scale: {selection['alpha_scale']:.6f}.\n"
    text += '\n| Format | Intervention | Responses | Empty | Repetition | Low number count | Composite | not_interpretable |\n|---|---|---|---|---|---|---|---|\n'
    for c in report['development_checks']:
        text += '| '+' | '.join([c['format'],c['intervention'],str(c['n_responses']),
                 *(str(c['degeneration'][k]['count']) for k in DEGENERATION),str(c['not_interpretable']).lower()])+' |\n'
    (args.output_dir/'summary.md').write_text(text)
    print(f"Phase 2: {len(test)} test responses; {report['n_problems']} test problems; 2000 resamples")


if __name__=='__main__':
    main()
