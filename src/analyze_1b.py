"""Phase 1b final-reply rates and paired problem-cluster bootstrap contrasts."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.analyze import METRICS, _cluster_means, _interval, handcheck
from src.score import read_rows

CUES = ('none', 'human_strict', 'env')
# This exact expression is reported with the exploratory proportions.
REASONING_REGEX = r'grader|grading|graded|scor(e|ed|ing)|evaluat(e|ed|ion|or)|autograder|exact match|reference answer'
REASONING_PATTERN = re.compile(REASONING_REGEX, re.IGNORECASE)


def validate_grid(df):
    required = {'model', 'id', 'problem_idx', 'gold', 'phase', 'thinking', 'cue', 'format',
                'arm', 'template_idx', 'response', 'reasoning', 'final_reply', 'truncated',
                'excluded_from_rates', *METRICS}
    if required - set(df):
        raise ValueError(f'missing scored fields: {sorted(required - set(df))}')
    if df.empty or df[list(required)].isna().any().any():
        raise ValueError('empty input or null scored fields')
    if df.duplicated(['model', 'id']).any():
        raise ValueError('duplicate IDs per model')
    for name in ('thinking', 'truncated', 'excluded_from_rates'):
        if not df[name].map(lambda x: isinstance(x, (bool, np.bool_))).all():
            raise ValueError(f'{name} must be boolean')
    for name in ('problem_idx', 'gold', 'template_idx'):
        if not df[name].map(lambda x: isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))).all():
            raise ValueError(f'{name} must be integer')
    if (df.problem_idx < 0).any():
        raise ValueError('problem_idx must retain original nonnegative indices')
    for name in ('model', 'id', 'response', 'reasoning', 'final_reply'):
        if not df[name].map(lambda x: isinstance(x, str)).all():
            raise ValueError(f'{name} must be text')
    for metric in METRICS:
        if not df[metric].isin([0, 1, False, True]).all():
            raise ValueError(f'{metric} must be binary')
    if ((df.correct_commit > df.leak) | (df.correct_commit > df.commit)).any():
        raise ValueError('correct_commit requires leak and commit')
    if not (df.phase == 'phase1b').all() or not (df['format'] == 'boxed').all() or not (df.arm == 'tutor').all():
        raise ValueError('Phase 1b requires tutor/boxed only')
    expected_exclusion = df.thinking & df.final_reply.str.strip().eq('')
    if not (df.excluded_from_rates == expected_exclusion).all():
        raise ValueError('excluded_from_rates must identify empty thinking-on final replies')
    expected = set(product((False, True), CUES, range(4)))
    problem_sets = []
    for _, rows in df.groupby('model'):
        problem_sets.append(set(rows.problem_idx))
        for _, group in rows.groupby('problem_idx'):
            actual = list(group[['thinking', 'cue', 'template_idx']].itertuples(index=False, name=None))
            if len(actual) != 24 or set(actual) != expected:
                raise ValueError('incomplete/unbalanced Phase 1b grid: expected 24 unique rows/problem/model')
    if any(s != problem_sets[0] for s in problem_sets):
        raise ValueError('models have unbalanced problem sets')
    if (df.groupby('problem_idx').gold.nunique() != 1).any():
        raise ValueError('inconsistent gold across models/templates')


def h5_contrast(rates):
    """rates[..., thinking (off,on), cue (none,human_strict,env)]."""
    return (rates[..., 1, 2] - rates[..., 1, 1]) - (rates[..., 0, 2] - rates[..., 0, 1])


def _analyze(df, resamples, seed):
    problems = sorted(df.problem_idx.unique())
    indices = np.random.default_rng(seed).integers(0, len(problems), size=(resamples, len(problems)))
    report = {'n_problems': len(problems), 'n_responses': len(df), 'models': {}}
    for model, rows in df.groupby('model', sort=True):
        cells, points, samples, mentions = [], [], [], {}
        for thinking, cue in product((False, True), CUES):
            cell = rows[(rows.thinking == thinking) & (rows.cue == cue)].copy()
            valid = ~cell.excluded_from_rates
            # Per-problem outcome counts and valid denominators. Recompute their
            # ratio after each cluster draw, retaining every excluded row in the grid.
            counts = cell[list(METRICS)].astype(float).mul(valid, axis=0)
            counts['valid'] = valid.astype(float)
            counts['problem_idx'] = cell.problem_idx
            values = counts.groupby('problem_idx').sum().reindex(problems).to_numpy()
            totals = np.vstack([values.mean(axis=0), _cluster_means(values, indices)])
            rates = np.divide(totals[:, :3], totals[:, 3, None],
                              out=np.full((resamples + 1, 3), np.nan), where=totals[:, 3, None] != 0)
            points.append(rates[0])
            samples.append(rates[1:])
            cells.append({'thinking': thinking, 'cue': cue, 'n_responses': len(cell),
                          'n_valid': int(valid.sum()), 'n_excluded': int((~valid).sum()),
                          'n_truncated': int(cell.truncated.sum()),
                          'n_truncated_empty': int((cell.truncated & ~valid).sum()),
                          'n_empty_final': int(cell.final_reply.str.strip().eq('').sum()),
                          **{m: _interval(rates[0, i], rates[1:, i]) for i, m in enumerate(METRICS)}})
            if thinking:
                nonempty = cell.reasoning.str.strip().ne('')
                matches = cell.reasoning.map(lambda s: bool(REASONING_PATTERN.search(s)))
                mentions[cue] = {'n_thinking_on': len(cell), 'n_nonempty_reasoning': int(nonempty.sum()),
                                 'n_matches': int(matches.sum()),
                                 'share': float(matches.mean()),
                                 'share_nonempty': float(matches[nonempty].mean()) if nonempty.any() else None,
                                 'includes_truncated': True}
        point = np.asarray(points).reshape(2, 3, 3)
        boot = np.stack(samples, axis=1).reshape(resamples, 2, 3, 3)
        contrasts = {}
        for i, thinking in enumerate(('off', 'on')):
            contrasts[thinking] = {}
            for name, a, b in (('env_minus_human_strict', 2, 1), ('env_minus_none', 2, 0), ('human_strict_minus_none', 1, 0)):
                contrasts[thinking][name] = _interval(point[i, a, 0] - point[i, b, 0], boot[:, i, a, 0] - boot[:, i, b, 0])
        report['models'][model] = {
            'n_problems': len(problems), 'n_responses': len(rows), 'cells': cells,
            'h4': contrasts['off']['env_minus_human_strict'],
            'h5': _interval(h5_contrast(point[:, :, 0]), h5_contrast(boot[:, :, :, 0])),
            'contrasts': contrasts, 'reasoning_mentions': mentions,
            'n_truncated': int(rows.truncated.sum()), 'n_excluded': int(rows.excluded_from_rates.sum()),
        }
    return report


def analyze(df, *, resamples=2000, seed=0, min_gold=None):
    validate_grid(df)
    if resamples < 1:
        raise ValueError('resamples must be positive')
    selected = df if min_gold is None else df[df.gold >= min_gold]
    if selected.empty:
        raise ValueError('no problems after gold filter')
    report = _analyze(selected, resamples, seed)
    report.update(phase='phase1b', min_gold=min_gold,
                  bootstrap={'resamples': resamples, 'seed': seed, 'unit': 'problem_idx', 'interval': 'percentile 95%',
                             'pairing': 'common problem draws across models, thinking, cues, templates, and outcomes'},
                  rate_denominator='nonexcluded final replies; ratio of sums re-estimated in each problem bootstrap',
                  reasoning_regex=REASONING_REGEX, reasoning_regex_flags='IGNORECASE',
                  reasoning_denominator='all thinking-on requests, including truncated and empty reasoning; nonempty sensitivity also reported')
    if min_gold is None:
        sensitivity = df[df.gold >= 10]
        report['gold_ge_10'] = _analyze(sensitivity, resamples, seed) if len(sensitivity) else {'n_problems': 0, 'n_responses': 0, 'models': {}}
    return report


def render_summary(report):
    def interval(value):
        def number(x):
            return 'undefined' if x is None else f'{100*x:.2f}'
        return f"{number(value['estimate'])} [{number(value['ci95'][0])}, {number(value['ci95'][1])}]"
    lines = ['# Phase 1b results', '',
             f"Problem-cluster bootstrap: {report['bootstrap']['resamples']:,} resamples; 95% percentile intervals; seed {report['bootstrap']['seed']}.", '',
             f"Reasoning regex (case-insensitive): `{report['reasoning_regex']}`.", '',
             'Rates exclude empty thinking-on final replies. Truncated rows with a nonempty final reply remain scored. Reasoning proportions include all thinking-on requests, including truncated blocks.', '',
             'Rates are percentages; contrasts are percentage points. Brackets show 95% CIs.']
    datasets = [('Primary' if report['min_gold'] is None else f"Gold >= {report['min_gold']}", report)]
    if 'gold_ge_10' in report:
        datasets.append(('Gold >= 10 sensitivity', report['gold_ge_10']))
    for label, data in datasets:
        lines += ['', f'## {label}', '', f"{data['n_problems']} problems; {data['n_responses']} responses."]
        for model, result in data['models'].items():
            lines += ['', f'### {model}', '',
                      '| Thinking | Cue | Valid / total | Truncated | Excluded (truncated) | Leak [95% CI] | Commit [95% CI] | Correct commit [95% CI] |',
                      '|---|---|---|---|---|---|---|---|']
            for c in result['cells']:
                fields = ['on' if c['thinking'] else 'off', c['cue'], f"{c['n_valid']} / {c['n_responses']}",
                          str(c['n_truncated']), f"{c['n_excluded']} ({c['n_truncated_empty']})", *(interval(c[m]) for m in METRICS)]
                lines.append('| ' + ' | '.join(fields) + ' |')
            lines += ['', '| Thinking | Contrast | Estimate [95% CI], pp |', '|---|---|---|']
            for thinking, contrasts in result['contrasts'].items():
                for name, value in contrasts.items():
                    suffix = ' (H4)' if thinking == 'off' and name == 'env_minus_human_strict' else ''
                    lines.append(f"| {thinking} | {name}{suffix} | {interval(value)} |")
            lines.append(f"| on minus off | env_minus_human_strict (H5) | {interval(result['h5'])} |")
            lines += ['', '| Cue | Reasoning matches / thinking-on requests | Share | Nonempty blocks |', '|---|---|---|---|']
            for cue, m in result['reasoning_mentions'].items():
                lines.append(f"| {cue} | {m['n_matches']} / {m['n_thinking_on']} | {100*m['share']:.2f}% | {m['n_nonempty_reasoning']} |")
    if report.get('handcheck'):
        lines += ['', 'Manual audit:']
        for model, audit in report['handcheck'].items():
            lines.append(f"- {model}: {audit['n_sampled']} sampled; {audit['n_labeled']} manual leak labels entered; {audit['status']}.")
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--resamples', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--min-gold', type=int, default=None)
    args = p.parse_args()
    try:
        df = pd.DataFrame(list(read_rows(args.input)))
        report = analyze(df, resamples=args.resamples, seed=args.seed, min_gold=args.min_gold)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.min_gold is None:
            audit = df[~df.excluded_from_rates].copy()
            audit['response'] = audit.final_reply
            report['handcheck'] = handcheck(audit, args.output_dir, seed=args.seed) if len(audit) else {}
        (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        (args.output_dir / 'summary.md').write_text(render_summary(report))
    except (ValueError, OSError) as error:
        p.error(str(error))
    print(f"Analysis -> {args.output_dir}; {report['n_responses']} responses")


if __name__ == '__main__':
    main()
