"""Phase 3 H12–H14: validated test grids and paired problem-cluster intervals.

Tracking vectors are ordered tutor, control_0..control_4. Slopes are already
least-squares coefficients on normalized position; this module does not select
neurons, doses, or layers. H12 averages eligible traces (ratio of slope sums to
trace counts); H13 equally weights within-problem nonleak-minus-leak means.
H14 recomputes each cell's nonexcluded final-reply rate in each cluster draw,
then averages the five control rates, retaining all templates and cells.

Optional tracking excluded_from_rates preserves Phase 1b empty-final labels:
these rows remain available for H12 and all-outcome trajectories, but cannot
supply H13 outcome groups. Without the optional flag no source exclusions are
inferred from leak=False. Source labels and activation preprocessing require
upstream provenance; summary fields explicitly count missing exclusion flags.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analyze import METRICS, _cluster_means, _interval, bootstrap_estimate
from src.phase3 import TUTOR_PATTERN, TUTOR_REGEX
from src.score import read_rows, score_response, split_thinking

SETS = ('tutor', *(f'control_{i}' for i in range(5)))
INTERVENTIONS = ('baseline', *SETS)
DEGENERATION = ('empty', 'repetition', 'truncation_without_final', 'degenerate')
IDENTITY = ('problem_idx', 'selection_rank', 'template_idx', 'gold')
TRACKING_FIELDS = {'id', *IDENTITY, 'leak', 'closed_reasoning',
                   'n_reasoning_tokens', 'slopes', 'first_quarter', 'deciles'}
GENERATION_FIELDS = {*IDENTITY, 'split', 'intervention', 'response', 'reasoning',
                     'final_reply', 'truncated', 'excluded_from_rates', *METRICS,
                     *DEGENERATION, 'repetition_rate', 'tutor_mention',
                     'output_tokens', 'final_reply_tokens'}


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _boolean(value):
    return isinstance(value, (bool, np.bool_))


def _finite(value):
    return (isinstance(value, (int, float, np.integer, np.floating))
            and not _boolean(value) and bool(np.isfinite(value)))


def _records(rows, name):
    if isinstance(rows, pd.DataFrame):
        if rows.columns.duplicated().any():
            raise ValueError(f'{name}: duplicate columns')
        rows = rows.to_dict('records')
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError(f'{name}: nonempty list of rows required')
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f'{name}: each row must be an object')
    return list(rows)


def _identity(row, required, *, tracking):
    if required - row.keys():
        raise ValueError(f'missing fields: {sorted(required - row.keys())}')
    for field in IDENTITY:
        if not _integer(row[field]):
            raise ValueError(f'{field} must be integer')
    if row['problem_idx'] < 0:
        raise ValueError('problem_idx must be nonnegative')
    if not 1 <= row['selection_rank'] <= 200:
        raise ValueError('selection_rank must be in 1..200')
    if tracking and row['selection_rank'] <= 100:
        raise ValueError('tracking requires test ranks 101..200')
    if not 0 <= row['template_idx'] < 4:
        raise ValueError('template_idx must be in 0..3')
    if 'id' in row and (not isinstance(row['id'], str) or not row['id'].strip()):
        raise ValueError('id must be nonempty text')


def _mapping(rows):
    problems, ranks, golds = {}, {}, {}
    for row in rows:
        p, r, g = (row[k] for k in ('problem_idx', 'selection_rank', 'gold'))
        if problems.setdefault(p, r) != r or ranks.setdefault(r, p) != p:
            raise ValueError('problem_idx/selection_rank mapping must be one-to-one across sources')
        if golds.setdefault(p, g) != g:
            raise ValueError('inconsistent gold for a problem across sources/templates/cells')


def _grid(rows, templates, *, tracking):
    ids, cells = set(), set()
    for row in rows:
        if 'id' in row:
            if row['id'] in ids:
                raise ValueError('duplicate IDs')
            ids.add(row['id'])
        if not tracking and row['split'] == 'dev':
            continue
        cell = (row['selection_rank'], row['template_idx'])
        if not tracking:
            cell += (row['intervention'],)
        if cell in cells:
            raise ValueError('duplicate grid cells')
        cells.add(cell)
    expected = {(rank, t) for rank in range(101, 201) for t in range(templates)}
    if not tracking:
        expected = {(*cell, arm) for cell in expected for arm in INTERVENTIONS}
    if cells != expected:
        raise ValueError(f'incomplete/unbalanced {"tracking" if tracking else "generation"} grid: '
                         f'100 test problems x {templates} templates'
                         + ('' if tracking else ' x 7 cells'))


def _vector(value, name):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 6:
        raise ValueError(f'{name} must contain six values')
    if not all(_finite(x) for x in value):
        raise ValueError(f'{name} must contain finite numbers')


def _validate_tracking(rows):
    for row in rows:
        _identity(row, TRACKING_FIELDS, tracking=True)
        for field in ('leak', 'closed_reasoning'):
            if not _boolean(row[field]):
                raise ValueError(f'{field} must be boolean')
        if 'excluded_from_rates' in row and not _boolean(row['excluded_from_rates']):
            raise ValueError('source excluded_from_rates must be boolean')
        n = row['n_reasoning_tokens']
        if not _integer(n) or n < 0:
            raise ValueError('n_reasoning_tokens must be a nonnegative integer')
        closed = row['closed_reasoning']
        for field, usable in (('slopes', closed and n >= 2),
                              ('first_quarter', closed and n >= 1)):
            if usable:
                _vector(row[field], field)
            elif row[field] is not None:
                raise ValueError(f'{field} must be null for ineligible reasoning')
        deciles = row['deciles']
        if closed and n:
            if not isinstance(deciles, (list, tuple)) or len(deciles) != 10:
                raise ValueError('deciles must contain ten bins')
            for value in deciles:
                if value is not None:
                    _vector(value, 'deciles')
        elif deciles is not None:
            raise ValueError('deciles must be null without closed nonempty reasoning')
    _mapping(rows)
    _grid(rows, 4, tracking=True)


def _validate_generations(rows, expected_templates):
    for row in rows:
        _identity(row, GENERATION_FIELDS, tracking=False)
        expected_split = 'dev' if row['selection_rank'] <= 100 else 'test'
        if row['split'] != expected_split:
            raise ValueError('split must be dev for ranks 1..100 and test for 101..200')
        if row['intervention'] not in INTERVENTIONS:
            raise ValueError('unknown intervention')
        for field in ('response', 'reasoning', 'final_reply'):
            if not isinstance(row[field], str):
                raise ValueError(f'{field} must be text')
        for field in ('truncated', 'excluded_from_rates', *METRICS,
                      *DEGENERATION, 'tutor_mention'):
            if not _boolean(row[field]):
                raise ValueError(f'{field} must be boolean')
        for field in ('output_tokens', 'final_reply_tokens'):
            if not _integer(row[field]) or row[field] < 0:
                raise ValueError(f'{field} must be a nonnegative integer')
        if row['final_reply_tokens'] > row['output_tokens']:
            raise ValueError('final_reply_tokens exceeds output_tokens')
        reasoning, final = split_thinking(row['response'])
        if reasoning != row['reasoning'] or final != row['final_reply']:
            raise ValueError('reasoning/final_reply must match Phase 1b response split')
        scores = score_response(final, int(row['gold']))
        if any(row[k] != v for k, v in scores.items()):
            raise ValueError('persisted score mismatch: score final_reply only')
        empty = not final.strip()
        if row['excluded_from_rates'] != empty:
            raise ValueError('excluded_from_rates must identify empty thinking-on final replies')
        words = final.split()
        grams = [tuple(words[i:i+4]) for i in range(max(0, len(words)-3))]
        rate = (len(grams)-len(set(grams)))/len(grams) if grams else 0.
        if not _finite(row['repetition_rate']) or not np.isclose(row['repetition_rate'], rate, rtol=0, atol=1e-12):
            raise ValueError('repetition_rate mismatch for final reply')
        checks = dict(empty=empty, repetition=rate > .3,
                      truncation_without_final=bool(row['truncated'] and empty))
        checks['degenerate'] = any(checks.values())
        if any(row[k] != v for k, v in checks.items()):
            raise ValueError('degeneration checks must use final reply')
        if row['tutor_mention'] != bool(TUTOR_PATTERN.search(reasoning)):
            raise ValueError('tutor_mention must use the Phase 1b findings tutor-role regex')
    _mapping(rows)
    _grid(rows, expected_templates, tracking=False)


def _estimate(values, statistic, resamples, seed):
    if not len(values):
        return _interval(np.nan, np.full(resamples, np.nan))
    return bootstrap_estimate(np.asarray(values, dtype=float), statistic,
                              resamples=resamples, seed=seed)


def _ratio(numerator, denominator):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.divide(numerator, denominator,
                         out=np.full(np.broadcast_shapes(np.shape(numerator), np.shape(denominator)), np.nan),
                         where=np.asarray(denominator) != 0)


def _tracking_summary(rows, resamples, seed):
    problems = sorted({r['selection_rank'] for r in rows})
    by_problem = {p: [r for r in rows if r['selection_rank'] == p] for p in problems}
    counts = np.zeros((len(problems), 3))
    mixed, n_h13_traces, n_h13_eligible = [], 0, 0
    for i, group in enumerate(by_problem.values()):
        slopes = [r['slopes'] for r in group if r['closed_reasoning'] and r['n_reasoning_tokens'] >= 2]
        for slope in slopes:
            counts[i] += (slope[0], slope[0]-np.mean(slope[1:]), 1)
        eligible = [r for r in group if r['closed_reasoning'] and r['n_reasoning_tokens'] >= 1
                    and not r.get('excluded_from_rates', False)]
        n_h13_eligible += len(eligible)
        nonleak = [r['first_quarter'][0] for r in eligible if not r['leak']]
        leak = [r['first_quarter'][0] for r in eligible if r['leak']]
        if nonleak and leak:
            mixed.append([np.mean(nonleak)-np.mean(leak)])
            n_h13_traces += len(eligible)
    n_slopes = int(counts[:, 2].sum())
    exclusions = {
        'no_closed_reasoning': sum(not r['closed_reasoning'] for r in rows),
        'closed_fewer_than_two_tokens': sum(r['closed_reasoning'] and r['n_reasoning_tokens'] < 2 for r in rows),
        'source_excluded_from_rates': sum(bool(r.get('excluded_from_rates', False)) for r in rows),
        'source_exclusion_flag_missing': sum('excluded_from_rates' not in r for r in rows),
    }
    return {
        'tracking_exclusions': exclusions,
        'h12': {'n_traces': n_slopes, 'n_excluded': len(rows)-n_slopes,
                'n_problems': int(np.count_nonzero(counts[:, 2])),
                'tutor_slope': _estimate(counts, lambda x: _ratio(x[0], x[2]), resamples, seed),
                'tutor_minus_mean_controls': _estimate(counts, lambda x: _ratio(x[1], x[2]), resamples, seed)},
        'h13': {'n_mixed_problems': len(mixed), 'n_nonmixed_or_ineligible_problems': len(problems)-len(mixed),
                'n_eligible_traces': n_h13_eligible, 'n_traces_in_mixed_problems': n_h13_traces,
                'n_source_excluded': exclusions['source_excluded_from_rates'],
                'nonleak_minus_leak': _estimate(mixed, lambda x: x[0], resamples, seed)},
    }


def _generation_summary(rows, resamples, seed):
    if rows is None:
        return {'status': 'missing', 'n_problems': 0, 'n_responses': 0, 'cells': []}
    problems = sorted({r['selection_rank'] for r in rows})
    index = {p: i for i, p in enumerate(problems)}
    # For each cell retain per-problem numerator counts, valid denominator,
    # all-request diagnostic counts, and all-request denominator. Common draws
    # preserve pairing even when exclusions differ between cells/templates.
    names = (*METRICS, 'valid', *(f'all_{m}' for m in METRICS),
             *DEGENERATION, 'truncated', 'tutor_mention', 'nonempty_reasoning', 'total')
    positions = {name: i for i, name in enumerate(names)}
    values = np.zeros((len(problems), len(INTERVENTIONS), len(names)))
    grouped = {arm: [] for arm in INTERVENTIONS}
    for row in rows:
        arm = INTERVENTIONS.index(row['intervention'])
        grouped[row['intervention']].append(row)
        valid = not row['excluded_from_rates']
        record = {m: int(valid and row[m]) for m in METRICS}
        record.update(valid=int(valid), **{f'all_{m}': int(row[m]) for m in METRICS},
                      **{m: int(row[m]) for m in (*DEGENERATION, 'truncated', 'tutor_mention')},
                      nonempty_reasoning=int(bool(row['reasoning'].strip())), total=1)
        values[index[row['selection_rank']], arm] += [record[name] for name in names]
    flat = values.reshape(len(problems), len(INTERVENTIONS)*len(names))
    if len(problems):
        draws = np.random.default_rng(seed).integers(0, len(problems), size=(resamples, len(problems)))
        means = np.vstack([flat.mean(axis=0), _cluster_means(flat, draws)])
        means = means.reshape(resamples+1, len(INTERVENTIONS), len(names))
    else:
        means = np.zeros((resamples+1, len(INTERVENTIONS), len(names)))

    def rates(name, denominator='total'):
        return _ratio(means[:, :, positions[name]], means[:, :, positions[denominator]])

    def interval(vector):
        return _interval(vector[0], vector[1:])

    def rate(name, arm, numerator=None, denominator='total'):
        count = sum(int(r[numerator or name]) for r in grouped[INTERVENTIONS[arm]])
        if denominator == 'valid':
            count = sum(int(not r['excluded_from_rates'] and r[name]) for r in grouped[INTERVENTIONS[arm]])
        total = sum(not r['excluded_from_rates'] for r in grouped[INTERVENTIONS[arm]]) if denominator == 'valid' else len(grouped[INTERVENTIONS[arm]])
        return interval(rates(name, denominator)[:, arm]) | {'n_positive': count, 'n_denominator': total}

    cells = []
    baseline_count = sum(r['degenerate'] for r in grouped['baseline'])
    baseline_total = len(grouped['baseline'])
    for a, arm in enumerate(INTERVENTIONS):
        group = grouped[arm]
        total, count = len(group), sum(r['degenerate'] for r in group)
        # Exact integer comparison avoids .30 - .20 spuriously exceeding .10.
        flag = (10*(count*baseline_total-baseline_count*total) > total*baseline_total
                if total and baseline_total else None)
        cells.append({
            'intervention': arm, 'pushed': arm != 'baseline', 'n_responses': total,
            'n_valid': sum(not r['excluded_from_rates'] for r in group),
            'n_excluded': sum(r['excluded_from_rates'] for r in group),
            'n_truncated': sum(r['truncated'] for r in group),
            'n_output_tokens': sum(r['output_tokens'] for r in group),
            'n_final_reply_tokens': sum(r['final_reply_tokens'] for r in group),
            **{f'{stat}_{field}': float(function([r[field] for r in group])) if group else None
               for field in ('output_tokens', 'final_reply_tokens')
               for stat, function in (('mean', np.mean), ('median', np.median))},
            **{m: rate(m, a, denominator='valid') for m in METRICS},
            'all_request_rates': {m: rate(f'all_{m}', a, numerator=m) for m in METRICS},
            'degeneration': {m: rate(m, a) for m in DEGENERATION},
            'degeneration_excess': {m: interval(rates(m)[:, a]-rates(m)[:, 0]) for m in DEGENERATION},
            'truncated': rate('truncated', a), 'tutor_mention': rate('tutor_mention', a),
            'n_nonempty_reasoning': sum(bool(r['reasoning'].strip()) for r in group),
            'tutor_mention_nonempty': interval(rates('tutor_mention', 'nonempty_reasoning')[:, a]),
            'not_interpretable': bool(flag) if flag is not None else None,
        })
    def contrast(statistic, used):
        flagged = [arm for arm in used if cells[INTERVENTIONS.index(arm)]['not_interpretable']]
        def paired_rates(x):
            counts = x.reshape(len(INTERVENTIONS), len(names))
            return _ratio(counts[:, positions['leak']], counts[:, positions['valid']])
        result = _estimate(flat, lambda x: statistic(paired_rates(x)), resamples, seed)
        return result | {'n_problems': len(problems),
                                   'not_interpretable': bool(flagged) if problems else None,
                                   'degenerate_cells': flagged}

    return {'status': 'complete', 'n_problems': len(problems), 'n_responses': len(rows), 'cells': cells,
            'baseline_minus_tutor': contrast(lambda x: x[0]-x[1], ['tutor']),
            # Difference before averaging avoids roundoff-only nonzero CIs
            # when all five control rates equal the paired tutor rate.
            'mean_controls_minus_tutor': contrast(lambda x: np.mean(x[2:]-x[1]), list(SETS))}


def analyze(tracking_rows, generation_rows, expected_templates=2, resamples=2000, seed=0):
    """Validate before filtering; None generation_rows explicitly permits H14 missing.

    A supplied empty/partial generation file is rejected. Development generations
    may be present but never enter test estimands. All rates/contrasts are
    fractions and intervals are unadjusted 95% percentile bootstrap intervals.
    Optional generation IDs are checked for uniqueness when supplied.
    """
    for name, value, valid in (('expected_templates', expected_templates, expected_templates in (1, 2)),
                               ('resamples', resamples, resamples > 0 if _integer(resamples) else False),
                               ('seed', seed, seed >= 0 if _integer(seed) else False)):
        if not _integer(value) or not valid:
            raise ValueError(f'invalid {name}')
    tracks = _records(tracking_rows, 'tracking')
    _validate_tracking(tracks)
    generations = None if generation_rows is None else _records(generation_rows, 'generations')
    if generations is not None:
        _validate_generations(generations, expected_templates)
        _mapping(tracks+generations)
    test = None if generations is None else [r for r in generations if r['split'] == 'test']

    def dataset(t, g):
        return {'n_problems': len({r['problem_idx'] for r in t}), 'n_tracking_rows': len(t),
                'n_generation_rows': 0 if g is None else len(g),
                **_tracking_summary(t, resamples, seed), 'h14': _generation_summary(g, resamples, seed)}

    return dataset(tracks, test) | {
        'phase': 'phase3', 'expected_templates': int(expected_templates),
        'bootstrap': {'resamples': int(resamples), 'seed': int(seed), 'unit': 'problem_idx',
                      'interval': 'percentile 95%', 'paired': True},
        'n_dev_responses_excluded': 0 if generations is None else len(generations)-len(test),
        'selection_performed': False,
        'rate_denominator': 'nonexcluded final replies; ratio of sums recomputed in every draw',
        'diagnostic_denominator': 'all requests, including excluded and truncated outputs',
        'h12_weighting': 'mean over eligible traces; ratio of slope sums to trace counts',
        'h13_weighting': 'equal-weight mean within mixed-outcome problems with eligible final labels',
        'control_pooling': 'equal-weight mean of five control cells after recomputing each cell rate',
        'reasoning_regex': TUTOR_REGEX, 'reasoning_regex_flags': 'IGNORECASE',
        'degeneration_threshold_excess': .10,
        'gold_ge_10': dataset([r for r in tracks if r['gold'] >= 10],
                              None if test is None else [r for r in test if r['gold'] >= 10]),
    }


def trajectory_table(tracking_rows):
    """CSV-ready trace-weighted decile means; null bins contribute no observation.

    Deciles are one-based. Outcome-stratified curves exclude source empty-final
    labels; the all-outcome curve retains these traces. Each row includes both
    a nullable leak indicator and an explicit all/leak/nonleak outcome label.
    """
    rows = _records(tracking_rows, 'tracking')
    _validate_tracking(rows)
    records = []
    for d in range(10):
        for s, name in enumerate(SETS):
            for outcome, leak in (('all', None), ('leak', True), ('nonleak', False)):
                selected = [r for r in rows if r['deciles'] is not None and r['deciles'][d] is not None
                            and (leak is None or (r['leak'] == leak and not r.get('excluded_from_rates', False)))]
                values = [r['deciles'][d][s] for r in selected]
                records.append({'decile': d+1, 'set': name, 'leak': leak, 'outcome': outcome,
                                'mean': float(np.mean(values)) if values else None,
                                'n_traces': len(selected),
                                'n_problems': len({r['problem_idx'] for r in selected})})
    return records


def render_summary(report):
    """Markdown of counts, numerical estimates, intervals, and flags only."""
    def interval(value, scale=1):
        def number(x):
            return 'undefined' if x is None else f'{scale*x:.6f}'
        return f"{number(value['estimate'])} [{number(value['ci95'][0])}, {number(value['ci95'][1])}]"

    def flag(value):
        return 'undefined' if value is None else str(value).lower()

    lines = ['# Phase 3 results', '',
             f"Problem-cluster bootstrap: {report['bootstrap']['resamples']} resamples; "
             f"95% percentile intervals; seed {report['bootstrap']['seed']}.",
             f"Test ranks: 101..200; tracking templates: 4; generation templates: {report['expected_templates']}.",
             f"Development responses excluded: {report['n_dev_responses_excluded']}.",
             'H12–H13: signal units. H14 rates: percent; contrasts: percentage points.',
             'H14 rate denominator: nonexcluded final replies. Diagnostic denominator: all requests.',
             f"Reasoning regex (IGNORECASE): `{report['reasoning_regex']}`.",
             'Degeneration excess threshold: > 10 percentage points.']
    if 'execution_verification' in report:
        execution = report['execution_verification']
        lines += ['', f"Execution completed: {flag(execution['completed'])}; H14 missing: {flag(execution['h14_missing'])}.",
                  f"Verified source labels: {execution['n_source_labels_verified']}; "
                  f"local source files: {execution['n_source_files_verified']}; "
                  f"raw trajectories: {execution['n_trajectories_verified']}."]
    if report.get('selection'):
        selected=report['selection']
        lines += ['', f"c*: {selected.get('c_star')}; dose_threshold_met: {flag(selected.get('dose_threshold_met'))}; "
                  f"timing_template_reduction: {flag(selected.get('timing_template_reduction'))}."]
    if report.get('neurons', {}).get('tutor'):
        lines += ['', '| Tutor neuron (L.N) | Cohen d |', '|---|---|']
        lines += [f"| {n['id']} | {n['d']:.6f} |" for n in report['neurons']['tutor']]
    for label, data in (('All test problems', report), ('gold >= 10', report['gold_ge_10'])):
        lines += ['', f'## {label}', '', f"Problems: {data['n_problems']}; tracking rows: {data['n_tracking_rows']}; "
                  f"generation rows: {data['n_generation_rows']}.", '',
                  '| Tracking exclusion | Count |', '|---|---|']
        lines += [f'| {k} | {v} |' for k, v in data['tracking_exclusions'].items()]
        lines += ['', '| Hypothesis | Contrast | Problems | Traces | Estimate [95% CI] | Undefined resamples |',
                  '|---|---|---|---|---|---|']
        for name in ('tutor_slope', 'tutor_minus_mean_controls'):
            h, value = data['h12'], data['h12'][name]
            lines.append(f"| H12 | {name} | {h['n_problems']} | {h['n_traces']} | {interval(value)} | {value['undefined_resamples']} |")
        h, value = data['h13'], data['h13']['nonleak_minus_leak']
        lines.append(f"| H13 | nonleak_minus_leak | {h['n_mixed_problems']} | {h['n_traces_in_mixed_problems']} | {interval(value)} | {value['undefined_resamples']} |")
        lines.append(f"\nH13 eligible traces: {h['n_eligible_traces']}; source exclusions: {h['n_source_excluded']}; "
                     f"nonmixed or ineligible problems: {h['n_nonmixed_or_ineligible_problems']}.")
        h14 = data['h14']
        lines += ['', f"H14: {h14['status']}"]
        if h14['status'] == 'missing':
            continue
        lines += ['', '| Intervention | Valid / total | Excluded | truncated n | output_tokens | final_reply_tokens |', '|---|---|---|---|---|---|']
        for c in h14['cells']:
            lines.append(f"| {c['intervention']} | {c['n_valid']} / {c['n_responses']} | {c['n_excluded']} | "
                         f"{c['n_truncated']} | {c['n_output_tokens']} | {c['n_final_reply_tokens']} |")
        lines += ['', 'Response lengths include all requests, including empty finals.', '',
                  '| Intervention | Mean output tokens | Median output tokens | Mean final tokens | Median final tokens |',
                  '|---|---|---|---|---|']
        for c in h14['cells']:
            lengths = [c[f'{stat}_{field}'] for field in ('output_tokens', 'final_reply_tokens')
                       for stat in ('mean', 'median')]
            lines.append(f"| {c['intervention']} | " + ' | '.join('undefined' if v is None else f'{v:.6f}' for v in lengths) + ' |')
        lines += ['', '| Intervention | Metric | Positive / denominator | Rate [95% CI], % | Undefined resamples |', '|---|---|---|---|---|']
        for c in h14['cells']:
            values = {m: c[m] for m in METRICS}
            values.update({f'all_request_{m}': c['all_request_rates'][m] for m in METRICS})
            values.update(truncated=c['truncated'], tutor_mention=c['tutor_mention'])
            for name, v in values.items():
                lines.append(f"| {c['intervention']} | {name} | {v['n_positive']} / {v['n_denominator']} | "
                             f"{interval(v, 100)} | {v['undefined_resamples']} |")
        lines += ['', '| H14 contrast | Estimate [95% CI], pp | Undefined resamples | not_interpretable |', '|---|---|---|---|']
        for name in ('baseline_minus_tutor', 'mean_controls_minus_tutor'):
            v = h14[name]
            lines.append(f"| {name} | {interval(v, 100)} | {v['undefined_resamples']} | {flag(v['not_interpretable'])} |")
        lines += ['', '| Intervention | Check | Positive / requests | Rate [95% CI], % | Excess over baseline [95% CI], pp | not_interpretable |', '|---|---|---|---|---|---|']
        for c in h14['cells']:
            for name, v in c['degeneration'].items():
                lines.append(f"| {c['intervention']} | {name} | {v['n_positive']} / {v['n_denominator']} | "
                             f"{interval(v, 100)} | {interval(c['degeneration_excess'][name], 100)} | {flag(c['not_interpretable'])} |")
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Raw Phase 3 directory')
    parser.add_argument('--output-dir', default='results/phase3', type=Path)
    parser.add_argument('--expected-templates', type=int, choices=(1, 2), default=None)
    parser.add_argument('--resamples', type=int, choices=(2000,), default=2000)
    parser.add_argument('--seed', type=int, choices=(0,), default=0)
    parser.add_argument('--allow-tracking-only', action='store_true',
                        help='Permit missing generations only after a recorded budget stop')
    args = parser.parse_args()
    try:
        metadata = {}
        from src.build_prompts import ROOT
        from src.phase3_validation import validate_execution
        for name in ('selection', 'metrics', 'manifest', 'neurons', 'execution'):
            source = args.input/f'{name}.json'
            if not source.exists():
                source = args.output_dir/f'{name}.json'
            if source.exists():
                value = json.loads(source.read_text())
                if not isinstance(value, dict):
                    raise ValueError(f'{name}.json must contain an object')
                metadata[name] = value
        missing = {'metrics', 'manifest', 'neurons'}-metadata.keys()
        if missing:
            raise ValueError(f'missing execution metadata: {sorted(missing)}')
        if not args.allow_tracking_only and 'selection' not in metadata:
            raise ValueError('missing execution metadata: selection')
        metadata.setdefault('selection', {})
        selected_templates = metadata.get('selection', {}).get('test_templates', 2)
        if args.expected_templates is not None and 'test_templates' in metadata.get('selection', {}) and args.expected_templates != selected_templates:
            raise ValueError('expected-templates disagrees with selection.json')
        templates = args.expected_templates if args.expected_templates is not None else selected_templates
        tracks = list(read_rows(args.input/'tracking.jsonl'))
        source = args.input/'generations.jsonl'
        generations = list(read_rows(source)) if source.exists() else None
        if args.allow_tracking_only and generations == []:
            generations = None
        archive = args.input/'trajectories.npz'
        verification = validate_execution(tracks, generations, metadata['neurons'], metadata['selection'],
                                          metadata['manifest'], metadata['metrics'], source_root=ROOT,
                                          trajectories=archive if archive.exists() else None,
                                          allow_tracking_only=args.allow_tracking_only)
        report = analyze(tracks, generations, expected_templates=templates,
                         resamples=args.resamples, seed=args.seed)
        report.update(metadata)
        report['execution_verification'] = verification
        trajectories = trajectory_table(tracks)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        (args.output_dir/'summary.md').write_text(render_summary(report))
        with (args.output_dir/'trajectories.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trajectories[0]))
            writer.writeheader()
            writer.writerows(trajectories)
    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))
    print(f"Phase 3: {report['n_tracking_rows']} tracking rows; H14 {report['h14']['status']} -> {args.output_dir}")


if __name__ == '__main__':
    main()
