"""Phase 1c paired problem-cluster rates and risk contrasts, without interpretation.

Input is scored Phase 1c rows (use src.score.score_row before analyze). Complete
52-row grids are required for each supplied checkpoint/problem; entire absent
checkpoints are allowed for smoke runs and explicitly marked unavailable.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analyze import METRICS, _cluster_means, _interval, handcheck
from src.score import read_rows

CHECKPOINTS = ('SFT', 'DPO', 'RLVR')
LADDERS = ('instruct', 'think')
PROSE = ('plain', 'env')
FORMATS = ('none', 'boxed', 'dapo', 'answer_tag', 'result_tag', 'boxed_step')
RL_FORMATS = ('boxed', 'dapo', 'answer_tag')
MODELS = {
    ladder: {stage: f'allenai/Olmo-3-7B-{ladder.title()}{suffix}'
             for stage, suffix in zip(CHECKPOINTS, ('-SFT', '-DPO', ''))}
    for ladder in LADDERS
}
CELLS = tuple(('tutor', prose, fmt) for prose, fmt in product(PROSE, FORMATS)) + (('ceiling', 'plain', 'none'),)


def validate_grid(df: pd.DataFrame) -> None:
    """Validate identities, scoring metadata, and complete grids before filtering."""
    required = {'model', 'id', 'problem_idx', 'gold', 'phase', 'ladder', 'checkpoint',
                'thinking', 'cue', 'prose', 'format', 'arm', 'template_idx', 'response',
                'reasoning', 'final_reply', 'truncated', 'excluded_from_rates', *METRICS}
    if required - set(df):
        raise ValueError(f'missing scored fields: {sorted(required - set(df))}')
    if df.empty or df[list(required)].isna().any().any():
        raise ValueError('empty input or null scored fields')
    if df.duplicated(['model', 'id']).any():
        raise ValueError('duplicate IDs per model')
    if not df.phase.eq('phase1c').all():
        raise ValueError('analysis requires phase1c main-run rows only; no calibration data')
    for name in ('calibration', 'is_calibration'):
        if name in df and df[name].map(lambda x: x is not None and not pd.isna(x) and x != False).any():
            raise ValueError('calibration data must not enter Phase 1c analysis')
    if not df.ladder.isin(LADDERS).all():
        raise ValueError('unknown ladder')
    if not df.checkpoint.isin(CHECKPOINTS).all():
        raise ValueError('unknown checkpoint')
    for name in ('thinking', 'truncated', 'excluded_from_rates'):
        if not df[name].map(lambda x: isinstance(x, (bool, np.bool_))).all():
            raise ValueError(f'{name} must be boolean')
    if not df.thinking.eq(df.ladder.eq('think')).all():
        raise ValueError('thinking must be true exactly for the think ladder')
    for name in ('problem_idx', 'gold', 'template_idx'):
        if not df[name].map(lambda x: isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))).all():
            raise ValueError(f'{name} must be integer')
    if df.problem_idx.lt(0).any():
        raise ValueError('problem_idx must retain original nonnegative indices')
    for name in ('model', 'id', 'response', 'reasoning', 'final_reply'):
        if not df[name].map(lambda x: isinstance(x, str)).all():
            raise ValueError(f'{name} must be text')
    if df.id.str.strip().eq('').any():
        raise ValueError('id must be nonempty')
    if not df.apply(lambda r: r.model == MODELS[r.ladder][r.checkpoint], axis=1).all():
        raise ValueError('model HF ID must match the registered ladder/checkpoint')
    if not df.cue.eq(df.prose).all() or not df.prose.isin(PROSE).all():
        raise ValueError('cue must equal prose, both plain or env')
    for metric in METRICS:
        if not df[metric].isin([0, 1, False, True]).all():
            raise ValueError(f'{metric} must be binary')
    if ((df.correct_commit > df.leak) | (df.correct_commit > df.commit)).any():
        raise ValueError('correct_commit requires leak and commit')
    expected_exclusion = df.thinking & df.final_reply.str.strip().eq('')
    if not df.excluded_from_rates.eq(expected_exclusion).all():
        raise ValueError('excluded_from_rates must identify empty thinking-on final replies')
    expected = {(*cell, t) for cell in CELLS for t in range(4)}
    for _, ladder_rows in df.groupby('ladder'):
        problem_sets = []
        for _, stage_rows in ladder_rows.groupby('checkpoint'):
            problem_sets.append(set(stage_rows.problem_idx))
            for _, rows in stage_rows.groupby('problem_idx'):
                actual = list(rows[['arm', 'prose', 'format', 'template_idx']].itertuples(index=False, name=None))
                if len(actual) != 52 or set(actual) != expected:
                    raise ValueError('incomplete/unbalanced Phase 1c grid: expected 52 unique rows/problem/checkpoint')
        if any(s != problem_sets[0] for s in problem_sets):
            raise ValueError('checkpoints have unbalanced problem sets within a ladder')
    if df.groupby('problem_idx').gold.nunique().ne(1).any():
        raise ValueError('inconsistent gold across problems/templates/checkpoints/ladders')


def _contrasts(rates: dict, resamples: int) -> dict:
    """All estimands use leak; pooled effects give each RL format equal weight.

    Arrays contain the point estimate followed by paired bootstrap replicates.
    Do not pool valid-response denominators across formats: that would change
    the registered format mixture when empty-final exclusions differ by format.
    """
    def leak(stage, prose, fmt):
        return rates[stage][('tutor', prose, fmt)][:, 0]

    def effect(stage, fmt, prose='plain'):
        if fmt == 'pooled_rl':
            return np.mean([effect(stage, f, prose) for f in RL_FORMATS], axis=0)
        return leak(stage, prose, fmt) - leak(stage, prose, 'none')

    def specificity(stage):
        # The shared none term cancels exactly. Avoid an unnecessary undefined
        # result if that cell has no valid final replies in a particular draw.
        return np.mean([leak(stage, 'plain', f) for f in RL_FORMATS], axis=0) - leak(stage, 'plain', 'result_tag')

    def interval(stages, statistic):
        missing = [s for s in stages if s not in rates]
        if missing:
            return _interval(np.nan, np.full(resamples, np.nan)) | {
                'status': 'unavailable', 'missing_checkpoints': missing}
        values = statistic()
        return _interval(values[0], values[1:]) | {'status': 'ok' if np.isfinite(values[0]) else 'undefined'}

    return {
        'plain_format_effects': {
            s: {f: interval((s,), lambda s=s, f=f: effect(s, f))
                for f in (*FORMATS[1:], 'pooled_rl')} for s in CHECKPOINTS},
        'h6': interval(('DPO', 'RLVR'), lambda: effect('RLVR', 'pooled_rl') - effect('DPO', 'pooled_rl')),
        'h7': {
            'rlvr_specificity': interval(('RLVR',), lambda: specificity('RLVR')),
            'difference_in_differences': interval(('DPO', 'RLVR'), lambda:
                specificity('RLVR') - specificity('DPO')),
        },
        'h8': interval(('RLVR',), lambda: effect('RLVR', 'boxed_step')),
        'per_format_h6': {f: interval(('DPO', 'RLVR'), lambda f=f: effect('RLVR', f) - effect('DPO', f))
                          for f in RL_FORMATS},
        'env_format_interactions': {
            s: {f: interval((s,), lambda s=s, f=f: effect(s, f, 'env') - effect(s, f))
                for f in (*FORMATS[1:], 'pooled_rl')} for s in CHECKPOINTS},
        'sft_dpo_format_changes': {
            f: interval(('SFT', 'DPO'), lambda f=f: effect('DPO', f) - effect('SFT', f))
            for f in (*FORMATS[1:], 'pooled_rl')},
    }


def _summarize(rows: pd.DataFrame, stages: list[str], *, resamples: int, seed: int) -> dict:
    problems = sorted(int(p) for p in rows.problem_idx.unique())
    n = len(problems)
    indices = np.random.default_rng(seed).integers(0, n, size=(resamples, n)) if n else None
    checkpoints, rates = {}, {}
    for stage in stages:
        stage_rows = rows[rows.checkpoint == stage]
        cells, rates[stage] = [], {}
        for arm, prose, fmt in CELLS:
            cell = stage_rows[(stage_rows.arm == arm) & (stage_rows.prose == prose) & (stage_rows['format'] == fmt)]
            valid = ~cell.excluded_from_rates
            # Resample whole problems and recompute the ratio of valid successes
            # to valid responses. Keep empty finals in the structural grid only.
            counts = cell[list(METRICS)].astype(float).mul(valid, axis=0)
            counts['valid'] = valid.astype(float)
            counts['problem_idx'] = cell.problem_idx
            values = counts.groupby('problem_idx').sum().reindex(problems).to_numpy()
            totals = (np.vstack([values.mean(axis=0), _cluster_means(values, indices)])
                      if n else np.zeros((resamples + 1, 4)))
            estimates = np.divide(totals[:, :3], totals[:, 3, None],
                                  out=np.full((resamples + 1, 3), np.nan), where=totals[:, 3, None] != 0)
            rates[stage][(arm, prose, fmt)] = estimates
            cells.append({
                'arm': arm, 'prose': prose, 'cue': prose, 'format': fmt,
                'n_responses': len(cell), 'n_valid': int(valid.sum()), 'n_excluded': int((~valid).sum()),
                'n_truncated': int(cell.truncated.sum()),
                'n_truncated_empty': int((cell.truncated & ~valid).sum()),
                'n_empty_final': int(cell.final_reply.str.strip().eq('').sum()),
                **{m: _interval(estimates[0, i], estimates[1:, i]) for i, m in enumerate(METRICS)},
            })
        checkpoints[stage] = {
            'n_problems': n, 'n_responses': len(stage_rows), 'cells': cells,
            'n_excluded': int(stage_rows.excluded_from_rates.sum()),
            'n_truncated': int(stage_rows.truncated.sum()),
        }
    return {'n_problems': n, 'problem_indices': problems, 'n_responses': len(rows),
            'checkpoints': checkpoints, 'contrasts': _contrasts(rates, resamples)}


def _dataset(selected: pd.DataFrame, source: pd.DataFrame, *, resamples: int, seed: int) -> dict:
    report = {'n_problems': int(selected.problem_idx.nunique()), 'n_responses': len(selected), 'ladders': {}, 'deferred_ladders': [l for l in LADDERS if not source.ladder.eq(l).any()]}
    for ladder in LADDERS:
        original = source[source.ladder == ladder]
        if original.empty:
            continue
        rows = selected[selected.ladder == ladder]
        stages = [s for s in CHECKPOINTS if original.checkpoint.eq(s).any()]
        missing = [s for s in CHECKPOINTS if s not in stages]
        result = _summarize(rows, stages, resamples=resamples, seed=seed)
        expected_n = 200 if ladder == 'instruct' else 100
        result.update(
            available_checkpoints=stages, missing_checkpoints=missing,
            checkpoint_status='partial' if missing else 'complete',
            models={s: MODELS[ladder][s] for s in stages}, thinking=ladder == 'think',
            inference={'eligible': not missing and original.problem_idx.nunique() == expected_n,
                       'required_problems': expected_n, 'source_n_problems': int(original.problem_idx.nunique()),
                       'complete_checkpoints': not missing},
        )
        if missing:
            common = {'status': 'unavailable', 'missing_checkpoints': missing, 'n_problems': None,
                      'problem_indices': [], 'checkpoints': {}, 'contrasts': {}}
        else:
            ceiling = rows[(rows.arm == 'ceiling') & ~rows.excluded_from_rates & rows.correct_commit.astype(bool)]
            solved = [set(ceiling.loc[ceiling.checkpoint == s, 'problem_idx']) for s in CHECKPOINTS]
            shared = set.intersection(*solved)
            common = _summarize(rows[rows.problem_idx.isin(shared)], stages, resamples=resamples, seed=seed)
            common['status'] = 'ok' if shared else 'empty'
        result['common_ceiling_solved'] = common
        report['ladders'][ladder] = result
    return report


def analyze(df: pd.DataFrame, *, resamples: int = 2000, seed: int = 0, min_gold: int | None = None) -> dict:
    """Analyze scored main-run rows; validation precedes every subset operation."""
    validate_grid(df)
    if isinstance(resamples, (bool, np.bool_)) or not isinstance(resamples, (int, np.integer)) or resamples < 1:
        raise ValueError('resamples must be a positive integer')
    if min_gold is not None and (isinstance(min_gold, bool) or not isinstance(min_gold, (int, np.integer))):
        raise ValueError('min_gold must be an integer')
    selected = df if min_gold is None else df[df.gold >= min_gold]
    report = _dataset(selected, df, resamples=resamples, seed=seed)
    metadata = {
        'phase': 'phase1c', 'min_gold': min_gold,
        'bootstrap': {'resamples': int(resamples), 'seed': seed, 'unit': 'problem_idx', 'interval': 'percentile 95%',
                      'pairing': 'common whole-problem draws across checkpoints, cells, templates, and outcomes within each ladder/subset'},
        'rate_denominator': 'nonexcluded final replies; ratio of sums re-estimated in each problem bootstrap',
        'pooled_formats': list(RL_FORMATS), 'pooling': 'equal-weight mean of format-specific effects',
        'common_ceiling_rule': 'at least one nonexcluded correct commit among four ceiling templates in each of SFT, DPO, RLVR; membership fixed before resampling',
        'calibration_data_included': False,
    }
    report.update(metadata)
    if min_gold is None:
        sensitivity = _dataset(df[df.gold >= 10], df, resamples=resamples, seed=seed)
        sensitivity.update(metadata | {'min_gold': 10})
        report['gold_ge_10'] = sensitivity
    return report


def render_summary(report: dict) -> str:
    """Render numerical tables and design/availability metadata, without conclusions."""
    def interval(value):
        def number(x):
            return 'undefined' if x is None else f'{100 * x:.2f}'
        return f"{number(value['estimate'])} [{number(value['ci95'][0])}, {number(value['ci95'][1])}]"

    def flatten(values, prefix=''):
        for key, value in values.items():
            name = f'{prefix}/{key}' if prefix else key
            if 'estimate' in value:
                yield name, value
            else:
                yield from flatten(value, name)

    lines = ['# Phase 1c results', '', f"Gold minimum: {report['min_gold']}; {report['n_responses']} responses.", '',
             f"Problem-cluster bootstrap: {report['bootstrap']['resamples']:,} resamples; 95% percentile intervals; seed {report['bootstrap']['seed']}.", '',
             'Rates are percentages; contrasts are percentage points. Intervals are unadjusted for multiplicity and conditional on defined bootstrap replicates.', '',
             'Empty thinking-on final replies are excluded; nonempty truncated replies remain scored. RL-format pools are equal-weight means of boxed, dapo, answer_tag effects.']
    if 'think' in report.get('deferred_ladders', []):
        lines += ['', 'Think ladder: deferred (DESIGN.md Deviation 6).']
    for ladder, result in report['ladders'].items():
        lines += ['', f'## {ladder}', '',
                  f"Checkpoints: {result['checkpoint_status']}; available: {', '.join(result['available_checkpoints'])}; missing: {', '.join(result['missing_checkpoints']) or 'none'}.",
                  f"Inference eligible: {result['inference']['eligible']}; source problems: {result['inference']['source_n_problems']} / {result['inference']['required_problems']}."]
        for label, data in (('All selected problems', result), ('All-three-checkpoint common ceiling solved problems', result['common_ceiling_solved'])):
            lines += ['', f'### {label}', '', f"Problems: {data['n_problems']}; status: {data.get('status', 'available')}."]
            if data.get('status') == 'unavailable':
                continue
            lines += ['', '| Checkpoint | Arm | Prose | Format | Valid / total | Excluded (truncated) | Truncated | Empty final | Leak [95% CI] | Commit [95% CI] | Correct commit [95% CI] |',
                      '|---|---|---|---|---|---|---|---|---|---|---|']
            for stage, stage_result in data['checkpoints'].items():
                for c in stage_result['cells']:
                    fields = [stage, c['arm'], c['prose'], c['format'], f"{c['n_valid']} / {c['n_responses']}",
                              f"{c['n_excluded']} ({c['n_truncated_empty']})", str(c['n_truncated']), str(c['n_empty_final']),
                              *(interval(c[m]) for m in METRICS)]
                    lines.append('| ' + ' | '.join(fields) + ' |')
            lines += ['', '| Contrast | Estimate [95% CI], pp | Defined / total draws | Status |', '|---|---|---|---|']
            for name, value in flatten(data['contrasts']):
                lines.append(f"| {name.replace('h6', 'H6').replace('h7', 'H7').replace('h8', 'H8')} | {interval(value)} | {value['valid_resamples']} / {value['valid_resamples'] + value['undefined_resamples']} | {value['status']} |")
    if report.get('handcheck'):
        lines += ['', '## Scoring audit', '', '| Model | Sampled | Human leak labels | Agreement | Status |', '|---|---|---|---|---|']
        for model, audit in report['handcheck'].items():
            lines.append(f"| {model} | {audit['n_sampled']} | {audit['n_labeled']} | {audit['leak_agreement']} | {audit['status']} |")
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--resamples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--min-gold', type=int, default=None)
    args = parser.parse_args()
    try:
        df = pd.DataFrame(list(read_rows(args.input)))
        report = analyze(df, resamples=args.resamples, seed=args.seed, min_gold=args.min_gold)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.min_gold is None:
            audit = df[~df.excluded_from_rates].copy()
            audit['response'] = audit.final_reply
            report['handcheck'] = handcheck(audit, args.output_dir, seed=args.seed) if len(audit) else {}
        outputs = [('summary', report)]
        if 'gold_ge_10' in report:
            outputs.append(('summary_gold_ge_10', report['gold_ge_10']))
        for name, data in outputs:
            (args.output_dir / f'{name}.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            (args.output_dir / f'{name}.md').write_text(render_summary(data), encoding='utf-8')
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Analysis -> {args.output_dir}; {report['n_responses']} responses")


if __name__ == '__main__':
    main()
