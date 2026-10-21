"""Separate reduced-grid Phase 1c Think analysis (DESIGN.md Deviations 10–11).

Nothing is read or written at import. Only an explicit CLI invocation reads raw
runs, scores final replies, and writes summary.md/json. Never invoke on real
Think data while outcome analysis is embargoed. See docs/phase1c-think-analysis.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analyze import METRICS, _cluster_means, _interval
from src.analyze_1c import CHECKPOINTS, FORMATS, MODELS, RL_FORMATS, render_summary as render_instruct_summary
from src.score import score_row

THINK_MODELS = MODELS['think']
RUNS = {'A': 'phase1c-think-A', 'B': 'phase1c-think-B'}
TEMPLATES = {'A': (0, 1), 'B': (2, 3)}
TUTOR_STAGES = ('DPO', 'RLVR')
RESAMPLES = 2000
SEED = 0
N_PROBLEMS = 100
CELLS = tuple(('tutor', 'plain', fmt) for fmt in FORMATS) + (('ceiling', 'plain', 'none'),)
OMITTED = {
    'env_format_interactions': 'omitted: env prose cells were not run in the reduced Think grid',
    'sft_dpo_format_changes': 'omitted: SFT tutor cells were not run in the reduced Think grid',
}


def validate_grid(df: pd.DataFrame, *, allow_missing_b: bool = False) -> tuple[int, ...]:
    """Reject malformed raw metadata before scoring or selecting any subset.

    Original problem indices need not be contiguous. This establishes exact
    cardinality and matching metadata, not provenance of the fixed sample;
    upstream generation manifests own that check. Response values are only
    type-checked here, never searched, printed, or included in error messages.
    """
    required = {'model', 'id', 'problem_idx', 'gold', 'question', 'phase', 'ladder',
                'checkpoint', 'thinking', 'cue', 'prose', 'format', 'arm',
                'template_idx', 'response', 'finish_reason', 'run_id'}
    if required - set(df):
        raise ValueError(f'missing raw fields: {sorted(required - set(df))}')
    if df.empty or df[list(required)].isna().any().any():
        raise ValueError('empty input or null required fields')
    if not isinstance(allow_missing_b, bool):
        raise ValueError('allow_missing_b must be boolean')
    for name in ('model', 'id', 'question', 'phase', 'ladder', 'checkpoint', 'cue',
                 'prose', 'format', 'arm', 'response', 'finish_reason', 'run_id'):
        if not df[name].map(lambda x: isinstance(x, str)).all():
            raise ValueError(f'{name} must be text')
    if df.id.str.strip().eq('').any() or df.question.str.strip().eq('').any():
        raise ValueError('ID and question must be nonempty')
    for name in ('problem_idx', 'gold', 'template_idx'):
        if not df[name].map(lambda x: isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))).all():
            raise ValueError(f'{name} must be integer')
    if df.problem_idx.lt(0).any():
        raise ValueError('problem_idx must retain original nonnegative indices')
    if not df.thinking.map(lambda x: isinstance(x, (bool, np.bool_)) and bool(x)).all():
        raise ValueError('thinking must be boolean true for every Think row')
    if not df.phase.eq('phase1c').all() or not df.ladder.eq('think').all():
        raise ValueError('requires phase1c main-run Think rows only')
    for name in ('calibration', 'is_calibration'):
        if name in df and not df[name].map(lambda x: isinstance(x, (bool, np.bool_)) and not bool(x)).all():
            raise ValueError('calibration data/ambiguous calibration flags are forbidden')
    if not df.finish_reason.isin(('stop', 'length')).all():
        raise ValueError('finish_reason must be stop or length')
    if set(df.checkpoint) != set(CHECKPOINTS):
        raise ValueError('all three registered checkpoints are required')
    if not df.apply(lambda r: r.model == THINK_MODELS[r.checkpoint], axis=1).all():
        raise ValueError('model must match the registered Think checkpoint')
    if df.duplicated(['model', 'id']).any():
        raise ValueError('duplicate IDs per model')
    if not df.cue.eq('plain').all() or not df.prose.eq('plain').all():
        raise ValueError('cue and prose must both be plain in the reduced grid')
    if not df['format'].isin(FORMATS).all():
        raise ValueError('unknown format in reduced grid')
    if not df.run_id.isin(RUNS.values()).all():
        raise ValueError('run_id must identify Stage A or Stage B')
    for stage, run in RUNS.items():
        if not df.loc[df.run_id.eq(run), 'template_idx'].isin(TEMPLATES[stage]).all():
            raise ValueError(f'Stage {stage} template indices do not match its run_id')
    available = set(df.run_id)
    if RUNS['A'] not in available:
        raise ValueError('Stage A is required')
    has_b = RUNS['B'] in available
    if not has_b and not allow_missing_b:
        raise ValueError('Stage B missing; use --allow-missing-b only when the entire stage is absent')
    templates = (0, 1, 2, 3) if has_b else (0, 1)
    expected_ids = df.apply(lambda r:
        f'phase1c:think:{r.checkpoint}:p{r.problem_idx:04d}:plain:{r["format"]}:t{r.template_idx}:{r.arm}', axis=1)
    if not df.id.eq(expected_ids).all():
        raise ValueError('ID must exactly encode checkpoint/problem/prose/format/template/arm metadata')
    if df.groupby('problem_idx').gold.nunique().ne(1).any():
        raise ValueError('inconsistent gold for matched problem metadata')
    if df.groupby('problem_idx').question.nunique().ne(1).any():
        raise ValueError('inconsistent question for matched problem metadata')
    problems = set(df.problem_idx)
    if len(problems) != N_PROBLEMS:
        raise ValueError('reduced grid requires exactly 100 matched problems')
    for checkpoint in CHECKPOINTS:
        checkpoint_rows = df[df.checkpoint.eq(checkpoint)]
        if set(checkpoint_rows.problem_idx) != problems:
            raise ValueError('checkpoint problem sets must match the 100-problem grid')
        cells = CELLS if checkpoint in TUTOR_STAGES else (('ceiling', 'plain', 'none'),)
        expected = {(*cell, t) for cell in cells for t in templates}
        for _, group in checkpoint_rows.groupby('problem_idx'):
            actual = list(group[['arm', 'prose', 'format', 'template_idx']].itertuples(index=False, name=None))
            if len(actual) != len(expected) or set(actual) != expected:
                raise ValueError('incomplete/extra reduced Think grid (including partial Stage B or SFT tutor cells)')
    return templates


def load_runs(stage_a: str | Path, stage_b: str | Path, *, allow_missing_b: bool = False) -> pd.DataFrame:
    """Read only the three named generation files per run, never recursive data.

    A missing/empty B directory is allowed only with the flag. Any present B
    checkpoint file means all three files and their entire grids are required.
    File ownership and run metadata must match; no implicit deduplication.
    """
    records = []
    for stage, directory in (('A', Path(stage_a)), ('B', Path(stage_b))):
        expected = {THINK_MODELS[s].split('/')[-1] + '.jsonl': s for s in CHECKPOINTS}
        if directory.exists() and not directory.is_dir():
            raise ValueError(f'Stage {stage} input must be a directory')
        present = {p.name for p in directory.glob('*.jsonl')} if directory.exists() else set()
        # Request inputs are not generations. Other JSONL files are errors,
        # preventing accidental inclusion or hiding of an extra checkpoint.
        if present - set(expected) - {'requests.jsonl'}:
            raise ValueError(f'unexpected generation file in Stage {stage}')
        found = present & set(expected)
        if stage == 'B' and not found and allow_missing_b:
            continue
        if found != set(expected):
            label = 'partial Stage B' if stage == 'B' and found else f'Stage {stage} missing checkpoint files'
            raise ValueError(label)
        for name, checkpoint in expected.items():
            path = directory / name
            count = 0
            with path.open(encoding='utf-8') as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f'Stage {stage} {name}: invalid JSON at line {line_number}') from error
                    if not isinstance(row, dict):
                        raise ValueError(f'Stage {stage} {name}: generation row must be an object')
                    if (row.get('checkpoint') != checkpoint or row.get('model') != THINK_MODELS[checkpoint]
                            or row.get('run_id') != RUNS[stage]):
                        raise ValueError(f'Stage {stage} {name}: file checkpoint/model/run metadata mismatch')
                    records.append(row)
                    count += 1
            if not count:
                raise ValueError(f'Stage {stage} {name}: empty checkpoint file')
    df = pd.DataFrame(records)
    validate_grid(df, allow_missing_b=allow_missing_b)
    return df


def _contrasts(rates: dict) -> dict:
    """Same leak estimands as analyze_1c, restricted to the observed tutor cells."""
    def leak(stage, fmt):
        return rates[stage][('tutor', 'plain', fmt)][:, 0]

    def effect(stage, fmt):
        if fmt == 'pooled_rl':
            return np.mean([effect(stage, f) for f in RL_FORMATS], axis=0)
        return leak(stage, fmt) - leak(stage, 'none')

    def specificity(stage):
        # Cancel the shared none term, exactly as in the Instruct analyzer.
        return np.mean([leak(stage, f) for f in RL_FORMATS], axis=0) - leak(stage, 'result_tag')

    def interval(values):
        return _interval(values[0], values[1:]) | {'status': 'ok' if np.isfinite(values[0]) else 'undefined'}

    return {
        'plain_format_effects': {s: {f: interval(effect(s, f)) for f in (*FORMATS[1:], 'pooled_rl')}
                                 for s in TUTOR_STAGES},
        'h6': interval(effect('RLVR', 'pooled_rl') - effect('DPO', 'pooled_rl')),
        'h7': {'rlvr_specificity': interval(specificity('RLVR')),
               'difference_in_differences': interval(specificity('RLVR') - specificity('DPO'))},
        'h8': interval(effect('RLVR', 'boxed_step')),
        'per_format_h6': {f: interval(effect('RLVR', f) - effect('DPO', f)) for f in RL_FORMATS},
    }


def _summarize(rows: pd.DataFrame) -> dict:
    problems = sorted(int(p) for p in rows.problem_idx.unique())
    n = len(problems)
    indices = np.random.default_rng(SEED).integers(0, n, size=(RESAMPLES, n)) if n else None
    checkpoints, rates = {}, {}
    for stage in CHECKPOINTS:
        stage_rows = rows[rows.checkpoint.eq(stage)]
        cells, rates[stage] = [], {}
        for arm, prose, fmt in (CELLS if stage in TUTOR_STAGES else (('ceiling', 'plain', 'none'),)):
            cell = stage_rows[stage_rows.arm.eq(arm) & stage_rows['format'].eq(fmt)]
            valid = ~cell.excluded_from_rates
            counts = cell[list(METRICS)].astype(float).mul(valid, axis=0)
            counts['valid'] = valid.astype(float)
            counts['problem_idx'] = cell.problem_idx
            values = counts.groupby('problem_idx').sum().reindex(problems).to_numpy()
            totals = (np.vstack([values.mean(axis=0), _cluster_means(values, indices)])
                      if n else np.zeros((RESAMPLES + 1, 4)))
            estimates = np.divide(totals[:, :3], totals[:, 3, None],
                                  out=np.full((RESAMPLES + 1, 3), np.nan), where=totals[:, 3, None] != 0)
            rates[stage][(arm, prose, fmt)] = estimates
            cells.append({
                'arm': arm, 'prose': prose, 'cue': prose, 'format': fmt,
                'n_responses': len(cell), 'n_valid': int(valid.sum()), 'n_excluded': int((~valid).sum()),
                'n_truncated': int(cell.truncated.sum()),
                'n_truncated_empty': int((cell.truncated & ~valid).sum()),
                'n_empty_final': int((~valid).sum()),
                **{m: _interval(estimates[0, i], estimates[1:, i]) for i, m in enumerate(METRICS)},
            })
        checkpoints[stage] = {
            'n_problems': n, 'n_responses': len(stage_rows), 'cells': cells,
            'n_excluded': int(stage_rows.excluded_from_rates.sum()),
            'n_truncated': int(stage_rows.truncated.sum()),
        }
    return {'n_problems': n, 'problem_indices': problems, 'n_responses': len(rows),
            'checkpoints': checkpoints, 'contrasts': _contrasts(rates)}


def _dataset(rows: pd.DataFrame) -> dict:
    result = _summarize(rows)
    ceiling = rows[rows.arm.eq('ceiling') & ~rows.excluded_from_rates & rows.correct_commit]
    solved = [set(ceiling.loc[ceiling.checkpoint.eq(s), 'problem_idx']) for s in CHECKPOINTS]
    shared = set.intersection(*solved)
    common = _summarize(rows[rows.problem_idx.isin(shared)])
    common['status'] = 'ok' if shared else 'empty'
    result.update(
        available_checkpoints=list(CHECKPOINTS), missing_checkpoints=[], checkpoint_status='complete',
        models=THINK_MODELS.copy(), thinking=True, common_ceiling_solved=common,
        inference={'eligible': True, 'required_problems': N_PROBLEMS,
                   'source_n_problems': N_PROBLEMS, 'complete_checkpoints': True},
    )
    return {'n_problems': result['n_problems'], 'n_responses': len(rows),
            'ladders': {'think': result}, 'deferred_ladders': []}


def analyze(df: pd.DataFrame, *, allow_missing_b: bool = False) -> dict:
    """Validate raw grid, recompute final-only scores, then analyze all subsets.

    No output files, audits, traces, or response text are emitted by this API.
    The caller must not call it on real data until the analysis embargo ends.
    """
    templates = validate_grid(df, allow_missing_b=allow_missing_b)
    # Always recompute, ignoring any stale/precomputed score fields in raw input.
    scored = pd.DataFrame([score_row(row) for row in df.to_dict('records')])
    # Analysis needs outcomes/exclusions only, never reasoning or response text.
    scored = scored[['problem_idx', 'gold', 'checkpoint', 'arm', 'format',
                     'truncated', 'excluded_from_rates', *METRICS]]
    metadata = {
        'phase': 'phase1c-think', 'min_gold': None, 'templates': list(templates),
        'stage_b': {'status': 'complete' if len(templates) == 4 else 'missing_allowed',
                    'allow_missing_b': allow_missing_b},
        'bootstrap': {'resamples': RESAMPLES, 'seed': SEED, 'unit': 'problem_idx',
                      'interval': 'percentile 95%',
                      'pairing': 'common whole-problem draws across checkpoints, cells, templates, and outcomes within each subset'},
        'rate_denominator': 'nonexcluded final replies; ratio of sums re-estimated in each problem bootstrap',
        'pooled_formats': list(RL_FORMATS), 'pooling': 'equal-weight mean of format-specific effects',
        'common_ceiling_rule': 'at least one nonexcluded correct commit among available ceiling templates in each of SFT, DPO, RLVR; membership fixed before resampling',
        'omitted_exploratory_analyses': OMITTED.copy(),
        'per_format_h6_status': 'exploratory, as in the Phase 1c preregistration',
        'calibration_data_included': False, 'smoke_data_included': False,
        'system_prompt': 'prompts/system_tutor.txt',
        'scoring': 'existing src.score.score_row; only final reply after the reasoning block; empty finals excluded and counted; nonempty truncated finals scored',
    }
    report = _dataset(scored) | metadata
    report['gold_ge_10'] = _dataset(scored[scored.gold.ge(10)]) | (metadata | {'min_gold': 10})
    return report


def render_summary(report: dict) -> str:
    """Render all and sensitivity into one numerical report, without conclusions."""
    lines = ['# Phase 1c Think results', '',
             f"Available templates: {report['templates']}; Stage B: {report['stage_b']['status']}.",
             'Tutor prompt: prompts/system_tutor.txt (registered calibration decision).',
             'SFT has ceiling rows only; DPO and RLVR have plain tutor rows in all six formats.',
             'Solvability: at least one valid correct commit per checkpoint among the available ceiling templates; membership fixed before bootstrap.',
             'Exploratory env × format and SFT → DPO analyses omitted: env prose and SFT tutor cells were not run.',
             'Per-format H6 breakdowns are exploratory. No hand-check agreement is claimed by this report.', '']
    for label, data in (('All problems', report), ('Sensitivity: gold >= 10', report['gold_ge_10'])):
        lines += [f'## {label}', '']
        # Reuse the existing table formatting without changing Instruct behavior.
        rendered = render_instruct_summary(data).splitlines()[2:]
        lines.extend(('#' + line if line.startswith('#') else line) for line in rendered)
        lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-a', type=Path, default=Path('results/raw/phase1c-think-A'))
    parser.add_argument('--stage-b', type=Path, default=Path('results/raw/phase1c-think-B'))
    parser.add_argument('--allow-missing-b', action='store_true', help='allow entirely absent Stage B, never a partial stage')
    parser.add_argument('--output-dir', type=Path, default=Path('results/phase1c-think'))
    args = parser.parse_args()
    # Protect existing Instruct output and all raw data from a mistyped target.
    target = args.output_dir.resolve()
    protected = [Path('results/phase1c').resolve(), Path('results/raw').resolve(),
                 args.stage_a.resolve(), args.stage_b.resolve()]
    if any(target == p or target.is_relative_to(p) for p in protected):
        parser.error('output-dir must be separate from Instruct results and raw inputs')
    try:
        df = load_runs(args.stage_a, args.stage_b, allow_missing_b=args.allow_missing_b)
        report = analyze(df, allow_missing_b=args.allow_missing_b)
        encoded = json.dumps(report, indent=2, allow_nan=False) + '\n'
        markdown = render_summary(report)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'summary.json').write_text(encoded, encoding='utf-8')
        (args.output_dir / 'summary.md').write_text(markdown, encoding='utf-8')
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f'Analysis written to {args.output_dir}')


if __name__ == '__main__':
    main()
