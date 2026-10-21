"""CPU-only original Phase 3 partial-study report; no continuation/pilot data.

Usage: uv run python -m src.phase3_report --source-run-id phase3-20261020-v2
       --output-dir /fresh/path
Accounting certifies only the closed ledger prefix ending at the source run.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median

from src.analyze_3 import DEGENERATION, METRICS, analyze, render_summary, trajectory_table
from src.phase3_resume import load_source_package, require, safe_run_id, verify_hashes
from src.score import read_rows


def _development(rows):
    cells = []
    for cell in ('baseline', 'dose_1', 'dose_2', 'dose_4', 'dose_8'):
        group = [r for r in rows if r['generation_cell'] == cell]
        require(len(group) == 100, 'original development requires 100 rows in each of five cells')
        valid = [r for r in group if not r['excluded_from_rates']]
        def rates(selected, fields):
            return {field: dict(n_positive=sum(r[field] for r in selected),
                                n_denominator=len(selected),
                                rate=sum(r[field] for r in selected)/len(selected) if selected else None)
                    for field in fields}
        cells.append(dict(cell=cell, c=group[0]['c'], n_responses=len(group), n_valid=len(valid),
            n_excluded=len(group)-len(valid), rates=rates(valid, METRICS),
            all_request_rates=rates(group, METRICS),
            diagnostics=rates(group, (*DEGENERATION, 'truncated', 'tutor_mention')),
            lengths={field: dict(total=sum(r[field] for r in group),
                                mean=mean(r[field] for r in group), median=median(r[field] for r in group))
                     for field in ('output_tokens', 'final_reply_tokens')}))
    return dict(split='dev', n_responses=len(rows), cells=cells, included_in_h14=False)


def _markdown(report):
    lines = [render_summary(report).rstrip(), '', '## Original execution', '',
             f"Status: {report['source_status']}", f"Stop reason: {report['stop_reason']}",
             'Accounting scope: closed ledger prefix through source only.',
             'Latest lifetime accounting verified: false; latest zero containers verified: false.',
             '', '## Development generations', '',
             '| Cell | Valid / total | Excluded |', '|---|---|---|']
    dev = report['development']
    lines += [f"| {c['cell']} | {c['n_valid']} / {c['n_responses']} | {c['n_excluded']} |" for c in dev['cells']]
    lines += ['', '| Cell | Population | Metric | Positive / denominator | Rate |', '|---|---|---|---|---|']
    for c in dev['cells']:
        for population in ('rates', 'all_request_rates', 'diagnostics'):
            for metric, value in c[population].items():
                lines.append(f"| {c['cell']} | {population} | {metric} | {value['n_positive']} / {value['n_denominator']} | {value['rate']} |")
    lines += ['', '| Cell | Token field | Total | Mean | Median |', '|---|---|---|---|---|']
    for c in dev['cells']:
        for field, value in c['lengths'].items():
            lines.append(f"| {c['cell']} | {field} | {value['total']} | {value['mean']} | {value['median']} |")
    return '\n'.join(lines)+'\n'


def report(source_run_id, output_dir, *, root=None):
    """Validate source lineage and write three derived files to a new directory."""
    from src.build_prompts import ROOT
    root = Path(ROOT if root is None else root).resolve()
    source_run_id = safe_run_id(source_run_id)
    output_dir = Path(output_dir)
    require(not output_dir.exists() and not output_dir.is_symlink(), 'output directory must be fresh')
    attempts = json.loads((root/'results/phase3/gpu-attempts.json').read_text())
    require(isinstance(attempts, list) and all(isinstance(a, dict) for a in attempts), 'ledger array required')
    indices = [i for i, a in enumerate(attempts) if a.get('run_id') == source_run_id]
    require(len(indices) == 1, 'source allocation absent/ambiguous in lifetime ledger')
    prefix = attempts[:indices[0]+1]
    package = load_source_package(root, source_run_id, prefix)
    metrics, selection = package['source_metrics'], package['source_selection']
    require(metrics['status'] == 'failed' and prefix[-1]['status'] == 'failed'
            and metrics.get('stop_reason') == 'ValueError: Reduced test grid exceeds GPU budget',
            'original failed pretest budget gate required')
    require(selection['c_star'] == 8 and selection['dose_threshold_met'] is False
            and package['eligibility']['n_dev_rows'] == 500, 'complete original fallback-eight development required')
    hashes = package['source_artifacts_sha256']
    def rows(name):
        paths = [name_ for name_ in hashes if Path(name_).name == name]
        require(len(paths) == 1, f'exactly one validated {name} required')
        return list(read_rows(root/paths[0]))
    tracks, dev = rows('tracking.jsonl'), rows('generations.jsonl')
    verify_hashes(root, hashes)
    require(len(dev) == 500 and all(r['split'] == 'dev' for r in dev), 'zero test rows and 500 dev rows required')
    result = analyze(tracks, None, resamples=2000, seed=0)
    result.update(n_dev_responses_excluded=len(dev), selection_performed=True,
        selection=selection, neurons=package['neurons'], development=_development(dev),
        source_run_id=source_run_id, source_status=metrics['status'], stop_reason=metrics['stop_reason'],
        source_validation=dict(eligibility=package['eligibility'],
            source_sha256=package['source_manifest']['source_sha256'],
            source_artifacts_sha256=hashes, payload_sha256=package['payload_sha256'],
            n_source_labels_verified=len(tracks), n_trajectories_verified=len(tracks)),
        accounting=dict(scope='closed ledger prefix through source allocation only',
            prefix=prefix, prefix_seconds=package['eligibility']['previous_seconds'],
            later_allocations_excluded=len(attempts)-len(prefix),
            latest_lifetime_accounting_verified=False, latest_zero_containers_verified=False))
    table = trajectory_table(tracks)
    require(len(table) == 180, 'expected 180 decile rows')
    encoded, markdown = json.dumps(result, indent=2, allow_nan=False)+'\n', _markdown(result)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir/'summary.json').write_text(encoded)
    (output_dir/'summary.md').write_text(markdown)
    with (output_dir/'trajectories.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run-id', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    try:
        report(args.source_run_id, args.output_dir)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f'phase3_report: {error}\n')


if __name__ == '__main__':
    main()
