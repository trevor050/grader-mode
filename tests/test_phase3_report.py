"""Focused CPU report tests; only source-package validation is stubbed."""
from copy import deepcopy
import json

import pytest

from test_analyze_3 import tracking
from test_phase3_resume import frozen
from src import phase3_report as api
from src.phase3 import score_generation
from src.phase3_resume import file_hash, validate_eligibility


@pytest.fixture
def source(frozen, tmp_path, monkeypatch):
    case = deepcopy(frozen)
    case['tracking_rows'] = tracking()
    for row in case['tracking_rows']:
        row['excluded_from_rates'] = False
    case['tracking_rows'][0]['excluded_from_rates'] = True
    # Complete fallback-eight grid, including an empty final in every dev cell.
    baseline = [r for r in case['generation_rows'] if r['generation_cell'] == 'baseline']
    rows, batches = [], []
    for cell, dose in [('baseline', 0), ('dose_1', 1), ('dose_2', 2), ('dose_4', 4), ('dose_8', 8)]:
        for original in baseline:
            row = deepcopy(original)
            row.update(c=dose, generation_cell=cell, intervention='baseline' if dose == 0 else 'tutor',
                       id=f'{row["phase1b_id"]}:{cell}:c{dose}')
            if row['selection_rank'] == 1:
                row.update(response='<think>unfinished', finish_reason='length', output_tokens=3072,
                           final_reply_tokens=0, end_token_id=None)
                row.update(score_generation(row['response'], row['gold'], 'length'))
            rows.append(row)
        batch = deepcopy(case['metrics']['batches'][0])
        batch.update(cell=cell, c=dose, decode_steps=3072)
        batches.append(batch)
    case['generation_rows'] = rows
    case['metrics'].update(output_rows=500, batches=batches)
    case['selection'].update(c_star=8, dose_threshold_met=False, dev_baseline_leak=1.,
        dose_trials=[dict(c=c, n=100, n_valid=99, leak=1., reduction=0.) for c in (1, 2, 4, 8)])
    raw = tmp_path/'results/raw/source'
    raw.mkdir(parents=True)
    out = tmp_path/'results/phase3'
    out.mkdir(parents=True)
    attempts = deepcopy(case['attempts'])+[dict(run_id='later-active', status='running', source_run_id='source')]
    (out/'gpu-attempts.json').write_text(json.dumps(attempts))
    hashes = {}
    for name, data in [('tracking.jsonl', case['tracking_rows']), ('generations.jsonl', rows)]:
        path = raw/name
        path.write_text(''.join(json.dumps(row)+'\n' for row in data))
        hashes[str(path.relative_to(tmp_path))] = file_hash(path)
    calls = []
    def load(root, source_id, prefix):
        calls.append(prefix)
        eligibility = validate_eligibility(**{**case, 'attempts': prefix})
        return dict(source_metrics=case['metrics'], source_selection=case['selection'],
                    eligibility=eligibility, neurons=case['neurons'], source_manifest=case['manifest'],
                    source_artifacts_sha256=hashes, payload_sha256='synthetic-boundary')
    monkeypatch.setattr(api, 'load_source_package', load)
    return tmp_path, case, calls


def test_actual_calculations_keep_dev_out_of_h14(source):
    root, case, calls = source
    result = api.report('source', root/'report', root=root)
    expected = api.analyze(case['tracking_rows'], None)
    assert result['h12'] == expected['h12'] and result['h13'] == expected['h13']
    assert result['h14']['status'] == result['gold_ge_10']['h14']['status'] == 'missing'
    assert result['n_generation_rows'] == 0 and result['n_dev_responses_excluded'] == 500
    assert result['bootstrap']['resamples'] == 2000 and result['bootstrap']['seed'] == 0
    assert len(calls[0]) == 2 and calls[0][-1]['run_id'] == 'source'
    assert result['accounting']['later_allocations_excluded'] == 1
    assert result['accounting']['latest_zero_containers_verified'] is False
    assert result['source_validation']['eligibility']['n_test_rows'] == 0
    cell = result['development']['cells'][0]
    assert (cell['n_valid'], cell['n_excluded']) == (99, 1)
    assert cell['rates']['leak'] == dict(n_positive=99, n_denominator=99, rate=1.)
    assert cell['all_request_rates']['leak']['rate'] == .99
    assert cell['diagnostics']['degenerate']['rate'] == .01
    assert cell['lengths']['output_tokens']['total'] == 3072+99*30
    assert cell['lengths']['final_reply_tokens']['median'] == 5
    assert len((root/'report/trajectories.csv').read_text().splitlines()) == 181
    assert json.loads((root/'report/summary.json').read_text()) == result
    summary = (root/'report/summary.md').read_text()
    assert 'Status: failed' in summary and 'ValueError: Reduced test grid exceeds GPU budget' in summary
    assert 'H14: missing' in summary and 'Development responses excluded: 500' in summary


@pytest.mark.parametrize('mutation', [
    lambda c: c['metrics'].update(stop_reason='RuntimeError: CUDA error'),
    lambda c: c['metrics'].update(status='budget_stopped'),
    lambda c: c['generation_rows'].pop(),
    lambda c: c['generation_rows'][0].update(split='test'),
    lambda c: c['attempts'][-1].pop('zero_containers_verified_unix'),
])
def test_rejects_ineligible_original_without_output(source, mutation):
    root, case, _ = source
    mutation(case)
    # Ledger is read from disk, while package inputs are supplied at the boundary.
    (root/'results/phase3/gpu-attempts.json').write_text(json.dumps(case['attempts']))
    with pytest.raises(ValueError):
        api.report('source', root/'report', root=root)
    assert not (root/'report').exists()


@pytest.mark.parametrize('occupied', ['empty', 'file', 'symlink'])
def test_rejects_occupied_outputs_before_validation(source, occupied):
    root, _, calls = source
    output = root/'report'
    if occupied == 'empty':
        output.mkdir()
    elif occupied == 'file':
        output.write_text('retain')
    else:
        output.symlink_to(root/'missing')
    with pytest.raises(ValueError, match='fresh'):
        api.report('source', output, root=root)
    assert calls == []


def test_cli_requires_source_and_fresh_output(monkeypatch):
    monkeypatch.setattr('sys.argv', ['phase3_report'])
    with pytest.raises(SystemExit) as error:
        api.main()
    assert error.value.code == 2
