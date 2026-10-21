"""Paired token, bootstrap, and allocation contracts for the neutral control."""
from copy import deepcopy
import numpy as np
import pytest


def api():
    from src import phase3_neutral
    return phase3_neutral


def test_span_asserts_saved_and_both_replays_token_for_token():
    m = api()
    response = '<think>abc</think>x'
    # Character tokenizer, padding offsets must never enter either span.
    prompt = 'system|user|'
    text = prompt + response
    offsets = [(0, 0)] + [(i, i+1) for i in range(len(text))]
    ids = [99] + list(range(len(text)))
    positions = m.replay_positions(prompt, response, offsets)
    assert [ids[j] for j in positions] == list(range(len(prompt)+7, len(prompt)+10))
    m.assert_tokens([1, 2, 3], [1, 2, 3], [1, 2, 3], 'trace')
    for sequences in (([1, 2], [1, 3], [1, 2]), ([1], [1], [1, 2])):
        with pytest.raises(AssertionError, match='trace'):
            m.assert_tokens(*sequences, 'trace')
    assert m.replay_positions(prompt, '<think>unclosed', offsets) == []


def test_pilot_includes_provisioning_prior_failures_and_shutdown():
    m = api()
    assert m.project(10, 400, 40, 30)['projected_gpu_usd'] == pytest.approx((40+30+10*100*1.3+120)*.000542)
    assert m.project(100, 400, 40, 30)['fits'] is False
    assert m.limits([dict(status='failed', wall_seconds=100)], 1000)['absolute_deadline_unix'] == pytest.approx(1000+2.5/.000542-100-120)
    for attempts in ([dict(status='running', wall_seconds=1)], [dict(status='submission_unresolved', wall_seconds=1)], [dict(status='completed', wall_seconds=5000)]):
        with pytest.raises(ValueError):
            m.limits(attempts, 1000)


def fixture_rows():
    rows = []
    for rank in range(101, 201):
        for t in range(4):
            values = np.tile(np.linspace(1, 0, 20)[:, None], (1, 6))
            values *= (1 if rank <= 150 else 3)
            row = dict(id=f'{rank}-{t}', problem_idx=rank, selection_rank=rank, template_idx=t,
                       gold=42, leak=t<2, excluded_from_rates=False, closed_reasoning=True,
                       n_reasoning_tokens=20, token_identity_verified=True,
                       reading_tutor=[2.]*6, reading_neutral=[-1.]*6,
                       **api().summarize_trajectory(values))
            rows.append(row)
    return rows


def test_paired_analysis_known_slopes_anchor_and_cluster_ci():
    m = api()
    rows = fixture_rows()
    result = m.analyze(rows)
    assert len(result['deciles']) == 180
    assert len(result['slopes']) == len(result['reading']) == 18
    slope = result['slopes'][0]
    assert slope['estimate'] == pytest.approx(-2)
    # Four identical templates within a problem must move as a cluster.
    rng = np.random.default_rng(0)
    draws = rng.integers(0, 100, (2000, 100))
    expected = np.percentile(np.where(draws < 50, -1., -3.).mean(1), [2.5, 97.5])
    assert slope['ci95'] == pytest.approx(expected)
    assert result['reading'][0]['estimate'] == 3
    assert result['reading'][0]['tutor_mean'] == 2
    assert result['reading'][0]['neutral_mean'] == -1
    assert result['bootstrap'] == dict(resamples=2000, seed=0, unit='problem_idx', interval='percentile 95%', paired=True)
    rows[0]['excluded_from_rates'] = True
    result = m.analyze(rows)
    assert result['slopes'][0]['n_traces'] == 400
    assert result['slopes'][1]['n_traces'] == 199


@pytest.mark.parametrize('change', [lambda r: r.pop(), lambda r: r.append(deepcopy(r[0])),
    lambda r: r[0].update(token_identity_verified=False),
    lambda r: r[0]['deciles'][0].__setitem__(0, float('nan')),
    lambda r: r[0].update(reading_tutor=[1.])])
def test_analysis_refuses_partial_or_invalid_data(change):
    rows = fixture_rows()
    change(rows)
    with pytest.raises(ValueError):
        api().analyze(rows)


def test_modal_runner_has_no_warm_pool_retries_or_unbounded_gpu():
    from src import modal_app_3_neutral as runner
    import inspect
    source = inspect.getsource(runner)
    assert "gpu='L40S'" in source
    assert 'retries=0' in source and 'min_containers=0' in source and 'buffer_containers=0' in source
    assert 'single_use_containers=True' in source and 'max_containers=1' in source
    assert 'terminate_containers=True' in source and "'billing', 'summary'" in source
    assert 'assert_tokens(' in source and 'pilot' in source


def test_gpu_worker_import_does_not_require_local_analysis_dependencies():
    import subprocess
    import sys
    program = '''
import importlib.abc
import sys
class BlockAnalysis(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('pandas', 'scipy', 'statsmodels'):
            raise ImportError('GPU image excludes ' + fullname)
sys.meta_path.insert(0, BlockAnalysis())
import src.modal_app_3_neutral
'''
    result = subprocess.run([sys.executable,'-c',program],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr


def test_report_pairs_saved_token_scores_and_writes_numeric_artifacts(tmp_path):
    import json
    from pathlib import Path
    m = api()
    original = tmp_path/'results/raw'/m.SOURCE
    neutral = tmp_path/'results/raw/neutral'
    out = tmp_path/'results/phase3'
    for p in (original,neutral,out):p.mkdir(parents=True)
    rows = fixture_rows()
    arrays = []
    old_rows = []
    for r in rows:
        x = np.tile(np.linspace(1,0,20)[:,None],(1,6))*(1 if r['selection_rank']<=150 else 3)
        arrays.append(x)
        r['source_response_sha256'] = 'a'*64
        old_rows.append({**r, **m.summarize_trajectory(2*x)})
    scores = np.concatenate(arrays)
    for path, values, records in ((original,2*scores,old_rows),(neutral,scores,rows)):
        np.savez_compressed(path/('trajectories.npz' if path==original else 'neutral-trajectories.npz'),
            scores=values,token_ids=np.arange(8000),offsets=np.arange(0,8001,20),ids=np.asarray([r['id'] for r in rows]))
        (path/('tracking.jsonl' if path==original else 'neutral-tracking.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records))
    marker = tmp_path/'frozen';marker.write_text('source')
    metrics = dict(status='completed',manifest=dict(source_hashes={'frozen':m.file_hash(marker)}),reasoning_tokens=8000,pilot={'fits':True})
    (neutral/'neutral-metrics.json').write_text(json.dumps(metrics))
    (out/'neutral-execution.json').write_text(json.dumps(dict(zero_containers_verified=True,gpu_usd_upper=.25)))
    result = m.write_report(tmp_path,'neutral')
    assert result['slopes'][0]['estimate'] == pytest.approx(-2)
    assert len((out/'neutral-control.csv').read_text().splitlines()) == 181
    assert len((out/'neutral-traces.csv').read_text().splitlines()) == 401
    assert json.loads((out/'neutral-control.json').read_text()) == result
    assert '## reading' in (out/'neutral-control.md').read_text()
    with np.load(neutral/'neutral-trajectories.npz') as a:
        data = {k:a[k] for k in a.files}
    data['token_ids'][1] = 99999
    np.savez_compressed(neutral/'neutral-trajectories.npz',**data)
    with pytest.raises(AssertionError,match='tokens changed'):
        m.write_report(tmp_path,'neutral')
