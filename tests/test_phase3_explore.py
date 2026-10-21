"""Synthetic CPU contracts only. No Modal RPC, model downloads, or GPU calls."""
from copy import deepcopy
import json

import numpy as np
import pytest

from src.phase3 import CAP_SECONDS, RATE, build_requests


@pytest.fixture
def requests():
    # Deliberately nonnumeric selection order; no generated outcomes are supplied.
    return build_requests([dict(problem_idx=500-i, question=f'Synthetic {i}?', gold=42)
                           for i in range(200)])


@pytest.fixture
def neurons():
    top = dict(id='22.5128', layer=22, neuron=5128,
               d=4.38622263324684, delta=.1027524962792331)
    return dict(tutor=[top]+[dict(id=f'39.{i}', layer=39, neuron=i,
                                d=4-i/100, delta=.2+i/100) for i in range(19)])


def attempt(run_id='prior', seconds=100.):
    return dict(run_id=run_id, status='completed', wall_seconds=seconds,
                started_unix=100., zero_containers_verified_unix=100.+seconds,
                ended_unix=100.+seconds)


def test_exact_three_original_selection_template_zero(requests):
    from src.phase3_explore import select_probes
    probes = select_probes(requests)
    assert [(r['problem_idx'], r['selection_rank'], r['template_idx']) for r in probes] == [
        (500, 1, 0), (499, 2, 0), (498, 3, 0)]
    assert all(r['split'] == 'dev' and r['thinking'] and r['cue'] == 'none'
               and r['format'] == 'boxed' and r['arm'] == 'tutor' for r in probes)
    with pytest.raises(ValueError):
        select_probes([r for r in requests if r != probes[1]])


def test_fixed_conditions_and_coordinate_magnitudes(neurons):
    from src.phase3_explore import CELLS, condition
    assert CELLS == ('baseline', 'top1_plus', 'top1_minus', 'top1_zero', 'tutor20_minus')
    assert condition('baseline', neurons)['coordinates'] == []
    for cell, coefficient in [('top1_plus', 8), ('top1_minus', -8), ('top1_zero', 0)]:
        spec = condition(cell, neurons)
        assert spec['coefficient'] == coefficient
        assert spec['coordinates'][0]['id'] == '22.5128'
        assert spec['coordinates'][0]['amount'] == pytest.approx(coefficient*.1027524962792331)
        assert spec['mode'] == ('zero' if cell.endswith('zero') else 'add')
    all20 = condition('tutor20_minus', neurons)
    assert len(all20['coordinates']) == 20
    assert [c['amount'] for c in all20['coordinates']] == [-8*n['delta'] for n in neurons['tutor']]


@pytest.mark.parametrize('mode,amounts', [('add', [.0031, -3.0031]), ('zero', [0., 0.])])
def test_numpy_hook_generated_only_and_preserves_other_coordinates(mode, amounts):
    from src.phase3_explore import input_hook
    state = dict(generated=False)
    x = np.arange(30, dtype=np.float16).reshape(2, 3, 5)
    original = x.copy()
    hook = input_hook(np.array([1, 3]), np.array(amounts, dtype=np.float32), state, mode)
    assert hook(None, (x, 'other'))[0] is x
    state['generated'] = True
    y, other = hook(None, (x, 'other'))
    expected = original.astype(np.float32)
    expected[..., [1, 3]] = 0 if mode == 'zero' else expected[..., [1, 3]] + amounts
    np.testing.assert_array_equal(y, expected.astype(x.dtype))
    np.testing.assert_array_equal(x, original)
    assert other == 'other' and y.dtype == x.dtype


@pytest.mark.parametrize('mode', ['add', 'zero'])
def test_actual_torch_bf16_hook_and_weights_are_unchanged(mode):
    torch = pytest.importorskip('torch')
    from src.phase3_explore import input_hook
    layer = torch.nn.Linear(5, 3, bias=False, dtype=torch.bfloat16)
    weights = layer.weight.detach().clone()
    x = torch.arange(20, dtype=torch.bfloat16).reshape(2, 2, 5)
    before = x.clone()
    state = dict(generated=False)
    values = torch.tensor([.0031, -3.0031], dtype=torch.float32)
    handle = layer.register_forward_pre_hook(input_hook(torch.tensor([1, 3]), values, state, mode))
    try:
        assert torch.equal(layer(x), torch.nn.functional.linear(x, weights))
        state['generated'] = True
        expected = x.float()
        expected[..., [1, 3]] = 0 if mode == 'zero' else expected[..., [1, 3]] + values
        assert torch.equal(layer(x), torch.nn.functional.linear(expected.to(x.dtype), weights))
    finally:
        handle.remove()
    assert torch.equal(x, before) and torch.equal(layer.weight, weights)


@pytest.mark.parametrize('previous', [100., CAP_SECONDS-700., CAP_SECONDS-600.])
def test_both_caps_and_shutdown_reserve(previous):
    from src.phase3_explore import exploration_limits, SHUTDOWN_SECONDS
    limits = exploration_limits(previous, 1000.)
    assert limits['exposure_seconds'] <= 1.00/RATE
    assert previous+limits['exposure_seconds'] <= CAP_SECONDS
    assert limits['absolute_deadline_unix']+SHUTDOWN_SECONDS <= 1000.+limits['exposure_seconds']
    assert limits['local_cancel_unix'] <= limits['absolute_deadline_unix']
    assert limits['timeout_seconds']+limits['startup_timeout_seconds']+SHUTDOWN_SECONDS <= limits['exposure_seconds']


def test_refuse_before_spawn_when_setup_and_pair_cannot_fit():
    from src.phase3_explore import exploration_limits
    with pytest.raises(ValueError, match='setup.*paired'):
        exploration_limits(CAP_SECONDS-400, 1000.)


def test_first_gate_requires_two_additional_cells_then_slowest_completed():
    from src.phase3_explore import cells_fit, SHUTDOWN_SECONDS
    assert not cells_fit([100.], remaining_seconds=SHUTDOWN_SECONDS+259., cells=2)
    assert cells_fit([100.], remaining_seconds=SHUTDOWN_SECONDS+260., cells=2)
    assert not cells_fit([100., 200.], remaining_seconds=SHUTDOWN_SECONDS+259.)
    assert cells_fit([100., 200.], remaining_seconds=SHUTDOWN_SECONDS+260.)
    with pytest.raises(ValueError):
        cells_fit([float('nan')], remaining_seconds=1000.)


@pytest.mark.parametrize('mutation', [
    lambda a: a.update(status='running'),
    lambda a: a.update(status='shutdown_unresolved'),
    lambda a: a.pop('zero_containers_verified_unix'),
    lambda a: a.update(wall_seconds=1.),
])
def test_uncertain_ledger_blocks_preflight(mutation):
    from src.phase3_explore import lifetime_seconds
    row = attempt()
    mutation(row)
    with pytest.raises(ValueError):
        lifetime_seconds([row])


def test_all_prior_attempts_charged_and_unknown_submission_reserves_remainder():
    from src.phase3_explore import lifetime_seconds, close_allocation
    prior = [attempt('source'), attempt('registered-resume', 200.)]
    prior[-1]['source_run_id'] = 'source'
    assert lifetime_seconds(prior) == 300.
    for known, zero in [(False, False), (True, False)]:
        closed = close_allocation(call_known=known, elapsed=12., previous_seconds=300.,
                                  status='failed', zero_containers=zero)
        assert closed['wall_seconds'] == CAP_SECONDS-300.
        assert closed['status'] not in {'completed', 'failed'}


def make_row(requests, neurons, cell='baseline', **changes):
    from src.phase3_explore import make_generation_row
    arguments = dict(source=requests[0], cell=cell, neurons=neurons, source_run_id='registered',
                     run_id='explore-unique', response='<think>42, tutor</think>Try another step.',
                     output_tokens=30, end_token_id=151645, batch_seed=requests[0]['seed'],
                     rendered_prompt_sha256='a'*64, final_reply_tokens=4, incomplete=False)
    arguments.update(changes)
    return make_generation_row(**arguments)


def test_disjoint_identity_final_only_scoring_and_incomplete_retention(requests, neurons):
    row = make_row(requests, neurons)
    assert row['exploratory'] is True and row['analysis_excluded'] is True
    assert row['run_id'] == 'explore-unique' and row['source_run_id'] == 'registered'
    assert row['id'].startswith('explore-unique:') and row['source_request_id'] == requests[0]['id']
    assert not row['leak'] and not row['commit'] and row['tutor_mention']
    assert row['reasoning'] and row['final_reply'] == 'Try another step.'
    partial = make_row(requests, neurons, output_tokens=12, end_token_id=None, incomplete=True)
    assert partial['cell_complete'] is False and partial['budget_terminated'] is True
    with pytest.raises(ValueError, match='incomplete'):
        make_row(requests, neurons, output_tokens=12, end_token_id=None)


def test_counts_and_pairs_use_completed_matched_nonempty_toy_rows(requests, neurons):
    from src.phase3_explore import summarize
    base = [make_row(requests, neurons, source=r, response='<think>work</think>Try again.')
            for r in requests if r['selection_rank'] <= 3 and r['template_idx'] == 0]
    pushed = [make_row(requests, neurons, source=r, cell='top1_plus',
                       response='<think>work</think>\\boxed{42}')
              for r in requests if r['selection_rank'] <= 3 and r['template_idx'] == 0]
    pushed[2] = make_row(requests, neurons, source=requests[8], cell='top1_plus',
                         response='<think>unfinished', end_token_id=None, output_tokens=3072)
    partial = make_row(requests, neurons, cell='top1_minus', incomplete=True,
                       end_token_id=None, output_tokens=5)
    summary = summarize(base+pushed+[partial], completed_cells=['baseline', 'top1_plus'])
    assert summary['total_toy_examples'] == 6 and summary['incomplete_rows'] == 1
    assert summary['cells']['top1_plus']['n_valid'] == 2
    assert summary['cells']['top1_plus']['empty'] == 1
    pair = summary['pairs_against_baseline']['top1_plus']
    assert pair['n_matched_nonempty'] == 2 and pair['leak_difference_count'] == 2
    assert summary['partial_exploration'] and summary['skipped_cells'] == [
        'top1_minus', 'top1_zero', 'tutor20_minus']
    assert 'top1_minus' not in summary['cells']


def test_transport_rejects_rehashed_nonoriginal_probe_and_dose(requests, neurons):
    from src.phase3_explore import make_package, validate_package
    from src.phase3_resume import digest
    original = dict(source_run_id='registered', source_manifest={'sample': [
        dict(problem_idx=500-i, question=f'Synthetic {i}?', gold=42) for i in range(200)]},
        source_selection=dict(c_star=8, dose_threshold_met=False,
                              dose_trials=[dict(c=i, n=100) for i in (1, 2, 4, 8)]),
        neurons=neurons, dev_requests=[r for r in requests if r['split'] == 'dev' and r['template_idx'] == 0],
        prompt_hashes={r['id']: 'a'*64 for r in requests}, source_artifacts_sha256={'fake': 'b'*64})
    package = make_package(original, [attempt()])
    validate_package(package)
    for mutate in [lambda p: p['probes'].__setitem__(0, requests[4]),
                   lambda p: p['source_selection'].update(c_star=4),
                   lambda p: p['neurons']['tutor'][0].update(delta=1.)]:
        changed = deepcopy(package)
        mutate(changed)
        changed['payload_sha256'] = digest({k: v for k, v in changed.items() if k != 'payload_sha256'})
        with pytest.raises(ValueError):
            validate_package(changed)


def test_explorer_imports_without_optional_analysis_dependencies():
    import subprocess
    import sys
    script = '''
import sys
class Deny:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('pandas', 'statsmodels'):
            raise ImportError('absent from original GPU image: '+fullname)
sys.meta_path.insert(0, Deny())
import src.modal_app_3_explore
'''
    subprocess.run([sys.executable, '-c', script], check=True, capture_output=True, text=True)
