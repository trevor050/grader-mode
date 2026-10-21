"""CPU-only logit-probe contracts; never submits Modal work."""
from copy import deepcopy
import json
import subprocess
import sys

import numpy as np
import pytest


def test_last_position_bf16_clone_and_no_weight_mutation():
    import torch
    from src.phase3_logit_probe import last_position_hook
    layer = torch.nn.Linear(5, 2, bias=False, dtype=torch.bfloat16)
    x = torch.arange(30, dtype=torch.bfloat16).reshape(2, 3, 5)
    x[..., -1, 1] = 1.0078125
    original, weights = x.clone(), layer.weight.detach().clone()
    for mode in ('add', 'zero'):
        values = torch.tensor([.0031, -3.0031], dtype=torch.float32)
        hook = last_position_hook(torch.tensor([1, 3]), values, mode)
        y, extra = hook(layer, (x, 'extra'))
        expected = x.float().clone()
        expected[..., -1, [1, 3]] = 0 if mode == 'zero' else expected[..., -1, [1, 3]]+values
        assert torch.equal(y, expected.to(x.dtype)) and y.dtype == torch.bfloat16
        assert torch.equal(y[..., :-1, :], x[..., :-1, :]) and extra == 'extra'
        handle = layer.register_forward_pre_hook(hook)
        try:
            assert torch.equal(layer(x), torch.nn.functional.linear(y, weights))
        finally:
            handle.remove()
    assert torch.equal(x, original) and torch.equal(layer.weight, weights)


class CharTokenizer:
    is_fast = True
    def __call__(self, text, **kwargs):
        return dict(input_ids=[ord(c) for c in text],
                    offset_mapping=[(i, i+1) for i in range(len(text))])
    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))


def test_exact_cut_after_closed_think_and_whitespace():
    from src.phase3_logit_probe import prepare_prefix, text_hash
    prompt, response = 'PROMPT<think>\n', 'reasoning</think>\n  First final'
    result = prepare_prefix(CharTokenizer(), prompt, response, text_hash(prompt))
    assert result['eligible']
    prefix = prompt+'reasoning</think>\n  '
    assert result['input_ids'] == list(map(ord, prefix))
    assert result['actual_next_token_id'] == ord('F')
    assert result['last_input_offset'][0] >= len(prompt)
    assert result['response_prefix_sha256'] == text_hash(response[:response.index('First')])
    assert result['token_prefix_sha256'] and result['token_prefix_verified']
    for bad, reason in [('unfinished', 'missing_closed_think'), ('a</think> \n', 'empty_final')]:
        assert prepare_prefix(CharTokenizer(), prompt, bad, text_hash(prompt))['exclusion'] == reason
    with pytest.raises(ValueError, match='rendered prompt'):
        prepare_prefix(CharTokenizer(), prompt, response, '0'*64)


def test_inside_token_and_prompt_merge_are_excluded_honestly():
    from src.phase3_logit_probe import prepare_prefix, text_hash
    class Merge(CharTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            if ' F' not in text:
                return result
            i = text.index(' F')
            result['input_ids'][i:i+2] = [999]
            result['offset_mapping'][i:i+2] = [(i, i+2)]
            return result
    assert prepare_prefix(Merge(), 'P', 'x</think> Final', text_hash('P'))['exclusion'] == 'boundary_inside_token'
    class PrefixChange(CharTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            if '</think>' in text:
                result['input_ids'][0] += 1
            return result
    with pytest.raises(ValueError, match='prompt token prefix'):
        prepare_prefix(PrefixChange(), 'P', 'x</think> Final', text_hash('P'))


def test_full_vocab_log_probs_and_explicit_kl_direction():
    import torch
    from src.phase3_logit_probe import log_probs, contrast_stats
    baseline = log_probs(torch.tensor([0., 1., -2.], dtype=torch.bfloat16))
    pushed = log_probs(torch.tensor([1., 0., -2.], dtype=torch.bfloat16))
    assert baseline.dtype == np.float32
    np.testing.assert_allclose(np.exp(baseline).sum(), 1., rtol=1e-6)
    result = contrast_stats(baseline, pushed, 1, lambda ids: str(ids[0]))
    expected = np.sum(np.exp(pushed)*(pushed-baseline))
    assert result['kl_push_to_base'] == pytest.approx(float(expected))
    assert result['actual_next_token']['id'] == 1
    assert result['actual_next_token']['logp_change'] == pytest.approx(-1.)
    assert result['most_increased'][0]['id'] == 0
    assert result['most_decreased'][0]['id'] == 1
    assert result['baseline_top10'][0]['id'] == 1
    assert result['pushed_top10'][0]['id'] == 0
    assert result['baseline_argmax'] == result['baseline_top10'][0]
    assert result['pushed_argmax'] == result['pushed_top10'][0]
    assert result['baseline_top10'][0]['baseline_probability'] == pytest.approx(float(np.exp(baseline[1])))
    assert result['entropy_push'] == pytest.approx(float(-np.sum(np.exp(pushed)*pushed)))
    assert len(result['most_increased']) == 3
    with pytest.raises(ValueError):
        contrast_stats(baseline, np.array([np.nan]*3), 1, str)


def test_conditions_use_original_control_rank_magnitudes_and_signs():
    from src.phase3_logit_probe import CELLS, condition
    tutor = [dict(id=f'22.{5128+i}', layer=22, neuron=5128+i, delta=.1+i) for i in range(20)]
    control = [dict(id=f'22.{i}', layer=22, neuron=i, delta=999.) for i in range(20)]
    neurons = dict(tutor=tutor, controls=[control], control_signs=[[-1, 1]*10])
    assert len(CELLS) == 10 and condition('baseline', neurons)['coordinates'] == []
    assert condition('top1_plus64delta', neurons)['coordinates'][0]['amount'] == pytest.approx(6.4)
    assert condition('tutor20_minus8delta', neurons)['coordinates'][-1]['amount'] == pytest.approx(-152.8)
    spec = condition('control0_plus8', neurons)
    assert [r['amount'] for r in spec['coordinates']] == [8*abs(t['delta'])*s for t, s in zip(tutor, [-1, 1]*10)]
    assert [r['id'] for r in spec['coordinates']] == [r['id'] for r in control]
    assert condition('tutor20_zero', neurons)['mode'] == 'zero'


def test_nested_caps_measured_gate_and_unknown_reserves_entire_lifetime():
    from src.phase3 import CAP_SECONDS, RATE
    from src.phase3_logit_probe import probe_limits, grid_fits, close_allocation, lifetime_seconds, SHUTDOWN_SECONDS
    previous = (3.0216+1.)/RATE
    limits = probe_limits(previous, 1000.)
    assert limits['exposure_seconds']*RATE <= .60+1e-12
    assert limits['exposure_seconds']+previous <= CAP_SECONDS
    assert limits['timeout_seconds']+limits['startup_timeout_seconds']+SHUTDOWN_SECONDS <= limits['exposure_seconds']
    assert limits['absolute_deadline_unix']+SHUTDOWN_SECONDS <= 1000.+limits['exposure_seconds']
    assert grid_fits([1., 2.], 28, 28*2*1.3+SHUTDOWN_SECONDS)
    assert not grid_fits([1., 2.], 28, 28*2*1.3+SHUTDOWN_SECONDS-.1)
    with pytest.raises(ValueError):
        probe_limits(CAP_SECONDS-350, 1000.)
    row = dict(run_id='a', status='completed', wall_seconds=10., started_unix=1., ended_unix=11., zero_containers_verified_unix=11.)
    assert lifetime_seconds([row]) == 10.
    for mutate in (lambda a: a.update(status='running'), lambda a: a.pop('zero_containers_verified_unix')):
        bad = deepcopy(row); mutate(bad)
        with pytest.raises(ValueError):
            lifetime_seconds([bad])
    for known, zero in ((False, True), (True, False), (False, False)):
        closed = close_allocation(call_known=known, elapsed=10., previous_seconds=previous, status='failed', zero_containers=zero)
        assert closed['wall_seconds'] == CAP_SECONDS-previous
        with pytest.raises(ValueError):
            lifetime_seconds([{**row, **closed}])


def test_remote_import_without_pandas_statsmodels_or_calls():
    script = '''
import sys
class Deny:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('pandas', 'statsmodels'):
            raise ImportError(fullname)
sys.meta_path.insert(0, Deny())
import src.modal_app_3_logit_probe
'''
    subprocess.run([sys.executable, '-c', script], check=True, capture_output=True, text=True)


def test_preflight_refuses_active_explorer_before_reading_any_source(tmp_path):
    from src.phase3_logit_probe import load_package, SOURCE, EXPLORER
    running = [dict(run_id=EXPLORER, status='running')]
    with pytest.raises(ValueError, match='running/unresolved'):
        load_package(tmp_path, SOURCE, EXPLORER, 'phase3-logit-new', running)
    with pytest.raises(ValueError, match='fixed original/explorer'):
        load_package(tmp_path, SOURCE, 'another-explorer', 'phase3-logit-new', running)
    for status in ('completed', 'failed'):
        prior = [dict(run_id='phase3-logit-earlier', status=status,
                      wall_seconds=10., started_unix=1., ended_unix=11.,
                      zero_containers_verified_unix=11.)]
        with pytest.raises(ValueError, match='one logit allocation'):
            load_package(tmp_path, SOURCE, EXPLORER, 'phase3-logit-new', prior)
    assert list(tmp_path.iterdir()) == []


def test_remote_declaration_has_no_generation_and_requires_complete_grid():
    import ast
    from pathlib import Path
    tree = ast.parse(Path('src/modal_app_3_logit_probe.py').read_text())
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert 'generate' not in calls and 'manual_seed' not in calls
    probe = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'probe')
    options = {k.arg: ast.literal_eval(k.value) for k in probe.decorator_list[0].keywords
               if k.arg in ('gpu', 'min_containers', 'buffer_containers', 'max_containers', 'retries', 'single_use_containers')}
    assert options == dict(gpu='L40S', min_containers=0, buffer_containers=0,
                           max_containers=1, retries=0, single_use_containers=True)
