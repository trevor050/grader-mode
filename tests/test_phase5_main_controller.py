"""Outcome-blind Gemma main ordering, full-batch projection and paid-run bounds."""
from collections import Counter
import pytest
from src import vast_generate_5 as worker
from src.models_5 import GEMMA, SAMPLING


def rows():
    return worker.expand_requests([dict(problem_idx=i, question=f'Question {i}?', gold=i)
                                   for i in range(200)], GEMMA)


def test_priority_preserves_exact_grid_and_keeps_controls_last():
    original = rows()
    ordered = worker.order_requests(original, 'main_priority')
    assert {r['id'] for r in ordered} == {r['id'] for r in original}
    assert all(r['arm'] == 'main' for r in ordered[:1600])
    assert all(r['prefill'] in ('empty', 'reminder') for r in ordered[1600:2400])
    assert all(r['prefill'] in ('neutral', 'irrelevant') for r in ordered[2400:])
    assert len(Counter((r['thinking'], r['request_type']) for r in ordered[:256])) == 4
    assert ordered == worker.order_requests(original, 'main_priority')


def test_engine_options_leave_pilot_eager_and_allow_main_graphs():
    assert worker.engine_options({})['enforce_eager'] is True
    options = worker.engine_options(dict(batch_size=256, chunk_size=256, enforce_eager=False))
    assert options['batch_size'] == options['chunk_size'] == 256
    assert options['enforce_eager'] is False
    assert options['max_model_len'] == 8192
    assert SAMPLING == dict(temperature=.7, top_p=.95, top_k=-1, max_tokens=4096)


def measurement(speed=2000):
    return dict(requests=256, output_tokens=256000, generation_seconds=256000/speed,
                elapsed_seconds=400, setup_seconds=200,
                per_cell={f'gemma/main/{t}/{r}':dict(count=64, output_tokens=64000)
                          for t in ('off', 'on') for r in ('plain', 'explicit')})


def means():
    from src.modal_main_5 import cell_id
    return {cell_id(r): 1000 for r in rows()}


def test_projection_keeps_full_grid_when_fits():
    from src.modal_main_5 import project_gemma
    result = project_gemma(measurement(5000), worker.order_requests(rows(), 'main_priority')[256:],
                           means(), prior_usd=.282284, current_seconds=400)
    assert result['decision'] == 'full'
    assert result['dropped_cells'] == []
    assert result['projected_total_usd'] < 3
    assert result['runtime_margin'] == 1.2
    assert result['teardown_and_failure_reserve_usd'] >= .15


def test_projection_drops_both_whole_cells_prospectively():
    from src.modal_main_5 import project_gemma
    result = project_gemma(measurement(1100), worker.order_requests(rows(), 'main_priority')[256:],
                           means(), prior_usd=.282284, current_seconds=400)
    assert result['decision'] == 'drop_controls'
    assert set(result['dropped_cells']) == {'gemma/prefill/neutral', 'gemma/prefill/irrelevant'}
    assert result['remaining_rows'] == 2400 - 256
    assert result['controls_generated_before_decision'] == 0
    assert result['projected_total_usd'] <= 3
    assert result['projections'][0]['projected_total_usd'] > 3


def test_unaffordable_mandatory_stops_without_shortening_cells():
    from src.modal_main_5 import project_gemma
    result = project_gemma(measurement(100), worker.order_requests(rows(), 'main_priority')[256:],
                           means(), prior_usd=.282284, current_seconds=400)
    assert result['decision'] == 'stop_noninferential'
    assert result['inference_eligible'] is False
    assert result['remaining_rows'] == 2144


def test_gate_refuses_small_measurements_or_any_control_already_generated():
    from src.modal_main_5 import project_gemma
    m = measurement()
    m['requests'] = 16
    with pytest.raises(ValueError, match='256'):
        project_gemma(m, rows()[256:], means(), prior_usd=.282284, current_seconds=400)
    m = measurement()
    m['per_cell']['gemma/prefill/neutral'] = dict(count=1, output_tokens=1)
    with pytest.raises(ValueError, match='control'):
        project_gemma(m, rows()[256:], means(), prior_usd=.282284, current_seconds=400)


def test_model_budget_retains_pilot_and_never_resets_failed_attempts():
    from src.modal_main_5 import model_prior_spend, deadline_allocation
    snapshot = dict(attempts=[dict(provider='modal', charged_microusd=282284),
                              dict(provider='modal', charged_microusd=200000),
                              dict(provider='vast', charged_microusd=1000000)])
    assert model_prior_spend(snapshot) == .482284
    allocation = deadline_allocation(.482284)
    assert .482284 + allocation['reserved_usd'] <= 3
    assert allocation['teardown_seconds'] >= 120
    assert allocation['startup_seconds'] >= 120
    assert allocation['runtime_seconds'] > 0
    with pytest.raises(Exception, match='budget'):
        deadline_allocation(2.99)


def test_main_worker_commits_each_chunk_and_drops_before_controls(monkeypatch, tmp_path):
    import json
    import sys
    from types import SimpleNamespace
    from test_phase5_formats import Tokenizer
    from test_phase5_requests import RenderTokenizer
    tok = Tokenizer(GEMMA)
    tok.apply_chat_template = RenderTokenizer(GEMMA).apply_chat_template
    batches, commits, gates = [], [], []
    class LLM:
        def __init__(self, **kw):
            assert kw['enforce_eager'] is False and kw['max_num_seqs'] == 256
        def generate(self, prompts, params, **kw):
            batches.append(len(prompts))
            assert all(p.max_tokens == 4096 and p.temperature == .7 and p.top_p == .95
                       and p.top_k == -1 for p in params)
            ids = tok.encode('Try a step.<turn|>')
            return [SimpleNamespace(prompt_token_ids=p['prompt_token_ids'],outputs=[SimpleNamespace(
                    token_ids=ids,finish_reason='stop',stop_reason=106)]) for p in prompts]
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(
                        from_pretrained=lambda *a,**kw:tok)))
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(LLM=LLM,
                        SamplingParams=lambda **kw:SimpleNamespace(**kw)))
    monkeypatch.setattr(worker,'runtime_versions',lambda:dict(worker.PACKAGE_VERSIONS))
    monkeypatch.setattr(worker,'verify_tokenizer',lambda *a:None)
    def checkpoint():
        path = tmp_path/'gemma-4-12B-it.jsonl'
        if path.exists():
            commits.append(len(path.read_text().splitlines()))
    def gate(progress, remaining):
        assert commits == [256]
        assert progress['requests'] == 256
        assert all('/prefill/' not in c for c in progress['per_cell'])
        gates.append(progress['requests'])
        return ['gemma/prefill/neutral','gemma/prefill/irrelevant']
    stats = worker.generate(GEMMA,rows(),'test',{'stage':'main'},dict(
            output_dir=str(tmp_path),batch_size=256,chunk_size=256,enforce_eager=False,
            ordering='main_priority',checkpoint=checkpoint,first_chunk_gate=gate))
    assert gates == [256]
    assert batches == [256]*9 + [96]
    assert commits == [256*i for i in range(1,10)] + [2400,2400]
    assert stats['requests'] == stats['requested_count'] == 2400
    assert stats['registered_count'] == 3200
    assert len(stats['per_cell']) == 6
    assert all(c['count'] == 400 for c in stats['per_cell'].values())
    raw = [json.loads(s) for s in (tmp_path/'gemma-4-12B-it.jsonl').read_text().splitlines()]
    assert all(r['prefill'] not in ('neutral','irrelevant') for r in raw)
    assert {r['id'] for r in raw} <= {r['id'] for r in rows()}
