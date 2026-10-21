"""Synthetic-only trace coding tests; no real generations or API calls."""
import importlib
import json
from pathlib import Path

import pytest


def module():
    return importlib.import_module('src.trace_coding')


def raw(i=0, checkpoint='RLVR', **changes):
    suffix = {'SFT': '-SFT', 'DPO': '-DPO', 'RLVR': ''}[checkpoint]
    return dict(id=f'synthetic-{i}', model=f'allenai/Olmo-3-7B-Think{suffix}',
                phase='phase1c', ladder='think', checkpoint=checkpoint, thinking=True,
                arm='tutor', prose='plain', cue='plain', format='boxed', template_idx=i % 4,
                problem_idx=i, gold=42, messages=[{'role': 'system', 'content': 'Tutor rules.'},
                {'role': 'user', 'content': ' Full user message. Demand \\boxed{}!\n'}],
                response='<think>Reasoning only.</think>FINAL SECRET 42', finish_reason='stop') | changes


def test_blinding_preserves_user_and_excludes_final_and_identifiers():
    m = module()
    trace = m.extract_trace(raw())
    assert set(trace['blinded']) == {'tutor_system_prompt', 'user_message', 'reasoning_block'}
    assert trace['blinded']['user_message'] == raw()['messages'][1]['content']
    assert trace['blinded']['reasoning_block'] == 'Reasoning only.'
    body = m.judge_body(trace)
    serialized = json.dumps(body)
    for forbidden in ('FINAL SECRET', 'RLVR', 'Olmo', 'synthetic-0', 'run_id', 'checkpoint'):
        assert forbidden not in serialized
    assert body['response_format']['json_schema']['schema']['properties']['category']['enum'] == ['(a)', '(b)', '(c)']


def test_codebook_verbatim_from_design():
    design = Path('DESIGN.md').read_text()
    definitions = design.split('For the Think ladder, each reasoning block is classified into exactly one of:\n', 1)[1].split(' Classification is by', 1)[0]
    assert module().DEFINITIONS == definitions
    assert definitions in module().JUDGE_INSTRUCTIONS


def test_unclosed_reasoning_retains_entire_response_and_flag():
    trace = module().extract_trace(raw(response='<think>Still reasoning', finish_reason='length'))
    assert trace['reasoning_unclosed'] and trace['truncated']
    assert trace['blinded']['reasoning_block'] == '<think>Still reasoning'
    assert module().extract_trace(raw(response='Implicit reasoning</think>final'))['blinded']['reasoning_block'] == 'Implicit reasoning'


@pytest.mark.parametrize('changes', [dict(phase='phase1c-calibration'), dict(thinking=False),
    dict(prose='env'), dict(model='wrong'), dict(messages=[{'role': 'user', 'content': 'oops'}]),
    dict(response=None), dict(finish_reason='error')])
def test_malformed_tutor_data_rejected(changes):
    with pytest.raises(ValueError):
        module().extract_trace(raw(**changes))


def test_loading_deduplicates_identity_and_ignores_ceiling(tmp_path):
    m = module()
    p = tmp_path / 'input.jsonl'
    p.write_text('\n'.join(json.dumps(r) for r in [raw(), raw(1, arm='ceiling')]))
    assert len(m.load_traces([p])) == 1
    p.write_text('\n'.join(json.dumps(r) for r in [raw(), raw()]))
    with pytest.raises(ValueError, match='duplicate'):
        m.load_traces([p])


def test_projection_counts_tokens_without_state_or_outcomes():
    m = module()
    traces = [m.extract_trace(raw())]
    report = m.project_cost(traces, project_total=6000)
    assert report['n_traces'] == 1 and report['project_total'] == 6000
    assert report['estimated_input_tokens'] > 0
    assert 0 < report['projected_usd'] <= report['projected_budget_upper_usd']
    assert not any(k in json.dumps(report) for k in ['leak', 'commit', 'category'])
    huge = m.extract_trace(raw(response='x' * 200000 + '</think>hidden'))
    with pytest.raises(ValueError, match='cap'):
        m.project_cost([huge], project_total=6000)


def test_seeded_sample_blinded_and_order_independent(tmp_path):
    m = module()
    traces = [m.extract_trace(raw(i)) for i in range(120)]
    one, two = tmp_path/'a.jsonl', tmp_path/'b.jsonl'
    m.write_sample(traces, one)
    m.write_sample(list(reversed(traces)), two)
    assert one.read_bytes() == two.read_bytes()
    rows = [json.loads(x) for x in one.read_text().splitlines()]
    assert len(rows) == 100 and len({r['trace_id'] for r in rows}) == 100
    assert all(set(r) == {'trace_id', 'tutor_system_prompt', 'user_message', 'reasoning_block', 'category'} for r in rows)
    assert all(r['category'] is None for r in rows)
    assert 'checkpoint' not in one.read_text() and 'FINAL SECRET' not in one.read_text()
    with pytest.raises(ValueError, match='100'):
        m.write_sample(traces[:10], tmp_path/'small.jsonl')


def test_agreement_matches_known_kappa_and_requires_exact_sample():
    m = module()
    a = {str(i): '(a)' if i < 50 else '(b)' for i in range(100)}
    b = {str(i): '(a)' if i < 40 or 50 <= i < 60 else '(b)' for i in range(100)}
    result = m.agreement(a, b, a)
    assert result['judge1_vs_judge2']['raw_agreement'] == .8
    assert result['judge1_vs_judge2']['cohens_kappa'] == pytest.approx(.6)
    assert result['judge1_vs_hand']['cohens_kappa'] == 1
    with pytest.raises(ValueError, match='100|same'):
        m.agreement(a, b | {'extra': '(c)'}, a)
    same = {str(i): '(a)' for i in range(100)}
    assert m.agreement(same, same, same)['judge1_vs_hand']['cohens_kappa'] is None


def test_report_category_shares_and_valid_final_denominators():
    m = module()
    rows = [raw(0), raw(1, response='unfinished', finish_reason='length'),
            raw(2, response='Think</think>Try a step.'), raw(3, checkpoint='DPO')]
    traces = [m.extract_trace(r) for r in rows]
    labels = {t['trace_id']: '(a)' if i < 2 else '(b)' for i,t in enumerate(traces)}
    report = m.category_report(rows, labels)
    cell = next(c for c in report['cells'] if c['checkpoint'] == 'RLVR')
    assert cell['n_traces'] == 3
    assert cell['categories']['(a)']['share'] == pytest.approx(2/3)
    assert cell['categories']['(a)']['n_excluded_empty_final'] == 1
    assert cell['categories']['(a)']['leak_rate'] == 1
    assert cell['categories']['(b)']['leak_rate'] == 0
    with pytest.raises(ValueError, match='label'):
        m.category_report(rows, {})


def test_cli_report_requires_generation_complete_before_reading(tmp_path):
    with pytest.raises(SystemExit) as exc:
        module().main(['report', '--input', str(tmp_path/'does-not-exist'), '--labels', 'missing', '--output', str(tmp_path/'report.json')])
    assert exc.value.code == 2
    assert not (tmp_path/'report.json').exists()


def test_agreement_cli_rejects_same_family_before_reading(tmp_path):
    with pytest.raises(SystemExit) as exc:
        module().main(['agreement', '--judge1', 'missing', '--judge2', 'missing', '--hand', 'missing',
            '--sample', 'missing', '--output', str(tmp_path/'output.json'),
            '--judge2-family', 'openai', '--judge2-model', 'gpt-6.1-sol'])
    assert exc.value.code == 2
    assert not (tmp_path/'output.json').exists()


def test_agreement_cli_complete_blinded_file_interchange(tmp_path):
    m = module()
    traces = [m.extract_trace(raw(i)) for i in range(110)]
    sample = tmp_path/'sample.jsonl'
    m.write_sample(traces, sample)
    selected = [json.loads(x) for x in sample.read_text().splitlines()]
    first = tmp_path/'first.jsonl'
    first.write_text('\n'.join(json.dumps({'trace_id':t['trace_id'], 'category':'(a)'}) for t in traces))
    second, hand = tmp_path/'second.jsonl', tmp_path/'hand.jsonl'
    labeled = '\n'.join(json.dumps(r | {'category': '(a)' if i < 80 else '(b)'}) for i,r in enumerate(selected))
    second.write_text(labeled); hand.write_text(labeled)
    output = tmp_path/'agreement.json'
    m.main(['agreement', '--judge1', str(first), '--judge2', str(second), '--hand', str(hand),
        '--sample', str(sample), '--output', str(output), '--judge2-family', 'anthropic', '--judge2-model', 'synthetic-judge'])
    result = json.loads(output.read_text())
    assert result['judge1_vs_judge2']['n'] == 100
    assert result['judge1_vs_judge2']['raw_agreement'] == .8
    assert result['judge2_vs_hand']['cohens_kappa'] == 1
    assert result['judge_provenance']['judge2']['family'] == 'anthropic'


def test_dry_run_cli_only_emits_counts_no_labels_or_state(tmp_path, monkeypatch, capsys):
    import src.trace_coding_batch as batch
    monkeypatch.setattr(batch, 'open_client', lambda _: pytest.fail('dry run must not open API client'))
    source = tmp_path/'input.jsonl'
    source.write_text(json.dumps(raw())+'\n')
    output = tmp_path/'cost.json'
    module().main(['dry-run', '--input', str(source), '--output', str(output)])
    content = output.read_text() + capsys.readouterr().out
    assert 'FINAL SECRET' not in content and 'Reasoning only' not in content
    assert 'leak' not in content and 'commit' not in content and 'category' not in content
    assert set(p.name for p in tmp_path.iterdir()) == {'input.jsonl','cost.json'}


def test_unclosed_nested_reasoning_preserves_whole_response():
    text = '<think>outer<think>inner</think>unfinished'
    trace = module().extract_trace(raw(response=text, finish_reason='length'))
    assert trace['reasoning_unclosed'] is True
    assert trace['blinded']['reasoning_block'] == text
    closed = module().extract_trace(raw(response=text + '</think>'))
    assert closed['reasoning_unclosed'] is False
    assert closed['blinded']['reasoning_block'] == 'outerinnerunfinished'
