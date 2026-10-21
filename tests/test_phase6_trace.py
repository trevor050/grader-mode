import copy
import json
from pathlib import Path

import pytest

from src import phase6_trace as t


def row(model='Qwen/Qwen3-32B', pid=1, template=0):
    return dict(model=model, id=f'private-{pid}-{template}', problem_id=pid,
                template=template, arm='main', thinking=True, request_type='explicit',
                prefill='none', prefill_text='', reasoning='I must guide, not give the final answer.',
                messages=[dict(role='system', content='Tutor rule'), dict(role='user', content='Help me')],
                gold=17, violation=True, final_reply='SECRET FINAL')


def test_blind_allowlist_and_seeded_shuffle():
    rows = [row(pid=i, template=j) for i in range(4) for j in (0, 1)]
    blind, secret = t.blind_rows(rows)
    assert blind == t.blind_rows(list(reversed(rows)))[0]
    assert len({r['id'] for r in blind}) == 8
    assert all(set(r) == {'id', 'tutor_prompt', 'user_message', 'reasoning'} for r in blind)
    assert 'SECRET FINAL' not in json.dumps(blind)
    assert 'Qwen' not in json.dumps(blind)
    assert {r['source_id'] for r in secret} == {r['id'] for r in rows}
    assert blind != sorted(blind, key=lambda r: r['id'])


def test_stable_ids_when_new_pool_arrives():
    a = t.blind_rows([row()])[0][0]['id']
    b = t.blind_rows([row(), row(model='google/gemma-4-12B-it')])[0]
    assert a in {r['id'] for r in b}


@pytest.mark.parametrize('mutation', [dict(prefill_text='reminder'), dict(reasoning=None),
    dict(violation=1), dict(thinking='on'), dict(messages=[]), dict(template=2),
    dict(protocol_failure=1)])
def test_invalid_eligible_row_rejected(mutation):
    with pytest.raises(ValueError):
        t.blind_rows([row() | mutation])


def test_duplicates_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        t.blind_rows([row(), row()])


def test_selection_keeps_only_registered_cell():
    rs = [row(), row() | dict(thinking=False), row() | dict(arm='prefill'),
          row() | dict(request_type='plain')]
    assert t.select_cell(rs, 'Qwen/Qwen3-32B') == [rs[0]]


def test_grid_requires_both_templates_and_full_matched_problem_set():
    rows = [row(pid=i, template=j) for i in range(200) for j in (0, 1)]
    t.validate_grid(rows)
    with pytest.raises(ValueError):
        t.validate_grid(rows[:-1])
    with pytest.raises(ValueError):
        t.validate_grid(rows + [row(pid=201)])


def test_judge_exact_ids_labels_and_evidence():
    good = {'labels': [{'id': 'opaque', 'label': 'R', 'evidence': 'guide'}]}
    assert t.validate_labels(good, ['opaque'])[0]['label'] == 'R'
    for bad in [dict(labels=[]), dict(labels=good['labels'] * 2),
                dict(labels=[dict(id='other', label='R', evidence='')]),
                dict(labels=[dict(id='opaque', label='X', evidence='')]),
                dict(labels=[dict(id='opaque', label='N', evidence='', model='leak')])]:
        with pytest.raises(ValueError):
            t.validate_labels(bad, ['opaque'])


def test_kappa_and_degenerate_denominator():
    result = t.agreement(['R', 'R', 'N', 'L'], ['R', 'N', 'N', 'L'])
    assert result['agreement'] == .75
    assert result['kappa'] == pytest.approx(7 / 11)
    assert t.agreement(['R'], ['R'])['kappa'] is None
    with pytest.raises(ValueError):
        t.agreement(['R'], [])


def test_sample_defers_until_complete_and_is_seeded():
    rows = [{'id': str(i)} for i in range(120)]
    with pytest.raises(ValueError, match='complete'):
        t.agreement_sample(rows, pool_complete=False)
    a = t.agreement_sample(rows, pool_complete=True)
    assert len(a) == 100
    assert a == t.agreement_sample(rows[::-1], pool_complete=True)


def test_keyword_continuity_literal_regex():
    assert t.keyword_mention('SYSTEM PROMPT')
    assert t.keyword_mention('instructions')
    assert t.keyword_mention('tutorial')
    assert not t.keyword_mention('I cannot give the answer')


def test_commands_isolation(tmp_path):
    cmd = t.codex_command(Path('/p'), Path('/s'), Path('/o'))
    for flag in ['--ephemeral', '--skip-git-repo-check', '--output-schema']:
        assert flag in cmd
    assert cmd[-1] == '-'
    assert 'model_reasoning_effort=medium' in cmd
    assert 'model_instructions_file=/p' in cmd
    env = t.judge_env({'PATH': '/bin', 'OPENAI_API_KEY': 'secret', 'CODEX_HOME': 'bad',
                       'T3_SECRET': 'bad', 'HOME': '/home'})
    assert env['OPENAI_API_KEY'] == ''
    assert env['CODEX_HOME'] == '/tmp/cx-home'
    assert 'T3_SECRET' not in env
    claude = t.claude_command('only prompt')
    assert claude[claude.index('--tools') + 1] == ''
    assert claude[claude.index('--setting-sources') + 1] == ''
    assert '--strict-mcp-config' in claude


def test_association_eligibility_cluster_bootstrap_and_pending():
    secret = []
    labels = {}
    for i in range(40):
        r = row(pid=i) | dict(violation=i >= 20)
        b, s = t.blind_rows([r])
        secret += s
        labels[b[0]['id']] = 'R' if i < 20 else 'N'
    result = t.model_report(secret, labels)['Qwen/Qwen3-32B']
    assert result['eligible']
    assert result['difference_R_minus_N'] == -1
    assert result['ci95'] == [-1, -1]
    assert result['holds'] is True
    result = t.model_report(secret, {})['Qwen/Qwen3-32B']
    assert result['pending'] == 40 and result['holds'] is None


def test_blind_input_guard_rejects_hidden_metadata():
    blind = t.blind_rows([row()])[0]
    t.check_blind(blind)
    with pytest.raises(ValueError, match='allowlist'):
        t.check_blind([blind[0] | {'model': 'hidden'}])
    with pytest.raises(ValueError, match='duplicate'):
        t.check_blind(blind * 2)


def test_isolated_runner_validates_and_resumes_without_second_call(tmp_path, monkeypatch):
    from types import SimpleNamespace
    blind = t.blind_rows([row()])[0]
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert not list(Path(kwargs['cwd']).iterdir())
        assert kwargs['env']['OPENAI_API_KEY'] == ''
        assert kwargs['env']['CODEX_HOME'] == '/tmp/cx-home'
        assert set(json.loads(kwargs['input'])['traces'][0]) == t.FIELDS
        output = Path(command[command.index('-o') + 1])
        output.write_text(json.dumps({'labels': [{'id': blind[0]['id'], 'label': 'R', 'evidence': 'guide'}]}))
        return SimpleNamespace(returncode=0, stdout='{"type":"turn.completed"}\n', stderr='')

    monkeypatch.setattr(t.subprocess, 'run', fake_run)
    a = t.judge_batch(blind, 'sol', tmp_path, Path('/p'), Path('/s'), 'config-hash')
    b = t.judge_batch(blind, 'sol', tmp_path, Path('/p'), Path('/s'), 'config-hash')
    assert a == b and len(calls) == 1
    assert t.collect_labels(tmp_path, 'sol', blind) == {blind[0]['id']: 'R'}
    assert a['cwd_was_empty'] and a['input_sha256'][blind[0]['id']] == t.digest(blind[0])


def test_runner_rejects_tool_calls_and_retains_failed_attempt(tmp_path, monkeypatch):
    from types import SimpleNamespace
    blind = t.blind_rows([row()])[0]
    monkeypatch.setattr(t.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout='{"type":"item.completed","item":{"type":"command_execution"}}\n', stderr=''))
    with pytest.raises(ValueError, match='used tools'):
        t.judge_batch(blind, 'sol', tmp_path, Path('/p'), Path('/s'), 'cfg')
    assert t.collect_labels(tmp_path, 'sol', blind) == {}
    attempt = json.loads(next(tmp_path.glob('*.attempt-0.json')).read_text())
    assert attempt['status'] == 'failed'


def test_source_id_change_cannot_change_cached_judgment(tmp_path):
    blind = t.blind_rows([row()])[0]
    saved = dict(status='complete', prompt_sha256=t.digest(t.JUDGE_PROMPT.encode()),
                 input_ids=[blind[0]['id']], input_sha256={blind[0]['id']: 'wrong'},
                 labels=[dict(id=blind[0]['id'], label='N', evidence='none')])
    (tmp_path / 'sol-test.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='changed'):
        t.collect_labels(tmp_path, 'sol', blind)


def test_secondary_judge_requires_certified_combined_pool(tmp_path):
    blind = [dict(id=str(i), tutor_prompt='T', user_message='U', reasoning='R') for i in range(120)]
    with pytest.raises(ValueError, match='combined pool'):
        t.secondary_input(tmp_path, blind)
    sample = t.agreement_sample(blind, pool_complete=True)
    t.write_jsonl(tmp_path / 'trace-agreement-blind.jsonl', sample)
    t.write_json(tmp_path / 'trace-agreement-selection.json', dict(
        pool_sha256=t.digest(blind), pool_complete=True, seed=0,
        selected_ids=[r['id'] for r in sample]))
    assert t.secondary_input(tmp_path, blind) == sample
    with pytest.raises(ValueError, match='pool changed'):
        t.secondary_input(tmp_path, blind[:-1])


def test_attempt_accounting_counts_failures_and_repeated_presentations(tmp_path):
    for index, status in enumerate(['complete', 'failed', 'started']):
        t.write_json(tmp_path / f'sol-a{index}.attempt-0.json', dict(
            engine='sol', status=status, input_ids=['a', 'b'], elapsed_seconds=1.5))
    counts = t.attempt_accounting(tmp_path, 'sol')
    assert counts == dict(attempts=3, complete=1, failed=1, unfinished=1,
                          trace_presentations=6, elapsed_call_seconds=4.5)


def test_future_phase6_canonical_factors_and_conflicting_aliases():
    r = row(model='Qwen/Qwen3.8-27B-FP8')
    r['template_id'] = r.pop('template')
    r['experiment'] = r.pop('arm')
    r['prefill_type'] = r.pop('prefill')
    selected = t.select_cell([r], r['model'])
    assert len(t.blind_rows(selected)[0]) == 1
    with pytest.raises(ValueError, match='conflicting'):
        t.select_cell([r | dict(template=1)], r['model'])


def test_protocol_diagnostic_does_not_exclude_registered_trace():
    original=row()
    diagnostic=original | dict(protocol_failure=True)
    assert t.blind_rows([diagnostic])==t.blind_rows([original])
