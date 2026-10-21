"""Offline safety gates; these tests must never contact a compute provider."""
from concurrent.futures import ThreadPoolExecutor
import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.phase5_control import BudgetError, BudgetLedger, project_plan, registered_cells, monitor_vast


@pytest.fixture
def ledger(tmp_path):
    return BudgetLedger.initialize(tmp_path / 'ledger.json')


def reserve(ledger, name='pilot', **kwargs):
    return ledger.reserve(name, provider='vast', hourly_usd=1, runtime_seconds=60,
                          startup_seconds=60, teardown_seconds=60, **kwargs)


def test_missing_or_corrupt_ledger_never_resets_spend(tmp_path):
    path = tmp_path / 'ledger.json'
    with pytest.raises(BudgetError, match='initialize'):
        BudgetLedger(path).snapshot()
    path.write_text('{broken')
    with pytest.raises(BudgetError):
        BudgetLedger.initialize(path)
    assert path.read_text() == '{broken'


def test_valid_json_with_changed_cap_or_charge_is_rejected(ledger):
    reserve(ledger)
    document = json.loads(ledger.path.read_text())
    document['attempts'][0]['charged_microusd'] = -1
    ledger.path.write_text(json.dumps(document))
    with pytest.raises(BudgetError, match='invalid ledger'):
        ledger.snapshot()


def test_failed_atomic_replace_preserves_previous_ledger(ledger, monkeypatch):
    import src.phase5_control as control
    before = ledger.path.read_bytes()
    def fail(*args):
        raise OSError('simulated filesystem error')
    monkeypatch.setattr(control.os, 'replace', fail)
    with pytest.raises(OSError, match='filesystem'):
        reserve(ledger)
    assert ledger.path.read_bytes() == before
    assert ledger.snapshot()['attempts'] == []


def test_initialize_is_idempotent_and_lifetime_spend_survives(ledger):
    attempt = reserve(ledger)
    ledger.settle('pilot', actual_usd='.10', teardown_verified=True)
    again = BudgetLedger.initialize(ledger.path)
    assert again.snapshot()['spent_usd'] >= .10
    with pytest.raises(BudgetError, match='fresh'):
        reserve(again)
    assert attempt['reserved_usd'] == pytest.approx(.05)


def test_cap_includes_failed_attempts_and_retries(ledger):
    ledger.reserve('first', provider='modal', hourly_usd=10, runtime_seconds=1800,
                   startup_seconds=60, teardown_seconds=60)
    ledger.recover()
    ledger.verify_teardown('first', evidence={'zero_containers': True})
    with pytest.raises(BudgetError, match='cap'):
        ledger.reserve('retry', provider='vast', hourly_usd=10, runtime_seconds=600,
                       startup_seconds=60, teardown_seconds=60)


def test_recovery_charges_full_reservation_and_keeps_resource_block(ledger):
    attempt = reserve(ledger)
    ledger.bind_resource('pilot', '123')
    ledger.recover()
    snapshot = ledger.snapshot()
    assert snapshot['spent_usd'] == attempt['reserved_usd']
    assert snapshot['outstanding_usd'] == 0
    with pytest.raises(BudgetError, match='unresolved'):
        reserve(ledger, 'next')
    ledger.verify_teardown('pilot', evidence={'instances_after': []})
    ledger.settle('pilot', actual_usd=0, teardown_verified=True)
    assert ledger.snapshot()['spent_usd'] == attempt['reserved_usd']
    reserve(ledger, 'next')


def test_uncertain_teardown_charges_and_blocks(ledger):
    attempt = reserve(ledger)
    ledger.settle('pilot', actual_usd=0, teardown_verified=False)
    assert ledger.snapshot()['spent_usd'] == attempt['reserved_usd']
    with pytest.raises(BudgetError):
        reserve(ledger, 'retry')


def test_concurrent_reservations_have_one_winner(ledger):
    def try_reserve(i):
        try:
            reserve(BudgetLedger(ledger.path), str(i))
            return True
        except BudgetError:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(try_reserve, range(16))) == 1
    assert len(json.loads(ledger.path.read_text())['attempts']) == 1


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), True])
def test_invalid_money_rejected(ledger, value):
    with pytest.raises((ValueError, BudgetError)):
        ledger.reserve('bad', provider='vast', hourly_usd=value, runtime_seconds=60)


def test_settlement_cannot_be_repeated_with_lower_charge(ledger):
    reserve(ledger)
    ledger.settle('pilot', actual_usd='.25', teardown_verified=True)
    ledger.settle('pilot', actual_usd=0, teardown_verified=True)
    assert ledger.snapshot()['spent_usd'] >= .25


def test_terminal_settlement_does_not_charge_idle_days(ledger, monkeypatch):
    import src.phase5_control as control
    now = [1000.]
    monkeypatch.setattr(control.time, 'time', lambda: now[0])
    reserve(ledger)
    now[0] += 10
    ledger.settle('pilot', teardown_verified=True)
    charged = ledger.snapshot()['spent_usd']
    now[0] += 86400
    ledger.settle('pilot', teardown_verified=True)
    assert ledger.snapshot()['spent_usd'] == charged


def test_recovery_reconciles_only_attempt_and_keeps_charge(ledger):
    reserve(ledger)
    ledger.recover()
    ledger.verify_teardown('pilot', evidence={'instances_after': []})
    with pytest.raises(BudgetError):
        ledger.bind_resource('pilot', 'new-resource')


def test_projection_rejects_unreasonable_cell_token_means():
    means = {c['id']: 4097 for c in registered_cells()}
    with pytest.raises(ValueError, match='4096'):
        project_plan(means, {'gemma': 100000, 'nemotron': 100000}, {'gemma': 1, 'nemotron': 1})


def test_projection_uses_cell_specific_means_and_registered_group_drops():
    cells = registered_cells()
    assert len(cells) == 16 and sum(c['count'] for c in cells) == 6400
    means = {c['id']: 100 for c in cells}
    rates = {'gemma': 3600, 'nemotron': 3600}
    speeds = {'gemma': 100000, 'nemotron': 100000}
    expected = [[], ['gemma/prefill/neutral', 'gemma/prefill/irrelevant'],
        ['gemma/prefill/neutral', 'gemma/prefill/irrelevant',
         'nemotron/prefill/neutral', 'nemotron/prefill/irrelevant']]
    expected.append(expected[-1] + [f'nemotron/main/{t}/{r}'
        for t in ('off','on') for r in ('plain','explicit')])
    for spent, count, dropped in zip((0,1.5,2.5,4), (6400,5600,4800,3200), expected):
        plan = project_plan(means, speeds, rates, spent_usd=spent, cap_usd=9)
        assert plan['dropped_cells'] == dropped
        assert sum(c['count'] for c in plan['cells']) == count
        assert all(c['count'] == 400 for c in plan['cells'])
        assert plan['projected_total_usd'] <= 9
    with pytest.raises(BudgetError, match='fit'):
        project_plan(means, speeds, rates, spent_usd=5, cap_usd=9)


def test_projection_refuses_missing_cells_and_unaffordable_minimum():
    means = {c['id']: 4096 for c in registered_cells()}
    with pytest.raises(BudgetError, match='fit'):
        project_plan(means, {'gemma': 1, 'nemotron': 1}, {'gemma': 1, 'nemotron': 1})
    means.pop(next(iter(means)))
    with pytest.raises(ValueError, match='cell'):
        project_plan(means, {'gemma': 1000, 'nemotron': 1000}, {'gemma': 1, 'nemotron': 1})


class FakeVast:
    def __init__(self, *, pull_error=False, status_error=False, extra=False):
        self.events = []
        self.pull_error, self.status_error, self.extra = pull_error, status_error, extra
        self.destroyed = False

    def instances(self):
        if self.destroyed:
            return []
        return [{'id': 123, 'num_gpus': 1}] + ([{'id': 999}] if self.extra else [])

    def status(self):
        self.events.append('status')
        if self.status_error:
            raise RuntimeError('ssh broke')
        return {'started': True, 'done': True}

    def pull(self):
        self.events.append('pull')
        if self.pull_error:
            raise RuntimeError('pull broke')

    def destroy(self):
        self.events.append('destroy')
        self.destroyed = True


@pytest.mark.parametrize('failure', ['none', 'pull', 'status', 'extra'])
def test_vast_always_pulls_before_finally_destroy(ledger, failure):
    reserve(ledger)
    ledger.bind_resource('pilot', '123')
    backend = FakeVast(pull_error=failure == 'pull', status_error=failure == 'status', extra=failure == 'extra')
    if failure == 'none':
        monitor_vast(ledger, 'pilot', backend, poll_seconds=0)
    else:
        with pytest.raises((BudgetError, RuntimeError)):
            monitor_vast(ledger, 'pilot', backend, poll_seconds=0)
    assert backend.events.index('pull') < backend.events.index('destroy')
    assert ledger.snapshot()['attempts'][0]['teardown_verified']


@pytest.mark.parametrize('kind', ['startup', 'runtime'])
def test_vast_deadlines_pull_and_destroy(ledger, monkeypatch, kind):
    import src.phase5_control as control
    clock = [1000.]
    monkeypatch.setattr(control.time, 'time', lambda: clock[0])
    attempt = reserve(ledger)
    ledger.bind_resource('pilot', '123')
    backend = FakeVast()
    backend.status = lambda: {'started': kind != 'startup', 'done': False}
    clock[0] = attempt['startup_deadline_unix'] if kind == 'startup' else attempt['work_deadline_unix']
    with pytest.raises(TimeoutError, match='deadline'):
        monitor_vast(ledger, 'pilot', backend, poll_seconds=0)
    assert backend.events == ['pull', 'destroy']
    assert ledger.attempt('pilot')['teardown_verified']


def test_vast_teardown_failure_blocks_next_paid_attempt(ledger, monkeypatch):
    reserve(ledger)
    ledger.bind_resource('pilot', '123')
    backend = FakeVast()
    def fail_destroy():
        raise RuntimeError('provider unavailable')
    backend.destroy = fail_destroy
    with pytest.raises(BudgetError, match='teardown unresolved'):
        monitor_vast(ledger, 'pilot', backend, poll_seconds=0)
    assert ledger.attempt('pilot')['status'] == 'recovered_unresolved'
    with pytest.raises(BudgetError):
        reserve(ledger, 'retry')


def test_watchdog_uses_startup_deadline_until_worker_has_started(ledger, monkeypatch):
    import src.phase5_control as control
    now = [1000.]
    monkeypatch.setattr(control.time, 'time', lambda: now[0])
    ticket = reserve(ledger)
    ledger.bind_resource('pilot', '123')
    assert control.watchdog_deadline(ledger.attempt('pilot')) == ticket['startup_deadline_unix']
    ledger.note('pilot', startup_complete=True)
    assert control.watchdog_deadline(ledger.attempt('pilot')) == ticket['work_deadline_unix']


def test_full_cli_refused_without_provisioning():
    result = subprocess.run([sys.executable, '-m', 'src.phase5_control', '--full'],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'authorization' in result.stderr


def test_modal_dry_run_has_no_ledger_or_provider_side_effects(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'src/modal_app_5.py'
    result = subprocess.run([sys.executable, str(path), '--dry-run'], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload['pilot_only'] is True
    assert payload['gpu'] == 'L40S'
    assert payload['timeout_seconds'] <= 900
    assert not list(tmp_path.iterdir())


def test_modal_definition_is_single_use_and_no_warm_pool():
    source = Path('src/modal_app_5.py').read_text()
    tree = ast.parse(source)
    decorators = [d for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  for d in n.decorator_list if isinstance(d, ast.Call)
                  and isinstance(d.func, ast.Attribute) and d.func.attr == 'function']
    assert len(decorators) == 1
    values = {k.arg: ast.literal_eval(k.value) for k in decorators[0].keywords
              if isinstance(k.value, ast.Constant)}
    assert values['gpu'] == 'L40S'
    assert values['timeout'] <= 900
    assert values['min_containers'] == values['buffer_containers'] == 0
    assert values['max_containers'] == 1
    assert values['single_use_containers'] is True
    assert values['retries'] == 0
    assert 'vllm==0.32.0' in source


def test_modal_generation_preserves_exception_artifacts_without_gpu(tmp_path):
    from src.modal_app_5 import run_generation
    import time
    commits = []
    directory = tmp_path / 'failed'
    command = [sys.executable, '-c', 'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text("partial"); raise RuntimeError("fixture failure")',
               str(directory / 'partial.jsonl')]
    result = run_generation(directory, 'failed', time.time() + 10,
                            lambda: commits.append(True), command=command)
    assert result['status'] == 'failed'
    assert (directory / 'partial.jsonl').read_text() == 'partial'
    assert 'fixture failure' in (directory / 'generation.log').read_text()
    assert json.loads((directory / 'pilot-execution.json').read_text())['status'] == 'failed'
    assert len(commits) >= 2


def test_modal_generation_kills_stuck_child_and_commits(tmp_path):
    from src.modal_app_5 import run_generation
    import time
    commits = []
    result = run_generation(tmp_path / 'timeout', 'timeout', time.time() + .1,
                            lambda: commits.append(True), command=[sys.executable, '-c', 'import time; time.sleep(60)'])
    assert result['status'] == 'failed'
    assert 'TimeoutError' in result['error']
    assert len(commits) >= 2


def test_modal_generation_returns_pilot_metrics(tmp_path):
    from src.modal_app_5 import run_generation
    import time
    directory = tmp_path / 'success'
    command = [sys.executable, '-c', 'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(\'{"output_tokens":100,"generation_seconds":2}\')',
               str(directory / 'gemma.metrics.json')]
    result = run_generation(directory, 'success', time.time() + 10, lambda: None, command=command)
    assert result['status'] == 'completed'
    assert result['metrics']['gemma.metrics.json']['output_tokens'] == 100
    assert 'generation.log' in result['artifact_paths']


@pytest.mark.parametrize('failure', ['none', 'remote', 'pull', 'spawn'])
def test_modal_cpu_entrypoint_one_spawn_pulls_then_always_stops(tmp_path, monkeypatch, failure):
    import src.modal_app_5 as runner
    from contextlib import contextmanager, nullcontext
    from types import SimpleNamespace

    events = []
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src/vast_generate_5.py').write_text('# CPU fixture adapter')
    ledger = BudgetLedger.initialize(tmp_path / 'ledger.json')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'LEDGER_PATH', ledger.path)

    @contextmanager
    def app_context():
        events.append('app-start')
        try:
            yield
        finally:
            events.append('app-exit')

    monkeypatch.setattr(runner, 'app', SimpleNamespace(app_id='ap-fixture', run=app_context))
    monkeypatch.setattr(runner.modal, 'enable_output', nullcontext)
    monkeypatch.setattr(runner.modal.Dict, 'ephemeral', lambda: nullcontext({'state': {'started': True}}))
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **kw: SimpleNamespace(pid=999))

    class FakeCall:
        object_id = 'fc-fixture'
        def get(self, timeout):
            return {'status': 'failed' if failure == 'remote' else 'completed'}
        def cancel(self, terminate_containers):
            assert terminate_containers
            events.append('cancel')

    def spawn(*args):
        events.append('spawn')
        if failure == 'spawn':
            raise RuntimeError('uncertain spawn')
        return FakeCall()
    monkeypatch.setattr(runner, 'pilot', SimpleNamespace(spawn=spawn))

    def cli(*args, **kwargs):
        assert args[:2] == ('volume', 'get')
        events.append('pull')
        if failure == 'pull':
            raise RuntimeError('copy failed')
    monkeypatch.setattr(runner, 'modal_cli', cli)

    def stop(app_id):
        assert app_id == 'ap-fixture'
        events.append('stop')
        return True, {'containers_after': []}
    monkeypatch.setattr(runner, 'stop_and_verify', stop)
    if failure == 'none':
        assert runner.execute_pilot('cpu-pilot', ledger=ledger)['status'] == 'completed'
    else:
        with pytest.raises(RuntimeError):
            runner.execute_pilot('cpu-pilot', ledger=ledger)
    assert events.count('spawn') == 1
    assert events.index('pull') < events.index('stop') < events.index('app-exit')
    assert ledger.attempt('cpu-pilot')['teardown_verified'] is True
    assert (tmp_path / 'results/raw/cpu-pilot/local-execution.json').exists()


def test_registration_upgrade_keeps_old_cap_and_every_attempt_until_explicit(tmp_path):
    from src.phase5_control import REGISTRATION_COMMIT
    path = tmp_path/'aggregate-ledger.json'
    legacy = {'schema':1,'cap_microusd':6800000,'attempts':[], 'created_unix':1}
    path.write_text(json.dumps(legacy))
    ledger = BudgetLedger.initialize(path)
    assert ledger.snapshot()['available_usd'] == 6.8
    reserve(ledger)
    ledger.settle('pilot', actual_usd='.282284',teardown_verified=True)
    before = ledger.snapshot()
    BudgetLedger.initialize(path)
    assert ledger.snapshot()['cap_microusd'] == 6800000
    result = ledger.upgrade_registration(registration_commit=REGISTRATION_COMMIT)
    assert result['cap_microusd'] == 9000000
    assert result['attempts'] == before['attempts']
    assert result['spent_usd'] == .282284
    assert result['available_usd'] == 9 - .282284
    change = result['registration_upgrades'][0]
    assert (change['old_cap_microusd'],change['new_cap_microusd']) == (6800000,9000000)
    assert change['registration_commit'] == REGISTRATION_COMMIT
    assert ledger.upgrade_registration(registration_commit=REGISTRATION_COMMIT) == result
    with pytest.raises(ValueError,match='registration'):
        ledger.upgrade_registration(registration_commit='wrong')


def test_migration_refuses_unresolved_or_forged_cap(ledger):
    from src.phase5_control import REGISTRATION_COMMIT
    reserve(ledger)
    with pytest.raises(BudgetError,match='unresolved'):
        ledger.upgrade_registration(registration_commit=REGISTRATION_COMMIT)
    document=json.loads(ledger.path.read_text())
    document['cap_microusd']=9000000
    document.pop('registration_commit',None)
    document.pop('registration_upgrades',None)
    ledger.path.write_text(json.dumps(document))
    with pytest.raises(BudgetError,match='invalid ledger'):
        ledger.snapshot()


def test_projection_requires_both_measured_throughputs_and_ledger_cap():
    means = {c['id']:100 for c in registered_cells()}
    with pytest.raises(ValueError,match='throughput'):
        project_plan(means,{'gemma':100}, {'gemma':1,'nemotron':1})
    speeds={'gemma':100000,'nemotron':100000}
    rates={'gemma':3600,'nemotron':3600}
    old=project_plan(means,speeds,rates,cap_usd=6.8)
    new=project_plan(means,speeds,rates,cap_usd=9)
    assert len(old['cells']) < len(new['cells'])
    with pytest.raises(ValueError,match='cap'):
        project_plan(means,speeds,rates,cap_usd=9.01)


def test_cli_upgrade_is_explicit_and_no_other_writes(tmp_path):
    from src.phase5_control import REGISTRATION_COMMIT
    path=tmp_path/'aggregate-ledger.json'
    ledger=BudgetLedger.initialize(path)
    before=path.read_bytes()
    result=subprocess.run([sys.executable,'-m','src.phase5_control','upgrade-registration',
        '--ledger',str(path),'--registration-commit','bad'],capture_output=True,text=True)
    assert result.returncode and path.read_bytes()==before
    result=subprocess.run([sys.executable,'-m','src.phase5_control','upgrade-registration',
        '--ledger',str(path),'--registration-commit',REGISTRATION_COMMIT],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['cap_microusd']==9000000
    ledger.reserve('fits-new-cap',provider='modal',hourly_usd=1,runtime_seconds=8*3600,
        startup_seconds=1,teardown_seconds=1)
    assert ledger.snapshot()['outstanding_usd']>8


def test_projection_snapshot_uses_migrated_cap_and_retains_spend(ledger):
    from src.phase5_control import REGISTRATION_COMMIT
    reserve(ledger)
    ledger.settle('pilot',actual_usd='.282284',teardown_verified=True)
    means={c['id']:100 for c in registered_cells()}
    speeds={'gemma':100000,'nemotron':100000};rates={'gemma':3600,'nemotron':3600}
    old=project_plan(means,speeds,rates,budget_snapshot=ledger.snapshot())
    ledger.upgrade_registration(registration_commit=REGISTRATION_COMMIT)
    new=project_plan(means,speeds,rates,budget_snapshot=ledger.snapshot())
    assert old['cap_usd']==6.8 and new['cap_usd']==9
    assert old['previous_exposure_usd']==new['previous_exposure_usd']==.282284
    assert len(old['cells'])<len(new['cells'])
