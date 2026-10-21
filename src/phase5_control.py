"""Phase 5 lifetime budget and already-rented Vast supervision. Never rents GPUs.

Parent contract: initialize once, reserve BEFORE rent/spawn, then bind_resource.
The CLI's Vast reservation starts a detached watchdog before returning; rent with
the returned resource_label so it can recover an uncertain rent. Use this same
local ledger for both providers. Do not copy it onto eventually consistent volumes.
Full generation has no authorization in this module.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / 'results/phase5/aggregate-ledger.json'
LEGACY_CAP_MICROUSD = 6_800_000
CAP_MICROUSD = 9_000_000
CAP_USD = 9.0
REGISTRATION_COMMIT = "c3e705068f0c3622c3b5d4cb9575530004ad68ae"
LEGACY_REGISTRATION_COMMIT = "f15dcc41904f6fad858bdab6c8b7daac09238520"
UNRESOLVED = {'reserved', 'active', 'recovered_unresolved'}


class BudgetError(RuntimeError):
    """Budget, authorization or resource ownership gate refused an operation."""


def positive(value, name, *, zero=False):
    if isinstance(value, bool):
        raise ValueError(f'{name}: boolean is not a measurement')
    result = float(value)
    if not math.isfinite(result) or result < 0 or (not zero and result == 0):
        raise ValueError(f'{name}: finite {"nonnegative" if zero else "positive"} value required')
    return result


def micro(value):
    positive(value, 'money', zero=True)
    return int((Decimal(str(value)) * 1_000_000).to_integral_value(rounding=ROUND_CEILING))


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', value):
        raise ValueError('safe, nonempty run/attempt ID required')
    return value


def atomic_json(path, value):
    """Flush file and directory; callers hold the separate, stable flock inode."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class BudgetLedger:
    """Atomic process-safe ledger. No reset, refund, arbitrary cap override or delete API.

    An outstanding reservation is already budget exposure. Recovery charges its
    entire amount permanently; resource absence must separately be verified.
    Settlement takes the maximum of explicit billing, observed wall-time at the
    reserved all-in rate, and any prior charge, including recovered reservations.
    """

    def __init__(self, path=LEDGER_PATH):
        self.path = Path(path).resolve()

    @classmethod
    def initialize(cls, path=LEDGER_PATH):
        ledger = cls(path)
        ledger.path.parent.mkdir(parents=True, exist_ok=True)
        with ledger._lock():
            if ledger.path.exists():
                ledger._read()
            else:
                atomic_json(ledger.path, {'schema': 1, 'cap_microusd': LEGACY_CAP_MICROUSD,
                                         'attempts': [], 'created_unix': time.time()})
        return ledger

    @contextmanager
    def _lock(self):
        if not self.path.parent.exists():
            raise BudgetError('ledger missing: explicitly initialize once before launch')
        with self.path.with_suffix(self.path.suffix + '.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    def _read(self):
        try:
            value = json.loads(self.path.read_text())
            if (type(value['schema']) is not int or value['schema'] != 1
                    or type(value['cap_microusd']) is not int
                    or value['cap_microusd'] not in (LEGACY_CAP_MICROUSD, CAP_MICROUSD)):
                raise ValueError('ledger schema/cap changed')
            upgrades = value.get('registration_upgrades', [])
            if value['cap_microusd'] == CAP_MICROUSD:
                if (value.get('registration_commit') != REGISTRATION_COMMIT
                        or not isinstance(upgrades, list) or len(upgrades) != 1):
                    raise ValueError('unregistered cap upgrade')
                change = upgrades[0]
                if (not isinstance(change, dict)
                        or change.get('old_cap_microusd') != LEGACY_CAP_MICROUSD
                        or change.get('new_cap_microusd') != CAP_MICROUSD
                        or change.get('old_registration_commit') != LEGACY_REGISTRATION_COMMIT
                        or change.get('registration_commit') != REGISTRATION_COMMIT):
                    raise ValueError('invalid registration upgrade provenance')
                positive(change['unix'], 'upgrade timestamp')
            elif upgrades or value.get('registration_commit') not in (None, LEGACY_REGISTRATION_COMMIT):
                raise ValueError('legacy cap has inconsistent registration history')
            if not isinstance(value['attempts'], list):
                raise ValueError('invalid attempt history')
            ids = set()
            for row in value['attempts']:
                if row['attempt_id'] in ids:
                    raise ValueError('duplicate attempt history')
                ids.add(safe_id(row['attempt_id']))
                if row['provider'] not in {'vast', 'modal'} or row['status'] not in UNRESOLVED | {'completed', 'failed'}:
                    raise ValueError('invalid provider/status')
                for key in ('reserved_microusd', 'charged_microusd'):
                    if type(row[key]) is not int or row[key] < 0:
                        raise ValueError('invalid stored charge')
                if row['reserved_microusd'] <= 0:
                    raise ValueError('invalid stored reservation')
                positive(row['hourly_usd'], 'rate')
                if row['deadline_unix'] <= row['started_unix']:
                    raise ValueError('invalid stored deadline')
            return value
        except FileNotFoundError as exc:
            raise BudgetError('ledger missing: explicitly initialize once before launch') from exc
        except (ValueError, KeyError, TypeError, AssertionError) as exc:
            raise BudgetError('invalid ledger; preserve it and reconcile manually, never reset') from exc

    def _mutate(self, operation):
        with self._lock():
            value = self._read()
            result = operation(value)
            atomic_json(self.path, value)
            return result

    @staticmethod
    def _find(value, attempt_id):
        for row in value['attempts']:
            if row['attempt_id'] == attempt_id:
                return row
        raise BudgetError(f'unknown reservation: {attempt_id}')

    def snapshot(self):
        with self._lock():
            value = self._read()
        spent = sum(a['charged_microusd'] for a in value['attempts'])
        outstanding = sum(a['reserved_microusd'] for a in value['attempts']
                          if a['status'] in {'reserved', 'active'})
        return {**value, 'spent_usd': spent / 1e6, 'outstanding_usd': outstanding / 1e6,
                'available_usd': (value['cap_microusd'] - spent - outstanding) / 1e6}

    def upgrade_registration(self, *, registration_commit):
        """Explicit amendment-1 migration; retain all attempts and charged spend.

        This records authorization only, never starts generation or resources.
        Reading/initializing a legacy ledger never invokes this method.
        """
        if registration_commit != REGISTRATION_COMMIT:
            raise ValueError('exact amendment registration_commit required')
        def update(value):
            if any(a['status'] in UNRESOLVED for a in value['attempts']):
                raise BudgetError('unresolved reservation/resource prevents registration upgrade')
            if value['cap_microusd'] == CAP_MICROUSD:
                return
            value['registration_upgrades'] = [dict(
                old_cap_microusd=value['cap_microusd'], new_cap_microusd=CAP_MICROUSD,
                old_registration_commit=LEGACY_REGISTRATION_COMMIT,
                registration_commit=registration_commit, unix=time.time())]
            value.update(cap_microusd=CAP_MICROUSD, registration_commit=registration_commit)
        self._mutate(update)
        return self.snapshot()

    def attempt(self, attempt_id):
        return self._find(self.snapshot(), attempt_id)

    def reserve(self, attempt_id, *, provider, hourly_usd, runtime_seconds=900,
                startup_seconds=120, teardown_seconds=60):
        safe_id(attempt_id)
        if provider not in {'vast', 'modal'}:
            raise ValueError('provider must be vast or modal')
        hourly = positive(hourly_usd, 'all-in hourly rate')
        runtime = positive(runtime_seconds, 'runtime')
        startup = positive(startup_seconds, 'startup')
        teardown = positive(teardown_seconds, 'teardown')
        # Rate must include GPU, fixed CPU/RAM, storage and any provider charges.
        reservation = micro(Decimal(str(hourly)) * Decimal(str(runtime + startup + teardown)) / 3600)
        def update(value):
            if any(a['attempt_id'] == attempt_id for a in value['attempts']):
                raise BudgetError('fresh attempt ID required, including every retry')
            if any(a['status'] in UNRESOLVED for a in value['attempts']):
                raise BudgetError('unresolved reservation/resource; teardown and recover before another launch')
            spent = sum(a['charged_microusd'] for a in value['attempts'])
            if spent + reservation > value['cap_microusd']:
                raise BudgetError(f"lifetime ${value['cap_microusd']/1e6:.2f} cap: startup/runtime/teardown reservation does not fit")
            start = time.time()
            row = dict(attempt_id=attempt_id, provider=provider, status='reserved',
                       reserved_microusd=reservation, reserved_usd=reservation / 1e6,
                       charged_microusd=0, hourly_usd=hourly, started_unix=start,
                       startup_seconds=startup, runtime_seconds=runtime, teardown_seconds=teardown,
                       startup_deadline_unix=start + startup,
                       work_deadline_unix=start + startup + runtime,
                       deadline_unix=start + startup + runtime + teardown,
                       resource_id=None, resource_label='phase5-' + attempt_id,
                       teardown_verified=False, events=[])
            value['attempts'].append(row)
            return row.copy()
        return self._mutate(update)

    def bind_resource(self, attempt_id, resource_id):
        def update(value):
            row = self._find(value, attempt_id)
            if row['status'] not in {'reserved', 'active'}:
                raise BudgetError('reservation is no longer launchable')
            if time.time() >= row['startup_deadline_unix']:
                raise BudgetError('bounded startup deadline exceeded')
            if row['resource_id'] not in (None, str(resource_id)):
                raise BudgetError('single reservation cannot bind multiple resources')
            row.update(resource_id=str(resource_id), status='active')
        self._mutate(update)

    def note(self, attempt_id, **evidence):
        def update(value):
            self._find(value, attempt_id)['events'].append({'unix': time.time(), **evidence})
        self._mutate(update)

    def recover(self):
        def update(value):
            for row in value['attempts']:
                if row['status'] in {'reserved', 'active'}:
                    wall = micro(Decimal(str(max(0, time.time() - row['started_unix']))) *
                                 Decimal(str(row['hourly_usd'])) / 3600)
                    row.update(status='recovered_unresolved', recovered_unix=time.time(),
                               charged_microusd=max(row['charged_microusd'], row['reserved_microusd'], wall))
        self._mutate(update)
        return self.snapshot()

    def verify_teardown(self, attempt_id, *, evidence):
        if not evidence:
            raise ValueError('concrete provider absence evidence required')
        def update(value):
            row = self._find(value, attempt_id)
            if row['status'] in {'reserved', 'active'}:
                raise BudgetError('settle or recover before verifying teardown')
            row.update(status='failed' if row['status'] == 'recovered_unresolved' else row['status'],
                       teardown_verified=True, teardown_evidence=evidence,
                       zero_verified_unix=time.time())
        self._mutate(update)

    def settle(self, attempt_id, *, actual_usd=0, teardown_verified=False, success=False):
        billed = micro(actual_usd)
        def update(value):
            row = self._find(value, attempt_id)
            already_zero = row['teardown_verified']
            end = row.get('zero_verified_unix', row.get('ended_unix', time.time())) if already_zero else time.time()
            elapsed = max(0, end - row['started_unix'])
            wall_charge = micro(Decimal(str(elapsed)) * Decimal(str(row['hourly_usd'])) / 3600)
            charge = max(billed, row['charged_microusd'], wall_charge)
            verified = bool(teardown_verified or already_zero)
            if not verified:
                charge = max(charge, row['reserved_microusd'])
            row.update(charged_microusd=charge, ended_unix=end,
                       observed_wall_seconds=elapsed, teardown_verified=verified,
                       status=('completed' if success or row['status'] == 'completed' else 'failed')
                              if verified else 'recovered_unresolved')
        self._mutate(update)
        result = self.snapshot()
        if micro(result['spent_usd']) > result['cap_microusd']:
            raise BudgetError('actual exposure exceeded cap; recorded permanently, no new launch allowed')
        return result


def registered_cells(second_model='nemotron'):
    """Sixteen 400-reply cells; GLM replaces the entire second model only."""
    if second_model not in ('nemotron', 'glm'):
        raise ValueError('second_model must be nemotron or glm')
    cells = []
    for model in ('gemma', second_model):
        cells += [dict(id=f'{model}/main/{thinking}/{request}', model=model, count=400)
                  for thinking in ('off', 'on') for request in ('plain', 'explicit')]
        cells += [dict(id=f'{model}/prefill/{arm}', model=model, count=400)
                  for arm in ('empty', 'reminder', 'neutral', 'irrelevant')]
    return cells


def project_plan(cell_means, tokens_per_second, hourly_usd, *, spent_usd=0,
                 outstanding_usd=0, startup_seconds=None, teardown_reserve_usd=.30,
                 cap_usd=None, second_model='nemotron', budget_snapshot=None):
    """Outcome-blind full-cell projection, 20% runtime margin and strict drops.

    Supply the canonical ledger snapshot to use its cap and all exposure. With
    no snapshot/cap, conservatively retain the legacy $6.80 cap. No unmeasured
    second-model throughput is inferred from the Gemma pilot.
    """
    if budget_snapshot is not None:
        if cap_usd is not None or spent_usd != 0 or outstanding_usd != 0:
            raise ValueError('budget_snapshot cannot be mixed with explicit budget inputs')
        cap_usd = budget_snapshot['cap_microusd'] / 1e6
        spent_usd, outstanding_usd = budget_snapshot['spent_usd'], budget_snapshot['outstanding_usd']
    cap = positive(LEGACY_CAP_MICROUSD/1e6 if cap_usd is None else cap_usd, 'cap')
    if micro(cap) not in (LEGACY_CAP_MICROUSD, CAP_MICROUSD) or Decimal(str(cap))*1_000_000 != micro(cap):
        raise ValueError('cap must be the registered $6.80 or $9.00 ledger cap')
    cells = registered_cells(second_model)
    if not isinstance(cell_means, dict) or set(cell_means) != {c['id'] for c in cells}:
        raise ValueError('exact registered cell means required, no pooled/missing/extra cells')
    means = {k: positive(v, f'cell mean {k}', zero=True) for k, v in cell_means.items()}
    if any(v > 4096 for v in means.values()):
        raise ValueError('registered cell means cannot exceed 4096 generated tokens')
    models = ('gemma', second_model)
    for measurement, name in ((tokens_per_second,'throughput'), (hourly_usd,'hourly rate')):
        if not isinstance(measurement, dict) or set(measurement) != set(models):
            raise ValueError(f'exact measured {name} required for both models')
    speeds = {m: positive(tokens_per_second[m], f'{m} throughput') for m in models}
    rates = {m: positive(hourly_usd[m], f'{m} hourly rate') for m in models}
    startup = {m:0 for m in models} if startup_seconds is None else startup_seconds
    if not isinstance(startup,dict) or set(startup) != set(models):
        raise ValueError('startup seconds required for both models')
    base = positive(spent_usd, 'spent', zero=True) + positive(outstanding_usd, 'outstanding', zero=True)
    reserve = positive(teardown_reserve_usd, 'teardown reserve')
    setup = sum(positive(startup[m], 'startup seconds', zero=True) * rates[m] / 3600 for m in models)
    dropped, projections = [], []
    groups = ((), ('gemma/prefill/neutral','gemma/prefill/irrelevant'),
              (f'{second_model}/prefill/neutral',f'{second_model}/prefill/irrelevant'),
              tuple(f'{second_model}/main/{t}/{r}' for t in ('off','on') for r in ('plain','explicit')))
    for stage,group in enumerate(groups):
        dropped.extend(group)
        kept = [c for c in cells if c['id'] not in dropped]
        seconds = {m:sum(c['count']*means[c['id']]/speeds[m] for c in kept if c['model']==m) for m in models}
        total = base + setup + 1.20*sum(seconds[m]*rates[m]/3600 for m in models) + reserve
        projections.append(dict(drop_stage=stage, dropped_cells=dropped.copy(), projected_total_usd=total))
        if total <= cap:
            return dict(cells=kept, dropped_cells=dropped.copy(), drop_stage=stage,
                cap_usd=cap, projected_total_usd=total, generation_seconds_by_model=seconds,
                remaining_runtime_multiplier=1.20, mean_output_tokens_by_cell=means,
                tokens_per_second=speeds, hourly_usd=rates, startup_seconds=startup,
                previous_exposure_usd=base, teardown_reserve_usd=reserve,
                projections=projections, pilot_rows_discarded=True)
    raise BudgetError(f'minimum registered whole-cell grid does not fit lifetime ${cap:.2f} cap')


def require_pilot(full=False):
    if full:
        raise BudgetError('full run requires explicit new authorization and a separately reviewed gate')


def watchdog_deadline(attempt):
    started = any(event.get('startup_complete') is True for event in attempt['events'])
    return attempt['work_deadline_unix'] if started else attempt['startup_deadline_unix']


class VastCLI:
    """Only reads, copies, and destroys an already-rented, explicitly bound host."""

    def __init__(self, instance, host, port, remote_dir, output_dir, *,
                 startup_marker='progress.json', completion_marker='metrics.json'):
        self.instance = int(instance)
        if not re.fullmatch(r'[A-Za-z0-9.-]+', host):
            raise ValueError('invalid SSH host')
        self.host, self.port = host, int(port)
        if not 0 < self.port < 65536:
            raise ValueError('invalid SSH port')
        self.remote_dir, self.output_dir = remote_dir, Path(output_dir)
        self.operation_deadline = None
        if not remote_dir.startswith('/'):
            raise ValueError('remote output must be an absolute path')
        self.startup_marker, self.completion_marker = startup_marker, completion_marker

    @staticmethod
    def run(argv, *, timeout=30, **kwargs):
        return subprocess.run(argv, check=True, capture_output=True, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL, **kwargs).stdout

    def _timeout(self, limit):
        return limit if self.operation_deadline is None else max(.1, min(limit, self.operation_deadline - time.time()))

    def instances(self):
        return json.loads(self.run(['vastai', 'show', 'instances', '--raw'], timeout=self._timeout(10)))

    def status(self):
        code = ('import json,pathlib; d=pathlib.Path(' + repr(self.remote_dir) + '); '
                'print(json.dumps({"started":(d/' + repr(self.startup_marker) + ').exists(),'
                '"done":(d/' + repr(self.completion_marker) + ').exists()}))')
        return json.loads(self.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                                    '-p', str(self.port), 'root@' + self.host, 'python3 -c ' + shlex.quote(code)],
                                   timeout=self._timeout(15)))

    def pull(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Pull partial artifacts on error too; rsync is unnecessary on the host.
        self.run(['scp', '-r', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                  '-P', str(self.port), 'root@' + self.host + ':' + shlex.quote(self.remote_dir.rstrip('/') + '/.'),
                  str(self.output_dir)], timeout=self._timeout(30))

    def destroy(self):
        self.run(['vastai', 'destroy', 'instance', str(self.instance)], timeout=self._timeout(10))


def monitor_vast(ledger, attempt_id, backend, *, poll_seconds=2):
    """Supervise existing generation; pull immediately, destroy in finally.

    CLI reserve arms a detached watchdog beforehand. No generation or rental is
    submitted here. Partial or failed runs never become completed-run evidence.
    """
    attempt = ledger.attempt(attempt_id)
    caught = None
    done = pulled = zero = False
    startup_recorded = False
    try:
        if attempt['provider'] != 'vast' or not attempt['resource_id']:
            raise BudgetError('a bound Vast reservation is required')
        if attempt['status'] not in {'reserved', 'active'}:
            raise BudgetError('reservation is no longer active')
        if isinstance(backend, VastCLI):
            backend.operation_deadline = attempt['deadline_unix']
            ledger.note(attempt_id, watchdog_recovery=dict(host=backend.host, port=backend.port,
                        remote_dir=backend.remote_dir, output_dir=str(backend.output_dir),
                        startup_marker=backend.startup_marker, completion_marker=backend.completion_marker))
        while True:
            instances = backend.instances()
            if len(instances) != 1 or str(instances[0]['id']) != attempt['resource_id']:
                raise BudgetError('exactly one owned Vast instance required')
            if instances[0].get('num_gpus') != 1:
                raise BudgetError('exactly one GPU on the Vast instance required')
            state = backend.status()
            now = time.time()
            if state['started'] and not startup_recorded:
                ledger.note(attempt_id, startup_complete=True)
                startup_recorded = True
            if not state['started'] and now >= attempt['startup_deadline_unix']:
                raise TimeoutError('bounded Vast startup deadline exceeded')
            if now >= attempt['work_deadline_unix']:
                raise TimeoutError('Vast absolute work deadline; preserving teardown reserve')
            if state['done']:
                done = True
                break
            time.sleep(poll_seconds)
    except BaseException as error:
        caught = error
    finally:
        try:
            backend.pull()
            pulled = True
        except BaseException as error:
            caught = caught or error
        finally:
            # Failed copy/SSH must never prevent destruction. Retry within the
            # reserved teardown interval; detached watchdog remains independent.
            last_error = None
            for _ in range(3):
                try:
                    backend.destroy()
                except Exception as error:
                    last_error = str(error)
                try:
                    after = backend.instances()
                    zero = not any(str(i['id']) == attempt['resource_id'] for i in after)
                    if zero:
                        ledger.note(attempt_id, instances_after=after, target_absent=True)
                        break
                except Exception as error:
                    last_error = str(error)
                if time.time() >= attempt['deadline_unix']:
                    break
            ledger.note(attempt_id, pulled=pulled, done_marker=done, destroy_error=last_error,
                        error=None if caught is None else str(caught))
            ledger.settle(attempt_id, teardown_verified=zero, success=done and pulled and caught is None)
    if caught is not None:
        raise caught
    if not zero:
        raise BudgetError('Vast teardown unresolved; reservation fully charged and next launch blocked')
    return dict(status='completed', artifacts_pulled=pulled, target_absent=zero)


def arm_vast_watchdog(ledger, attempt_id):
    """Start before rental; rent with resource_label to cover a lost rent response."""
    attempt = ledger.attempt(attempt_id)
    if attempt['provider'] != 'vast':
        raise ValueError('Vast watchdog requires Vast reservation')
    log = ledger.path.parent / (safe_id(attempt_id) + '-watchdog.log')
    with log.open('ab') as handle:
        child = subprocess.Popen([sys.executable, '-m', 'src.phase5_control', 'watchdog',
                                  '--attempt-id', attempt_id, '--ledger', str(ledger.path)],
                                 cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=handle,
                                 start_new_session=True, close_fds=True)
    ledger.note(attempt_id, watchdog_pid=child.pid, watchdog_log=str(log))
    return child.pid


def vast_watchdog(ledger, attempt_id):
    """Independent final kill switch; never destroys resources outside this ticket."""
    while True:
        attempt = ledger.attempt(attempt_id)
        if attempt['teardown_verified']:
            return
        deadline = watchdog_deadline(attempt)
        if time.time() < deadline:
            time.sleep(min(1, deadline - time.time()))
            continue
        try:
            rows = json.loads(VastCLI.run(['vastai', 'show', 'instances', '--raw']))
            owned = [r for r in rows if (str(r['id']) == attempt['resource_id'] if attempt['resource_id']
                                        else r.get('label') == attempt['resource_label'])]
            try:
                recovery = next((event['watchdog_recovery'] for event in reversed(attempt['events'])
                                 if 'watchdog_recovery' in event), None)
                if recovery and len(owned) == 1:
                    VastCLI(owned[0]['id'], **recovery).pull()
            except Exception as error:
                print(f'WATCHDOG_ARTIFACT_RECOVERY_ERROR {error}', flush=True)
            finally:
                for row in owned:
                    VastCLI.run(['vastai', 'destroy', 'instance', str(row['id'])])
            after = json.loads(VastCLI.run(['vastai', 'show', 'instances', '--raw']))
            zero = not any((str(r['id']) == attempt['resource_id'] if attempt['resource_id']
                            else r.get('label') == attempt['resource_label']) for r in after)
            ledger.note(attempt_id, watchdog=True, instances_after=after)
            ledger.settle(attempt_id, actual_usd=attempt['reserved_usd'], teardown_verified=zero)
            if zero:
                return
        except Exception as error:
            print(f'WATCHDOG_RETRY {error}', flush=True)
        time.sleep(2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true')
    commands = parser.add_subparsers(dest='command')
    for name in ('init', 'status', 'recover'):
        commands.add_parser(name)
    upgrade = commands.add_parser('upgrade-registration')
    upgrade.add_argument('--registration-commit', required=True, choices=[REGISTRATION_COMMIT])
    upgrade.add_argument('--ledger', default=str(LEDGER_PATH))
    reserve_parser = commands.add_parser('reserve')
    reserve_parser.add_argument('--attempt-id', required=True)
    reserve_parser.add_argument('--provider', choices=['vast', 'modal'], required=True)
    reserve_parser.add_argument('--hourly-usd', type=float, required=True)
    reserve_parser.add_argument('--runtime-seconds', type=float, default=900)
    reserve_parser.add_argument('--startup-seconds', type=float, default=120)
    reserve_parser.add_argument('--teardown-seconds', type=float, default=60)
    bind = commands.add_parser('bind')
    bind.add_argument('--attempt-id', required=True)
    bind.add_argument('--resource-id', required=True)
    for name in ('watch-vast', 'watchdog'):
        sub = commands.add_parser(name)
        sub.add_argument('--attempt-id', required=True)
        if name == 'watchdog':
            sub.add_argument('--ledger', default=str(LEDGER_PATH), help=argparse.SUPPRESS)
        else:
            sub.add_argument('--instance', type=int, required=True)
            sub.add_argument('--host', required=True)
            sub.add_argument('--port', type=int, required=True)
            sub.add_argument('--remote-dir', required=True)
            sub.add_argument('--output-dir', required=True)
            sub.add_argument('--startup-marker', default='progress.json')
            sub.add_argument('--completion-marker', default='metrics.json')
    args = parser.parse_args(argv)
    require_pilot(args.full)
    ledger = BudgetLedger(getattr(args, 'ledger', LEDGER_PATH))
    if args.command == 'init':
        result = BudgetLedger.initialize().snapshot()
    elif args.command == 'upgrade-registration':
        result = ledger.upgrade_registration(registration_commit=args.registration_commit)
    elif args.command == 'status' or args.command is None:
        result = ledger.snapshot()
    elif args.command == 'recover':
        result = ledger.recover()
    elif args.command == 'reserve':
        result = ledger.reserve(args.attempt_id, provider=args.provider, hourly_usd=args.hourly_usd,
                                runtime_seconds=args.runtime_seconds, startup_seconds=args.startup_seconds,
                                teardown_seconds=args.teardown_seconds)
        if args.provider == 'vast':
            arm_vast_watchdog(ledger, args.attempt_id)
    elif args.command == 'bind':
        ledger.bind_resource(args.attempt_id, args.resource_id)
        result = ledger.attempt(args.attempt_id)
    elif args.command == 'watchdog':
        return vast_watchdog(ledger, args.attempt_id)
    else:
        attempt = ledger.attempt(args.attempt_id)
        if attempt['resource_id'] != str(args.instance):
            raise BudgetError('instance does not match reservation')
        backend = VastCLI(args.instance, args.host, args.port, args.remote_dir, args.output_dir,
                          startup_marker=args.startup_marker, completion_marker=args.completion_marker)
        result = monitor_vast(ledger, args.attempt_id, backend)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
