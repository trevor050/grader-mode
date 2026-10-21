"""Guarded Phase 5 second-family controller. Default/--prepare is entirely offline.

PREPARATION IS NOT EXECUTION AUTHORIZATION. A later, separate authorization must
name --execute --authorization-id <new-task-id> --gemma-analysis <completed.json>.
No score/analyze module is imported. Only output lengths/timing reach decisions.

A rental ticket is reserved and a detached watchdog armed BEFORE create. Each
failed smoke is pulled and destroyed, then (only for a runtime failure) one
Nemotron fix is permitted; two documented runtime failures prospectively select
GLM. Successful workers retain one LLM across smoke, throughput and main stages.
Exact pinned generation stays in vast_generate_5.py, which this module never edits.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time

from src.models_5 import NEMOTRON, GLM, PACKAGE_VERSIONS, REGISTRATION_COMMIT, model_spec
from src.phase5_control import (BudgetError, BudgetLedger, LEDGER_PATH, ROOT, UNRESOLVED,
                               atomic_json, positive, safe_id)

RESERVE_USD = .30
CREDIT_CEILING = 2.824799
IMAGE = 'vastai/pytorch:latest'
DISK_GB = 80
DOWNLOAD_GB = 100
UPLOAD_GB = 8
BATCH_SIZE = 256
MEASUREMENT_SECONDS = 900
STAGES = ('h20_core', 'h21_main', 'h20_controls')
REMOTE_ROOT = '/phase5'


def cell_id(row):
    return (f"prefill/{row['prefill']}" if row['arm'] == 'prefill' else
            f"main/{'on' if row['thinking'] else 'off'}/{row['request_type']}")


def cell_ids():
    return tuple(f'prefill/{p}' for p in ('empty', 'reminder', 'neutral', 'irrelevant')) + tuple(
        f'main/{t}/{r}' for t in ('off', 'on') for r in ('plain', 'explicit'))


def stage_rows(rows):
    """Reject partial input before any stage is released; no shortened cells."""
    from src.vast_generate_5 import _validate_requests
    if not rows or len({r['model'] for r in rows}) != 1 or rows[0]['model'] not in (NEMOTRON, GLM):
        raise ValueError('complete second-family grid required')
    _validate_requests(rows, rows[0]['model'])
    counts = Counter(cell_id(r) for r in rows)
    if (counts != Counter({key: 400 for key in cell_ids()})
            or len({r['problem_idx'] for r in rows}) != 200):
        raise ValueError('complete 200-problem, 400-row-per-cell grid required')
    result = {name: [] for name in STAGES}
    for row in rows:
        name = ('h21_main' if row['arm'] == 'main' else
                'h20_core' if row['prefill'] in ('empty', 'reminder') else 'h20_controls')
        result[name].append(row)
    return result


def balanced_pilot(rows, count):
    """Eight unscored runtime prompts or one full, balanced 256-row batch."""
    if count not in (8, BATCH_SIZE):
        raise ValueError('only 8 smoke or 256 throughput prompts')
    grouped = {k: [r for r in rows if cell_id(r) == k] for k in cell_ids()}
    if any(len(v) < count // 8 for v in grouped.values()):
        raise ValueError('complete cells required for balanced pilot')
    # Construction order alternates templates within each problem.
    return [r for group in grouped.values() for r in group[:count // 8]]


def measurement_summary(metrics, *, minimum=256):
    """Use measured end-to-end batch wall time, never compliance or parser fields."""
    count = metrics['requests']
    if type(count) is not int or count < minimum or count % 8:
        raise ValueError(f'at least {minimum} balanced measured requests required')
    if (metrics.get('batch_size') not in (64, 128, BATCH_SIZE)
            or metrics.get('chunk_size') != BATCH_SIZE):
        raise ValueError('full 256-row chunk and recorded resident concurrency required')
    generation = positive(metrics['generation_seconds'], 'generation seconds')
    wall = positive(metrics['wall_seconds'], 'wall seconds')
    if wall < generation:
        raise ValueError('wall time cannot omit generation')
    cells = metrics['per_cell']
    if set(cells) != set(cell_ids()):
        raise ValueError('exact measured cells required')
    means, tokens = {}, 0
    for key, item in cells.items():
        n, total = item['count'], item['output_tokens']
        if (type(n) is not int or n != count // 8 or type(total) is not int
                or not 0 <= total <= 4096*n):
            raise ValueError('balanced positive counts and capped token lengths required')
        means[key] = total/n
        tokens += total
    if type(metrics['output_tokens']) is not int or metrics['output_tokens'] != tokens or tokens <= 0:
        raise ValueError('inconsistent measured token total')
    return dict(tokens_per_second=tokens/wall, mean_output_tokens_by_cell=means,
                measured_requests=count, measurement_wall_seconds=wall)


def stage_gate(rows, metrics, *, credit_usd, ledger_remaining_usd, hourly_usd,
               seconds_left, transfer_remaining_usd=0):
    summary = measurement_summary(metrics)
    means = summary['mean_output_tokens_by_cell']
    # Prior completed stages may only make projections more conservative.
    for k, value in metrics.get('conservative_means', {}).items():
        if k not in means or not 0 <= positive(value, 'observed mean', zero=True) <= 4096:
            raise ValueError('invalid conservative cell mean')
        means[k] = max(means[k], value)
    speed = min(summary['tokens_per_second'],
                positive(metrics.get('conservative_speed', summary['tokens_per_second']), 'speed'))
    rate = positive(hourly_usd, 'all-in compute/storage rate')
    seconds = 1.2*sum(means[cell_id(r)] for r in rows)/speed
    network = positive(transfer_remaining_usd, 'network exposure', zero=True)
    cost = seconds*rate/3600+network
    available = min(positive(credit_usd, 'credit', zero=True),
                    positive(ledger_remaining_usd, 'ledger remaining', zero=True))-RESERVE_USD
    fits = cost <= available and seconds <= positive(seconds_left, 'seconds left', zero=True)
    return dict(decision='run' if fits else 'stop_whole_stage', count=len(rows),
                projected_usd=cost, projected_seconds=seconds, available_after_reserve_usd=available,
                reserve_usd=RESERVE_USD, runtime_margin=1.2, outcomes_consulted=False,
                tokens_per_second=speed, mean_output_tokens_by_cell=means)


def run_stages(rows, gate, generate, record):
    completed = []
    for index, (name, selected) in enumerate(stage_rows(rows).items()):
        decision = gate(name, selected)
        record(dict(stage=name, decision=decision, decided_unix=time.time(), outcomes_consulted=False))
        if decision['decision'] != 'run':
            return dict(completed_stages=completed, dropped_stages=list(STAGES[index:]),
                        inference_eligible=False, complete_selected_stages=True,
                        audit_status='not_assessed_by_generation_controller')
        generate(name, selected)
        completed.append(name)
    return dict(completed_stages=completed, dropped_stages=[], inference_eligible=False,
                complete_selected_stages=True, audit_status='not_assessed_by_generation_controller')


class RuntimeSmokeFailure(RuntimeError):
    """Only a bounded, unscored load/generate failure can select the fallback."""
    def __init__(self, evidence):
        super().__init__('unscored runtime smoke failed; see saved evidence')
        self.evidence = evidence


def run_families(candidate, persist_proof, clock=time.time):
    attempts = []
    for stage in ('startup', 'fix'):
        try:
            return candidate(NEMOTRON, stage, None)
        except RuntimeSmokeFailure as error:
            attempt = error.evidence
            if (attempt.get('stage') != stage or attempt.get('status') != 'failed'
                    or attempt.get('purpose') != 'runtime_smoke' or attempt.get('violation_scored') is not False
                    or not attempt.get('evidence') or attempt['ended_unix'] < attempt['started_unix']
                    or (attempts and attempt['started_unix'] < attempts[-1]['ended_unix'])):
                raise ValueError('invalid chronological runtime failure evidence')
            attempts.append(attempt)
    proof = dict(kind='runtime_startup_failure', model=NEMOTRON,
                 model_revision=model_spec(NEMOTRON)['revision'], registration_commit=REGISTRATION_COMMIT,
                 gpu='RTX 5090', attempts=attempts, outcomes_consulted=False,
                 selected_before_generation=True, decision_unix=clock())
    if proof['decision_unix'] < attempts[-1]['ended_unix']:
        raise ValueError('fallback decision precedes failure')
    # Persist before even creating/downloading the GLM rental. first_generation_unix
    # is appended from the worker's actual pre-generation timestamp, never invented.
    persist_proof(proof)
    return candidate(GLM, 'fallback', proof)


def public_account(value):
    result = {k: value[k] for k in ('credit', 'total_spend') if k in value}
    positive(result['credit'], 'provider credit', zero=True)
    # Vast total_spend is a signed accounting field, not spendable credit.
    if 'total_spend' in result:
        val = result['total_spend']
        if isinstance(val, bool) or not math.isfinite(float(val)):
            raise ValueError('provider total_spend must be signed finite')
    return result


def offer_costs(offer):
    verified = (offer.get('verification') == 'verified' if 'verification' in offer
                else offer.get('verified') is True)
    if (offer.get('gpu_name') != 'RTX 5090' or offer.get('num_gpus') != 1
            or positive(offer['reliability'], 'reliability') < .99
            or positive(offer['gpu_ram'], 'GPU RAM') < 32000
            or offer.get('rentable') is not True or not verified
            or ('verified' in offer and offer['verified'] is not True)):
        raise BudgetError('fresh verified, reliable single RTX 5090 offer required')
    # Vast prices storage monthly; search's dph_total may use only default 5GB.
    hourly = positive(offer['dph_base'], 'base hourly') + DISK_GB*positive(
        offer['storage_cost'], 'monthly disk cost', zero=True)/720
    down = positive(offer['inet_down_cost'], 'ingress $/GB', zero=True)
    up = positive(offer['inet_up_cost'], 'egress $/GB', zero=True)
    return dict(hourly_usd=hourly, network_usd=DOWNLOAD_GB*down+UPLOAD_GB*up,
                download_gb=DOWNLOAD_GB, upload_gb=UPLOAD_GB, disk_gb=DISK_GB,
                ingress_usd_per_gb=down, egress_usd_per_gb=up)


def deadline_allocation(credit, ledger_available, costs):
    exposure = min(positive(credit, 'credit'), positive(ledger_available, 'ledger budget'))-RESERVE_USD
    hourly, network = costs['hourly_usd'], costs['network_usd']
    # Bound total lifetime, including install, failures, storage and full transfer
    # envelope. Leave rounding slack; reserve() rounds UP to microdollars.
    seconds = math.floor((exposure-network)*3600/hourly)-2
    startup, teardown = 900, 120
    runtime = seconds-startup-teardown
    if runtime <= 0:
        raise BudgetError('remaining balance cannot fund bounded startup, network and reserve')
    all_in = hourly+network*3600/seconds
    return dict(hourly_usd=all_in, runtime_seconds=runtime, startup_seconds=startup,
                teardown_seconds=teardown, reserved_usd=all_in*seconds/3600)


def reserve_create(ledger, provider, attempt_id, offer, allocation, watchdog):
    ticket = ledger.reserve(attempt_id, provider='vast', **{k: allocation[k] for k in (
        'hourly_usd', 'runtime_seconds', 'startup_seconds', 'teardown_seconds')})
    watchdog(ledger, attempt_id)
    try:
        instance = provider.create(offer, ticket['resource_label'])
    except BaseException:
        ledger.note(attempt_id,create_uncertain=True)
        raise
    ledger.bind_resource(attempt_id, instance)
    return instance


def cleanup(provider):
    pulled = False
    errors = []
    try:
        provider.pull()
        pulled = True
    except Exception as error:
        errors.append(type(error).__name__ + ': artifact pull failed')
    finally:
        zero = False
        for _ in range(3):
            try:
                provider.destroy_owned()
            except Exception as error:
                errors.append(type(error).__name__ + ': owned destroy failed')
            try:
                after = provider.instances()
                zero = not after
                if zero:
                    break
            except Exception as error:
                errors.append(type(error).__name__ + ': zero-instance query failed')
    return dict(pulled=pulled, zero_instances=zero, errors=errors, unix=time.time())


class CachedEngine:
    """Compatibility adapter, lets unmodified generate() reuse exactly one LLM."""
    def __init__(self, factory):
        self.factory, self.options, self.engine = factory, None, None

    def __call__(self, **options):
        if self.options is None:
            self.engine = self.factory(**options)
            self.options = options.copy()
        elif options != self.options:
            raise ValueError('cached engine options drifted between stages')
        return self.engine


def timing_measurement(stats, *, wall_seconds, name, resident_sequences=BATCH_SIZE):
    setup = positive(stats['setup_seconds'], 'setup seconds', zero=True)
    wall = positive(wall_seconds, 'call wall seconds')
    # Smoke loads weights once; those paid seconds are already in the ledger,
    # not a measure of steady-state generation. Later calls retain all overhead.
    if name == 'smoke':
        wall -= setup
    generation = positive(stats['generation_seconds'], 'generation seconds')
    return dict(requests=stats['requests'], output_tokens=stats['output_tokens'],
                generation_seconds=generation, wall_seconds=max(generation, wall),
                setup_seconds=setup, batch_size=resident_sequences, chunk_size=BATCH_SIZE,
                per_cell=stats['per_cell'])


def throughput_allowance(*, credit_usd, ledger_remaining_usd, hourly_usd, seconds_left, now):
    """Prospective measurement exposure, independent of low-batch smoke speed.

    A timeout excludes the pilot and stops execution, it never shortens a main
    cell or selects a family. Network exposure is already debited by the caller.
    """
    seconds = MEASUREMENT_SECONDS
    available = min(positive(credit_usd, 'credit', zero=True),
                    positive(ledger_remaining_usd, 'ledger remaining', zero=True))-RESERVE_USD
    maximum_cost = seconds*positive(hourly_usd, 'hourly rate')/3600
    fits = maximum_cost <= available and seconds <= positive(seconds_left, 'seconds left', zero=True)
    return dict(decision='run' if fits else 'stop_measurement', allowance_seconds=seconds,
                maximum_usd=maximum_cost, available_after_reserve_usd=available,
                deadline_unix=now+seconds, smoke_speed_used=False, outcomes_consulted=False)


def runtime_fix_policy(error_text):
    """One predeclared, error-specific runtime repair; unknown failures stop.

    Keep the 256-row generation chunk, seeds, context and sampling fixed. A
    resident limit of 128 lets vLLM schedule that same chunk in waves. No second
    tuning pass is permitted. Eager mode targets graph/capture failures only.
    """
    text = error_text.lower()
    if any(s in text for s in ('out of memory', 'outofmemoryerror', 'no available memory for the cache',
                               'not enough memory', 'cache blocks are insufficient')):
        return dict(kind='memory_concurrency', resident_sequences=128, enforce_eager=False)
    if any(s in text for s in ('cuda graph', 'cudagraph', 'graph capture', 'stream is capturing')):
        return dict(kind='graph_capture', resident_sequences=BATCH_SIZE, enforce_eager=True)
    return None


def validate_runtime_smoke(directory, selected):
    """Replay saved protocol boundaries only, never score tutor adherence."""
    from src.phase5_split import split_response
    path = Path(directory)/(selected[0]['model'].split('/')[-1]+'.jsonl')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    expected = {r['id']: r for r in selected}
    if len(rows) != 8 or len(expected) != 8 or {r['id'] for r in rows} != set(expected):
        raise RuntimeError('smoke requires exactly the eight selected prompt identities')
    for row in rows:
        if any(row.get(k) != v for k,v in expected[row['id']].items()):
            raise RuntimeError('smoke request identity changed')
        ids = row['output_token_ids']
        if (not ids or len(ids) > 4096 or row.get('output_tokens') != len(ids)
                or row.get('stage') != 'pilot' or row.get('pilot') is not True
                or (row['finish_reason'] == 'length' and len(ids) != 4096)):
            raise RuntimeError('malformed smoke generation provenance/count/cap')
        parsed = split_response(row)
        fields = ('reasoning', 'final_reply', 'parse_status', 'no_final',
                  'protocol_failure', 'unexpected_off_thought')
        if any(k not in row or row[k] != parsed[k] for k in fields):
            raise RuntimeError('saved smoke split disagrees with token-ID replay')
        if parsed['protocol_failure'] or (parsed['no_final'] and row['finish_reason'] != 'length'):
            raise RuntimeError('smoke failed native generation protocol')
    return dict(requests=8, protocol_checked=True, violation_scored=False)


def network_counters():
    """Count all non-loopback interfaces conservatively, no addresses/content."""
    interfaces = [p for p in Path('/sys/class/net').iterdir() if p.name != 'lo']
    if not interfaces:
        raise BudgetError('network counters unavailable, cannot enforce paid transfer envelope')
    return tuple(sum(int((p/'statistics'/name).read_text()) for p in interfaces)
                 for name in ('rx_bytes', 'tx_bytes'))


def remote_stop_reason(baseline, current, *, deadline, now):
    if now >= deadline:
        return 'absolute_deadline'
    # A full GB headroom covers setup transfer and bytes in flight at shutdown.
    if current[0]-baseline[0] >= (DOWNLOAD_GB-1)*10**9:
        return 'download_budget'
    if current[1]-baseline[1] >= (UPLOAD_GB-1)*10**9:
        return 'upload_budget'
    if any(c < b for b,c in zip(baseline,current)):
        return 'network_counters_reset'
    return None


def guarded_launch(config_path):
    """Independent remote wall/transfer guard, before pip or model downloads.

    Kill only the exact process group created here. The local watchdog still
    destroys the owned rental if this guard or its SSH connection disappears.
    """
    config = json.loads(Path(config_path).read_text())
    directory = Path(REMOTE_ROOT+'/output')
    baseline = network_counters()
    with (directory/'bootstrap.log').open('ab') as log:
        child = subprocess.Popen(['bash', REMOTE_ROOT+'/bootstrap.sh'], stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log, start_new_session=True, close_fds=True)
        try:
            while child.poll() is None:
                try:
                    reason = remote_stop_reason(baseline,network_counters(),
                        deadline=config['deadline_unix'],now=time.time())
                    permit = directory/'throughput.permit.json'
                    if permit.exists() and not (directory/'throughput.done.json').exists():
                        if time.time() >= json.loads(permit.read_text())['measurement_deadline_unix']:
                            reason = 'throughput_measurement_deadline'
                except Exception:
                    reason = 'network_counter_failure'
                if reason:
                    atomic_json(directory/'error.json',dict(phase='resource_guard',error_type=reason,
                                                           ended_unix=time.time()))
                    break
                time.sleep(.25)
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait(timeout=5)
        if child.returncode and not (directory/'error.json').exists():
            atomic_json(directory/'error.json',dict(phase='process_exit',error_type='NonzeroExit',
                                                   ended_unix=time.time()))


PUBLIC_INSTANCE_KEYS = ('id', 'label', 'num_gpus', 'gpu_name', 'actual_status', 'cur_state',
                        'dph_total', 'disk_space', 'ssh_host', 'ssh_port')
PUBLIC_OFFER_KEYS = ('id', 'gpu_name', 'num_gpus', 'gpu_ram', 'reliability', 'verified', 'verification', 'rentable',
                     'dph_base', 'dph_total', 'storage_cost', 'inet_down_cost', 'inet_up_cost',
                     'inet_down', 'inet_up', 'cuda_max_good', 'disk_space')


class Provider:
    """CLI owns credentials; never expose account objects, env, or stderr."""
    def __init__(self, local_dir, *, label=None, instance=None, host=None, port=None):
        self.local = Path(local_dir)
        self.label, self.instance, self.host, self.port = label, instance, host, port
        self.remote = REMOTE_ROOT + '/output'
        self.operation_deadline = None

    def run(self, argv, *, timeout=30, input=None):
        if self.operation_deadline is not None:
            timeout = max(.1, min(timeout, self.operation_deadline-time.time()))
        try:
            result = subprocess.run(argv, input=input, capture_output=True, text=True,
                                    check=True, timeout=timeout)
            return result.stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            # CalledProcessError/TimeoutExpired text can include provider payloads.
            raise RuntimeError(f'{Path(argv[0]).name} operation failed ({type(exc).__name__}); no raw stderr exposed') from None

    def cli(self, *args):
        # The installed CLI can emit an API error only on stderr and exit 0
        # with empty stdout. Retry bounded READS, never uncertain creates.
        reads = args[:2] in (('show', 'instances'), ('show', 'user'), ('search', 'offers'))
        for attempt in range(3 if reads else 1):
            raw = self.run(['vastai', *map(str, args), '--raw'])
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                if not reads or attempt == 2:
                    raise RuntimeError('Vast CLI returned non-JSON; no provider payload exposed') from None
                time.sleep(2)

    def account(self):
        return public_account(self.cli('show', 'user'))

    def instances(self):
        return [{k:r[k] for k in PUBLIC_INSTANCE_KEYS if k in r}
                for r in self.cli('show', 'instances')]

    def scan(self, offer_id):
        # This is always a fresh API search, with requested disk size, never /tmp.
        rows = self.cli('search', 'offers', 'gpu_name=RTX_5090 num_gpus=1 reliability>=0.99 '
                        'verified=true rentable=true disk_space>=80', '--storage', DISK_GB,
                        '--limit', 200, '--order', 'dph_total')
        offered = next((r for r in rows if r['id'] == offer_id), None)
        if offered is None:
            raise BudgetError('requested offer absent from fresh scan; choose a fresh offer, never auto-rent an alternative')
        return {k:offered[k] for k in PUBLIC_OFFER_KEYS if k in offered}

    def create(self, offer, label):
        self.label = label
        reply = self.cli('create', 'instance', offer['id'], '--image', IMAGE, '--disk', DISK_GB,
                         '--label', label, '--ssh', '--direct', '--cancel-unavail', '--env',
                         '-e HF_HOME=/hf -e VLLM_USE_FLASHINFER_SAMPLER=0', '--retry', 1)
        if reply.get('success') is not True or type(reply.get('new_contract')) is not int:
            raise RuntimeError('Vast create response uncertain; label watchdog retains ownership')
        self.instance = reply['new_contract']
        return self.instance

    def owned(self, rows):
        return [r for r in rows if r.get('label') == self.label or
                (self.instance is not None and str(r['id']) == str(self.instance))]

    def destroy_owned(self):
        for row in self.owned(self.instances()):
            self.instance = row['id']
            # destroy prompts even with --raw, and successful --yes emits no
            # JSON acknowledgement. The following instances query is proof.
            self.run(['vastai', 'destroy', 'instance', str(row['id']), '--yes', '--raw'])

    def ssh(self, code, *, timeout=30):
        if not self.host or not self.port:
            raise RuntimeError('SSH not ready')
        return self.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                         '-o', 'StrictHostKeyChecking=accept-new', '-p', str(self.port),
                         'root@'+self.host, code], timeout=timeout)

    def put(self, source, destination):
        self.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                  '-o', 'StrictHostKeyChecking=accept-new', '-P', str(self.port),
                  str(source), 'root@'+self.host+':'+shlex.quote(destination)], timeout=40)

    def read(self, name):
        code = ('import pathlib; p=pathlib.Path(' + repr(self.remote+'/'+name) + '); '
                'print(p.read_text() if p.exists() else "null")')
        return json.loads(self.ssh('python3 -c '+shlex.quote(code)))

    def send(self, name, value):
        path = self.local / name
        atomic_json(path, value)
        # Atomic rename means the worker never observes a partially copied permit.
        self.put(path, self.remote+'/'+name+'.incoming')
        self.ssh('mv '+shlex.quote(self.remote+'/'+name+'.incoming')+' '+shlex.quote(self.remote+'/'+name))

    def pull(self):
        if not self.host:
            # There are no generation artifacts if SSH/bootstrap never started.
            return
        self.local.mkdir(parents=True, exist_ok=True)
        code = ('import pathlib; d=pathlib.Path('+repr(self.remote)+'); '
                'print(sum(p.stat().st_size for p in d.rglob("*") if p.is_file()))')
        size = int(self.ssh('python3 -c '+shlex.quote(code)))
        transfer_path = self.local/'pull-transfer.json'
        with transfer_path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            transferred = json.loads(transfer_path.read_text())['reserved_bytes'] if transfer_path.exists() else 0
            # Full SCP copies, including retries, are charged in advance. Keep 1GB
            # for SSH/log/control overhead and initial bytes sent before counters.
            if size < 0 or transferred+size > (UPLOAD_GB-1)*10**9:
                raise BudgetError('artifact transfer envelope exhausted before SCP; destroying rental')
            atomic_json(transfer_path,dict(reserved_bytes=transferred+size, unix=time.time()))
        self.run(['scp', '-r', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                  '-o', 'StrictHostKeyChecking=accept-new', '-P', str(self.port),
                  'root@'+self.host+':'+self.remote+'/.', str(self.local)], timeout=35)

    def wait_ssh(self, deadline):
        while time.time() < deadline:
            rows = self.instances()
            owned = self.owned(rows)
            if len(rows) != 1 or len(owned) != 1:
                raise BudgetError('exactly one owned provider instance required')
            row = owned[0]
            if row.get('ssh_host') and row.get('ssh_port'):
                self.host, self.port = row['ssh_host'], int(row['ssh_port'])
                try:
                    if self.ssh('python3 -c "print(12345)"', timeout=10).strip() == '12345':
                        return
                except RuntimeError:
                    pass
            time.sleep(2)
        raise TimeoutError('provider SSH startup deadline')


def arm_watchdog(ledger, attempt_id):
    log = ledger.path.parent / (attempt_id+'-vast-main-watchdog.log')
    with log.open('ab') as out:
        child = subprocess.Popen([sys.executable, '-m', 'src.phase5_vast_main', '--watchdog',
                                  '--attempt-id', attempt_id, '--ledger', str(ledger.path)],
                                 cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=out,
                                 start_new_session=True, close_fds=True)
    ledger.note(attempt_id, watchdog_pid=child.pid, watchdog_log=str(log))


def watchdog(ledger, attempt_id):
    """Detached label recovery also covers lost create responses; no broad kill."""
    while True:
        ticket = ledger.attempt(attempt_id)
        if ticket['teardown_verified']:
            return
        started = any(e.get('startup_complete') is True for e in ticket['events'])
        deadline = ticket['work_deadline_unix'] if started else ticket['startup_deadline_unix']
        if time.time() < deadline:
            time.sleep(min(1, deadline-time.time()))
            continue
        recovery = next((e['recovery'] for e in reversed(ticket['events']) if 'recovery' in e), {})
        provider = Provider(recovery.get('local_dir', ROOT/'results/raw'/attempt_id),
                            label=ticket['resource_label'], instance=ticket['resource_id'],
                            host=recovery.get('host'), port=recovery.get('port'))
        provider.operation_deadline = time.time()+60
        try:
            result = cleanup(provider)
            ledger.note(attempt_id, detached_watchdog=result)
            ledger.settle(attempt_id, actual_usd=ticket['reserved_usd'],
                          teardown_verified=result['zero_instances'], success=False)
            if result['zero_instances']:
                return
        except Exception as error:
            # Provider/ledger trouble must not silently end the final kill switch.
            print('WATCHDOG_RETRY '+type(error).__name__,flush=True)
        time.sleep(2)


def make_bundle(destination, grids):
    """Explicit allowlist only; never include .env, credentials, or raw outcomes."""
    paths = ('src/__init__.py', 'src/phase5_vast_main.py', 'src/phase5_control.py',
             'src/models_5.py', 'src/vast_generate_5.py', 'src/phase5_split.py',
             'src/build_prompts.py', 'DESIGN.md', 'prompts/templates.json',
             'prompts/templates_4.json', 'prompts/system_tutor.txt')
    payload = Path(destination).with_suffix('.requests.json')
    with payload.open('x') as out:
        json.dump(grids, out)
    with tarfile.open(destination, 'w:gz') as archive:
        for name in paths:
            archive.add(ROOT/name, arcname=name, recursive=False)
        archive.add(payload, arcname='requests.json', recursive=False)
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}


def worker(config_path):
    """Runs only on the rental, after the local ticket/authorization gates."""
    from src import vast_generate_5 as generator
    config = json.loads(Path(config_path).read_text())
    directory = Path(REMOTE_ROOT+'/output')
    directory.mkdir(parents=True, exist_ok=True)
    started, phase = time.time(), 'initialization'
    try:
        if (config.get('registration_commit') != REGISTRATION_COMMIT
                or not config.get('authorization_id') or config['model'] not in (NEMOTRON, GLM)):
            raise BudgetError('invalid remote authorization/registration')
        os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
        os.environ['HF_HOME'] = '/hf'
        # Check exact resolver results before any model load.
        versions = generator.runtime_versions()
        if any(versions[k] != v for k,v in PACKAGE_VERSIONS.items()):
            raise ValueError('pinned package version mismatch')
        import vllm
        vllm.LLM = CachedEngine(vllm.LLM)
        model = config['model']
        spec = json.loads(Path(REMOTE_ROOT+'/requests.json').read_text())[model]
        rows, manifest = spec['requests'], spec['manifest']
        stage_rows(rows)
        first_generation = None
        resident_sequences = config.get('resident_sequences', BATCH_SIZE)
        if resident_sequences not in (64, 128, BATCH_SIZE):
            raise ValueError('unregistered resident concurrency')

        def expired():
            if time.time() >= config['deadline_unix']:
                raise TimeoutError('remote absolute work deadline')

        def generate(name, selected, stage):
            nonlocal first_generation
            expired()
            tick = time.monotonic()
            if first_generation is None:
                first_generation = time.time()
                atomic_json(directory/'generation-start.json', dict(first_generation_unix=first_generation))
            # Stage directories retain raw bytes; all main rows share one run.
            run_id = config['attempt_id']+('-main' if stage == 'main' else '-'+name)
            stats = generator.generate(model, selected, run_id,
                {**manifest, 'stage':stage, 'controller_stage':name,
                 'execution_authorization_id':config['authorization_id'],
                 'fallback_provenance':config.get('fallback_provenance'),
                 'pilot_rows_excluded_from_main':True},
                dict(output_dir=str(directory/name), batch_size=resident_sequences, chunk_size=BATCH_SIZE,
                     enforce_eager=config['enforce_eager'], before_chunk=expired,
                     host='vast', gpu='RTX 5090'))
            if stats['requests'] != len(selected):
                raise RuntimeError('incomplete stage, never inference eligible')
            # Only timing/counts leave the worker control channel.
            cells = {}
            for key, item in stats['per_cell'].items():
                suffix = next((k for k in cell_ids() if key.endswith('/'+k)), None)
                if suffix is None:
                    raise ValueError('unknown generator cell metric')
                cells[suffix] = item
            result = timing_measurement({**stats,'per_cell':cells},
                                        wall_seconds=time.monotonic()-tick,name=name,
                                        resident_sequences=resident_sequences)
            result.update(first_generation_unix=first_generation,completed_unix=time.time())
            if name == 'smoke':
                result['protocol'] = validate_runtime_smoke(directory/name, selected)
            atomic_json(directory/(name+'.done.json'), result)
            return result

        phase = 'runtime_smoke'
        atomic_json(directory/'runtime-start.json',dict(started_unix=time.time(), purpose=phase))
        smoke = generate('smoke', balanced_pilot(rows, 8), 'pilot')
        if smoke['requests'] != 8 or any(item['output_tokens'] <= 0 for item in smoke['per_cell'].values()):
            raise RuntimeError('smoke produced an empty completion')
        phase = 'await_throughput'
        names = ['throughput', *STAGES]
        for name in names:
            permit = directory/(name+'.permit.json')
            while not permit.exists():
                expired()
                if (directory/'stop.json').exists():
                    return
                time.sleep(.2)
            gate = json.loads(permit.read_text())
            if (gate.get('action') != 'run' or gate.get('attempt_id') != config['attempt_id']
                    or gate.get('deadline_unix') != config['deadline_unix']):
                raise BudgetError('invalid stage permit')
            selected = balanced_pilot(rows, BATCH_SIZE) if name == 'throughput' else stage_rows(rows)[name]
            if gate['count'] != len(selected):
                raise BudgetError('permit cannot shorten a registered stage')
            if name == 'throughput' and not time.time() < gate['measurement_deadline_unix'] <= config['deadline_unix']:
                raise BudgetError('bounded throughput deadline required')
            phase = name
            generate(name, selected, 'pilot' if name == 'throughput' else 'main')
        atomic_json(directory/'finished.json',dict(completed_unix=time.time()))
    except BaseException as error:
        # Runtime trace is local to ignored raw artifacts; no raw provider/account
        # exception or environment is written to the public accounting channel.
        import traceback
        (directory/'runtime-trace.txt').write_text(traceback.format_exc())
        atomic_json(directory/'error.json', dict(phase=phase, error_type=type(error).__name__,
                    runtime_fix=runtime_fix_policy(traceback.format_exc()) if phase == 'runtime_smoke' else None,
                    started_unix=started, ended_unix=time.time(), violation_scored=False))
        raise SystemExit(1)


def require_gemma_analysis(path):
    """Validate completion metadata only; no contrast/violation value is consulted."""
    from src.models_5 import GEMMA
    raw = Path(path).read_bytes()
    report = json.loads(raw)
    grid = report.get('grid', {})
    audit = report.get('parser_agreement', {})
    if (report.get('phase') != 'phase5' or grid.get('model_scope') != GEMMA
            or grid.get('n_problems') != 200 or grid.get('n_responses', 0) < 1600
            or grid.get('mode') == 'pilot' or grid.get('problem_manifest_checked') is not True
            or grid.get('original_sample_checked') is not True):
        raise BudgetError('completed 200-problem Gemma analysis required before second-family execution')
    if (audit.get('status') != 'complete' or audit.get('n_required') != 40
            or audit.get('n_labeled') != 40 or audit.get('committed_number', {}).get('status') != 'complete'
            or audit.get('splitter', {}).get('status') != 'complete'
            or audit.get('splitter', {}).get('n_labeled') != 40):
        raise BudgetError('completed 40-label Gemma blind parser and final-boundary audit required before Vast')
    return dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                scope=GEMMA, analysis_present=True, blind_audit_complete=True, outcomes_consulted=False)


class Supervisor:
    def __init__(self, args, ledger, directory, bundle, baseline):
        self.args, self.ledger, self.directory = args, ledger, directory
        self.bundle, self.baseline, self.charged = bundle, baseline, 0.0

    def persist_proof(self, proof):
        atomic_json(self.directory/'fallback-provenance.json', proof)

    def candidate(self, model, stage, proof):
        policy = dict(kind='initial', resident_sequences=BATCH_SIZE, enforce_eager=False)
        if stage == 'fix':
            policy = getattr(self, 'next_fix', None)
            if not policy:
                raise BudgetError('first runtime error has no supported targeted fix; no speculative rental')
        if stage == 'fallback':
            policy = dict(kind='registered_fallback', resident_sequences=BATCH_SIZE, enforce_eager=True)
        attempt_id = safe_id(self.args.run_id+'-'+stage)
        local = self.directory/stage
        local.mkdir()
        backend = Provider(local)
        before = backend.account()
        if backend.instances():
            raise BudgetError('zero Vast instances required before any reservation/create')
        offer = backend.scan(self.args.offer_id)
        costs = offer_costs(offer)
        if costs['hourly_usd'] > self.args.max_hourly_usd:
            raise BudgetError('fresh compute/storage price exceeds explicit hourly ceiling')
        credit = min(CREDIT_CEILING, self.baseline['credit'])-self.charged
        credit = min(credit, before['credit'])
        allocation = deadline_allocation(credit, self.ledger.snapshot()['available_usd'], costs)
        atomic_json(local/'launch-accounting.json', dict(account_before=before, offer=offer,
            costs=costs, allocation=allocation, authorized_credit_ceiling_usd=CREDIT_CEILING,
            prior_attempts_conservative_usd=self.charged, reserve_usd=RESERVE_USD, runtime_policy=policy))
        failure = None
        report = None
        ticket = None
        try:
            # Own label before reserve/create, including every uncertain failure.
            backend.label = 'phase5-'+attempt_id
            reserve_create(self.ledger, backend, attempt_id, offer, allocation, arm_watchdog)
            ticket = self.ledger.attempt(attempt_id)
            backend.operation_deadline = ticket['work_deadline_unix']
            backend.wait_ssh(ticket['startup_deadline_unix'])
            actual = backend.instances()
            if (len(actual) != 1 or len(backend.owned(actual)) != 1
                    or positive(actual[0].get('dph_total'), 'actual instance rate') > costs['hourly_usd']+.00001):
                raise BudgetError('rented instance rate/ownership differs from fresh reservation')
            self.ledger.note(attempt_id, recovery=dict(local_dir=str(local), host=backend.host, port=backend.port))
            config = dict(model=model, attempt_id=attempt_id, registration_commit=REGISTRATION_COMMIT,
                          authorization_id=self.args.authorization_id, deadline_unix=ticket['work_deadline_unix'],
                          enforce_eager=policy['enforce_eager'], resident_sequences=policy['resident_sequences'],
                          runtime_policy=policy, fallback_provenance=proof)
            backend.ssh('mkdir -p '+REMOTE_ROOT+'/output')
            backend.put(self.bundle, REMOTE_ROOT+'/bundle.tar.gz')
            config_path = local/'worker-config.json'
            atomic_json(config_path, config)
            backend.put(config_path, REMOTE_ROOT+'/config.json')
            install = ' '.join(shlex.quote(k+'=='+v) for k,v in PACKAGE_VERSIONS.items())
            bootstrap = ('#!/bin/bash\nset -eu\ncd '+REMOTE_ROOT+'\n'
                'export HF_HOME=/hf VLLM_USE_FLASHINFER_SAMPLER=0 PIP_NO_CACHE_DIR=1 PHASE5_RUNTIME_IMAGE='+IMAGE+'\n'
                'if ! python3 -m pip install --disable-pip-version-check '+install+
                ' > output/install.log 2>&1; then\n'
                '  echo \'{"phase":"installation","error_type":"PackageInstallError"}\' > output/error.json\n'
                '  exit 1\nfi\n'
                'exec python3 -m src.phase5_vast_main --worker '+REMOTE_ROOT+'/config.json\n')
            bootstrap_path = local/'bootstrap.sh'
            bootstrap_path.write_text(bootstrap)
            backend.put(bootstrap_path, REMOTE_ROOT+'/bootstrap.sh')
            backend.ssh('cd '+REMOTE_ROOT+' && tar xzf bundle.tar.gz')
            backend.ssh('cd '+REMOTE_ROOT+' && nohup python3 -m src.phase5_vast_main --remote-launch '+
                        REMOTE_ROOT+'/config.json > '+REMOTE_ROOT+'/output/worker.log 2>&1 < /dev/null &')

            def remaining_budget():
                snapshot = self.ledger.snapshot()
                current = costs['network_usd']+(time.time()-ticket['started_unix'])*costs['hourly_usd']/3600
                account = backend.account()
                account_credit = min(account['credit'], credit-current)
                # The active ticket is already reserved; remove only that same
                # reservation from outstanding to avoid counting it twice.
                ledger_remaining = (snapshot['cap_microusd']/1e6-snapshot['spent_usd']-
                                    max(0, snapshot['outstanding_usd']-ticket['reserved_usd'])-current)
                atomic_json(local/'account-latest.json', dict(account=account, estimated_current_usd=current,
                            effective_credit_usd=account_credit, unix=time.time()))
                return max(0, account_credit), max(0, ledger_remaining)

            def wait_result(name, deadline):
                last_check = 0
                while time.time() < min(deadline, ticket['work_deadline_unix']):
                    if time.time()-last_check >= 10:
                        available, lifetime = remaining_budget()
                        if min(available, lifetime) <= RESERVE_USD:
                            raise BudgetError('live credit/lifetime gate reached reserve, partial stage is noninferential')
                        rows = backend.instances()
                        if len(rows) != 1 or len(backend.owned(rows)) != 1:
                            raise RuntimeError('owned rental disappeared during stage')
                        last_check = time.time()
                    error = backend.read('error.json')
                    if error:
                        atomic_json(local/'worker-error.json', error)
                        if error.get('phase') == 'runtime_smoke':
                            # Re-derive from the pulled traceback after destruction;
                            # never trust a worker-suggested arbitrary configuration.
                            raise RuntimeSmokeFailure(dict(attempt_id=attempt_id, stage=stage, status='failed',
                                purpose='runtime_smoke', violation_scored=False,
                                started_unix=error['started_unix'], ended_unix=error['ended_unix'],
                                evidence=str(local/'runtime-trace.txt')))
                        raise RuntimeError('worker failed outside runtime smoke; no outcome-selected fallback')
                    result = backend.read(name+'.done.json')
                    if result:
                        atomic_json(local/(name+'.done.json'), result)
                        return result
                    time.sleep(2)
                # Timeouts count only when the worker actually entered runtime
                # load/generate, not when provider setup/package install stalled.
                runtime = backend.read('runtime-start.json') if name == 'smoke' else None
                if runtime:
                    raise RuntimeSmokeFailure(dict(attempt_id=attempt_id, stage=stage, status='failed',
                        purpose='runtime_smoke', violation_scored=False, started_unix=runtime['started_unix'],
                        ended_unix=time.time(), evidence=str(local/'runtime-timeout.json')))
                raise TimeoutError('bounded startup/work stage deadline; no fallback on infrastructure failure')

            smoke = wait_result('smoke', ticket['startup_deadline_unix'])
            self.ledger.note(attempt_id, startup_complete=True)
            if smoke.get('batch_size') != policy['resident_sequences']:
                raise RuntimeError('reported smoke resident concurrency differs from prospective policy')
            if proof is not None:
                first = positive(smoke['first_generation_unix'], 'actual GLM generation timestamp')
                if first <= proof['decision_unix']:
                    raise BudgetError('GLM generation did not follow prospective fallback decision')
                proof['first_generation_unix'] = first
                self.persist_proof(proof)
            measurement_summary(smoke, minimum=8)  # Shape only, no speed projection.
            if smoke.get('protocol') != dict(requests=8, protocol_checked=True, violation_scored=False):
                raise RuntimeError('worker did not validate smoke protocol')
            available, lifetime = remaining_budget()
            allowance = throughput_allowance(credit_usd=available, ledger_remaining_usd=lifetime,
                hourly_usd=costs['hourly_usd'], seconds_left=max(0,ticket['work_deadline_unix']-time.time()),
                now=time.time())
            atomic_json(local/'throughput-allowance.json', allowance)
            self.ledger.note(attempt_id, throughput_allowance=allowance)
            if allowance['decision'] != 'run':
                report = dict(model=model, status='throughput_budget_stop', completed_stages=[],
                              inference_eligible=False, proof=proof)
                atomic_json(local/'result.json',report)
                return report

            def permit(name, selected):
                backend.send(name+'.permit.json', dict(action='run', count=len(selected),
                    attempt_id=attempt_id, deadline_unix=ticket['work_deadline_unix'],
                    **({'measurement_deadline_unix':allowance['deadline_unix']} if name == 'throughput' else {})))

            # Read exactly the bundled construction; don't resample after rent.
            grids = json.loads(self.bundle.with_suffix('.requests.json').read_text())
            rows = grids[model]['requests']
            permit('throughput', balanced_pilot(rows, BATCH_SIZE))
            measured = wait_result('throughput', allowance['deadline_unix'])
            measurement_summary(measured)
            if measured['batch_size'] != policy['resident_sequences']:
                raise RuntimeError('reported throughput resident concurrency differs from prospective policy')
            atomic_json(local/'throughput-receipt.json', dict(allowance=allowance, measured=measured,
                        received_unix=time.time(), smoke_speed_used=False, outcomes_consulted=False))

            def gate(name, selected):
                available, lifetime = remaining_budget()
                return stage_gate(selected, measured, credit_usd=available, ledger_remaining_usd=lifetime,
                    hourly_usd=costs['hourly_usd'], seconds_left=max(0,ticket['work_deadline_unix']-time.time()))

            def generate(name, selected):
                permit(name, selected)
                observed = wait_result(name, ticket['work_deadline_unix'])
                # Fail closed on missing/extra counts. Raw completeness is also
                # verified after pulling, before marking any stage complete.
                if observed['requests'] != len(selected):
                    raise RuntimeError('stage completion marker has wrong count')
                means = measured.setdefault('conservative_means', {})
                for key,item in observed['per_cell'].items():
                    mean = item['output_tokens']/item['count']
                    means[key] = max(means.get(key,0),mean)
                speed = observed['output_tokens']/positive(observed['wall_seconds'], 'stage time')
                if speed > 0:
                    measured['conservative_speed'] = min(measured.get('conservative_speed',float('inf')),speed)
                backend.pull()
                validate_pulled_stage(local/name, selected)

            decisions = []
            def record(value):
                decisions.append(value)
                atomic_json(local/'stage-decisions.json', decisions)
                self.ledger.note(attempt_id, stage_decision=value)
            report = dict(model=model, **run_stages(rows, gate, generate, record), proof=proof)
            backend.send('stop.json', dict(reason='all accepted stages finished or whole-stage budget stop'))
        except BaseException as error:
            failure = error
            atomic_json(local/'execution-error.json',dict(error_type=type(error).__name__,
                        inference_eligible=False, partial_stages_excluded=True, unix=time.time()))
        finally:
            # Even uncertain create and failed bind are recovered by ticket label.
            if any(a['attempt_id'] == attempt_id for a in self.ledger.snapshot()['attempts']):
                backend.operation_deadline = time.time()+110
                teardown = cleanup(backend)
                ticket = self.ledger.attempt(attempt_id)
                if (backend.instance is None and any(e.get('create_uncertain') is True for e in ticket['events'])):
                    # An immediate empty snapshot cannot exclude a late create.
                    # Keep the full ticket charged and detached label watchdog
                    # alive through the original startup window; no next rental.
                    teardown.update(zero_instances=False,uncertain_create_quarantined=True)
                atomic_json(local/'teardown.json', teardown)
                try:
                    after = backend.account()
                except Exception:
                    after = None
                wall = max(0,time.time()-ticket['started_unix'])*costs['hourly_usd']/3600
                provider_delta = max(0,before['credit']-after['credit']) if after else 0
                spend_delta = max(0,after.get('total_spend',0)-before.get('total_spend',0)) if after else 0
                charge = max(wall+costs['network_usd'],provider_delta,spend_delta)
                self.charged += charge
                self.ledger.note(attempt_id, provider_account_after=after, conservative_all_costs_usd=charge,
                                 teardown=teardown, error_type=type(failure).__name__ if failure else None)
                self.ledger.settle(attempt_id, actual_usd=charge, teardown_verified=teardown['zero_instances'],
                                   success=failure is None and teardown['pulled'])
                if not teardown['zero_instances']:
                    raise BudgetError('zero Vast instances not confirmed; watchdog active, no further launches')
                if not teardown['pulled']:
                    raise RuntimeError('artifact recovery failed; rental destroyed, no new family selection')
                if isinstance(failure, RuntimeSmokeFailure):
                    trace = local/'runtime-trace.txt'
                    self.next_fix = runtime_fix_policy(trace.read_text()) if trace.exists() else None
                    failure.evidence['targeted_fix'] = self.next_fix
                    atomic_json(local/'runtime-timeout.json', failure.evidence)
            # A pre-reserve error cannot possibly own a rental.
        if failure:
            raise failure
        if report['completed_stages']:
            main_path, manifest_path = local/'main.jsonl', local/'manifest.json'
            with main_path.open('xb') as out:
                for name in report['completed_stages']:
                    source = local/name/(model.split('/')[-1]+'.jsonl')
                    with source.open('rb') as handle:
                        shutil.copyfileobj(handle,out)
                out.flush()
                os.fsync(out.fileno())
            atomic_json(manifest_path,{**grids[model]['manifest'],
                'completed_stages':report['completed_stages'], 'dropped_stages':report['dropped_stages'],
                'fallback_provenance':proof, 'pilot_rows_excluded_from_main':True})
            report.update(main_output=str(main_path),manifest=str(manifest_path))
        atomic_json(local/'result.json', report)
        return report


def validate_pulled_stage(directory, selected):
    """Validate identity/count/provenance without reading replies for decisions."""
    from src.vast_generate_5 import _validate_requests
    model = selected[0]['model']
    path = Path(directory)/(model.split('/')[-1]+'.jsonl')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    _validate_requests(rows, model)
    if ({r['id'] for r in rows} != {r['id'] for r in selected} or len(rows) != len(selected)
            or any(r.get('stage') != 'main' or r.get('pilot') is not False for r in rows)):
        raise ValueError('pulled stage is partial/mixed/non-main; never inference eligible')


def plan():
    return dict(mode='prepare_only', paid_actions_performed=False,
        execution_requires='NEW execution task after completed Gemma blind audit and analysis; --execute --authorization-id --gemma-analysis',
        credit_ceiling_usd=CREDIT_CEILING, reserve_usd=RESERVE_USD, image=IMAGE,
        package_versions=PACKAGE_VERSIONS, stages=[dict(name=n,count=c) for n,c in zip(STAGES,(800,1600,800))],
        smoke_prompts=8, throughput_prompts=BATCH_SIZE, engine_reused=True,
        fix='one error-targeted attempt: memory failure -> 128 resident sequences; graph capture -> eager; unknown -> stop',
        throughput_measurement_allowance_seconds=MEASUREMENT_SECONDS,
        initial_offer_must_be_rescanned=True, no_violation_scoring=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    mode.add_argument('--remote-launch', type=Path, help=argparse.SUPPRESS)
    mode.add_argument('--watchdog', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--authorization-id')
    parser.add_argument('--gemma-analysis', type=Path)
    parser.add_argument('--offer-id', type=int, default=51405136)
    parser.add_argument('--max-hourly-usd', type=float, default=.75)
    parser.add_argument('--run-id')
    parser.add_argument('--attempt-id', help=argparse.SUPPRESS)
    parser.add_argument('--ledger', type=Path, default=LEDGER_PATH)
    args = parser.parse_args(argv)
    if args.remote_launch:
        guarded_launch(args.remote_launch)
        return 0
    if args.worker:
        worker(args.worker)
        return 0
    if args.watchdog:
        watchdog(BudgetLedger(args.ledger), safe_id(args.attempt_id))
        return 0
    if not args.execute:
        print(json.dumps(plan(), indent=2))
        return 0
    if not args.authorization_id or not args.gemma_analysis or not args.run_id:
        parser.error('NEW authorization-id, completed gemma-analysis, and fresh run-id required')
    safe_id(args.authorization_id)
    safe_id(args.run_id+'-fallback')
    gemma = require_gemma_analysis(args.gemma_analysis)
    ledger = BudgetLedger(args.ledger)  # Never initialize/upgrade/reset implicitly.
    snapshot = ledger.snapshot()
    if snapshot.get('registration_commit') != REGISTRATION_COMMIT or any(
            a['status'] in UNRESOLVED for a in snapshot['attempts']):
        raise BudgetError('registered amended ledger with zero unresolved attempts required')
    directory = ROOT/'results/raw'/args.run_id
    directory.mkdir(parents=True, exist_ok=False)
    read_only_provider = Provider(directory)
    baseline = read_only_provider.account()
    if read_only_provider.instances():
        raise BudgetError('zero Vast resources required before execution')
    from src.vast_generate_5 import build
    grids = {m: dict(zip(('requests','manifest'),build(m))) for m in (NEMOTRON,GLM)}
    bundle = directory/'bundle.tar.gz'
    source_hashes = make_bundle(bundle,grids)
    atomic_json(directory/'execution-authorization.json',dict(
        authorization_id=args.authorization_id, gemma_analysis=gemma, started_unix=time.time(),
        account_before=baseline, source_sha256=source_hashes, registration_commit=REGISTRATION_COMMIT,
        credit_ceiling_usd=CREDIT_CEILING, reserve_usd=RESERVE_USD))
    supervisor = Supervisor(args, ledger, directory, bundle, baseline)
    # SIGTERM must unwind into teardown just like keyboard interruption. SIGKILL
    # is handled by the independent label/deadline watchdog.
    def terminate(signum, frame):
        raise KeyboardInterrupt('controller terminated; recovering owned rental')
    signal.signal(signal.SIGTERM,terminate)
    result = run_families(supervisor.candidate,supervisor.persist_proof)
    atomic_json(directory/'result.json',result)
    print(json.dumps(dict(model=result['model'], completed_stages=result.get('completed_stages',[]),
                          artifacts=str(directory), zero_vast_instances=True)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
