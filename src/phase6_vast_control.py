"""Phase 6 only: bounded Vast rental, outcome-blind gates, incremental retrieval.

The registered worker owns prompts and generation. This controller reads only
timing, token counts and file names until all paid resources have been destroyed.
Every CLI call is serialized and spaced, including the independent watchdog.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'results/phase6'
LEDGER = EVIDENCE / 'vast-execution.json'
CAP = 5.0
RESERVE = .30
DISK = 100
DOWNLOAD_GB = 120
UPLOAD_GB = 8
IMAGE = 'vllm/vllm-openai@sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9'
REMOTE = '/phase6'
CLI_LOCK = Path('/tmp/phase6-vast-cli.lock')
STAGES = ('3.8-main', '3.8-prefill', '3.5-main')


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def run(argv, *, timeout=60, **kwargs):
    return subprocess.run(argv, text=True, capture_output=True, timeout=timeout,
                          check=True, **kwargs).stdout


def cli(*args, mutation=False):
    """Never retry an uncertain create; bounded backoff for reads/destruction."""
    attempts = 1 if args[:2] == ('create', 'instance') else 4
    for attempt in range(attempts):
        with CLI_LOCK.open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            lock.seek(0)
            prior = lock.read().strip()
            time.sleep(max(0, 4 - (time.time() - float(prior or 0))))
            try:
                result = subprocess.run(['vastai', *map(str, args), '--raw', '--retry', '1'],
                                        capture_output=True, text=True, timeout=50)
            except subprocess.TimeoutExpired:
                result = subprocess.CompletedProcess(args, 124, '', '')
            lock.seek(0)
            lock.truncate()
            lock.write(str(time.time()))
            lock.flush()
        if args[:2] == ('destroy', 'instance') and result.returncode == 0 and not result.stdout.strip():
            return {'destroy_sent': True}
        try:
            obj = json.loads(result.stdout)
            if result.returncode == 0 and not (isinstance(obj, dict) and obj.get('error')):
                return obj
        except json.JSONDecodeError:
            pass
        if attempt + 1 == attempts:
            # Provider stderr may contain credential-bearing request details.
            raise RuntimeError(f'Vast {args[0]} failed or returned malformed JSON; create was not retried')
        time.sleep(min(30, 5 * 2**attempt))


def account():
    obj = cli('show', 'user')
    return {k: obj[k] for k in ('credit', 'total_spend')}


def instances():
    keys = ('id', 'label', 'actual_status', 'cur_state', 'gpu_name', 'ssh_host', 'ssh_port',
            'dph_total', 'disk_space', 'start_date')
    return [{k: r[k] for k in keys if k in r} for r in cli('show', 'instances')]


def offer_costs(row):
    gpu = row.get('gpu_name', '')
    allowed = (gpu in ('L40S', 'RTX A6000') or gpu.startswith(('RTX PRO 6000', 'A100', 'H100')))
    ram = float(row.get('gpu_ram', 0))
    hourly = float(row['dph_base']) + DISK * float(row['storage_cost']) / 720
    network = DOWNLOAD_GB * float(row['inet_down_cost']) + UPLOAD_GB * float(row['inet_up_cost'])
    if (not allowed or ram < 48000 or row.get('num_gpus') != 1
            or float(row['reliability']) < .98 or row.get('rentable') is not True
            or row.get('verification', 'verified' if row.get('verified') else '') != 'verified'
            or float(row.get('cuda_max_good', 0)) < 13.0
            or not 0 < hourly <= 1.30 or not 0 <= network < CAP - RESERVE):
        raise ValueError('offer fails authorized GPU/runtime/cost criteria')
    return {'hourly_usd': hourly, 'network_envelope_usd': network,
            'download_envelope_gb': DOWNLOAD_GB, 'upload_envelope_gb': UPLOAD_GB,
            'disk_gb': DISK}


def choose_offer(rows, excluded):
    candidates = []
    for row in rows:
        try:
            cost = offer_costs(row)
        except (ValueError, KeyError, TypeError):
            continue
        if row['machine_id'] in excluded:
            continue
        # Favor 80/96GB cards for graphs and batching, then FP8-native Ada over
        # Ampere. Selection is performance/cost-only, before any outcomes.
        tier = (0 if gpu_native(row) and row['gpu_ram'] >= 80000 else
                1 if row['gpu_name'] == 'L40S' else 2 if row['gpu_ram'] >= 80000 else 3)
        candidates.append((tier, cost['hourly_usd'] + cost['network_envelope_usd'], row, cost))
    if not candidates:
        raise RuntimeError('no fresh qualifying offer within authorized criteria')
    _, _, row, cost = min(candidates, key=lambda x: x[:2])
    return row, cost


def gpu_native(row):
    return row['gpu_name'].startswith(('RTX PRO 6000', 'H100'))


def cell(row):
    return ('prefill/' + row['prefill'] if row['arm'] == 'prefill' else
            f"main/{'on' if row['thinking'] else 'off'}/{row['request_type']}")


def measurement(metrics):
    """Validate a full-batch measurement using lengths and time only."""
    n = metrics['requests']
    wall = float(metrics.get('wall_seconds', metrics['generation_seconds']))
    total = metrics['output_tokens']
    if type(n) is not int or n < 256 or not wall > 0 or type(total) is not int or total <= 0:
        raise ValueError('full 256-row positive throughput measurement required')
    means, counted, tokens = {}, 0, 0
    for key, value in metrics['per_cell'].items():
        count, output = value['count'], value['output_tokens']
        if type(count) is not int or count <= 0 or type(output) is not int or not 0 <= output <= count * 4096:
            raise ValueError('invalid per-cell measurement')
        means[key] = output / count
        counted += count
        tokens += output
    if counted != n or tokens != total:
        raise ValueError('measurement token/count totals disagree')
    return means, total / wall


def stage_gate(rows, metrics, *, remaining, hourly, seconds_left):
    means, speed = measurement(metrics)
    if not rows or any(cell(row) not in means for row in rows):
        raise ValueError('each released cell needs measured token lengths')
    seconds = 1.2 * sum(means[cell(row)] for row in rows) / speed
    cost = seconds * hourly / 3600
    available = max(0, remaining - RESERVE)
    fits = cost <= available and seconds + 90 <= seconds_left
    return {'decision': 'run' if fits else 'drop', 'rows': len(rows),
            'projected_usd': cost, 'available_after_reserve_usd': available,
            'shortfall_usd': max(0, cost - available), 'projected_seconds': seconds,
            'tokens_per_second': speed, 'mean_output_tokens_by_cell': means,
            'generation_margin': 1.2, 'outcomes_consulted': False,
            'decided_unix': time.time()}


def smoke_protocol(directory, metrics):
    """Token-boundary validation only; no numeric/content adherence scoring."""
    rows = [json.loads(line) for part in metrics['chunks']
            for line in (Path(directory) / part['path']).read_text().splitlines()]
    if len(rows) != 8 or len({r['id'] for r in rows}) != 8:
        raise ValueError('exactly eight distinct runtime-smoke replies required')
    off_boundaries = 0
    for row in rows:
        ids = row['output_token_ids']
        if not ids or len(ids) > 4096 or row['output_tokens'] != len(ids):
            raise ValueError('malformed smoke token count')
        if row['finish_reason'] == 'length' and len(ids) != 4096:
            raise ValueError('smoke cap provenance mismatch')
        if row['thinking'] and 248069 not in ids and row['finish_reason'] != 'length':
            raise ValueError('native thinking smoke stopped without a closing boundary')
        if not row['thinking'] and any(t in ids for t in (248068, 248069)):
            off_boundaries += 1
    # DESIGN deviation 8219501 permits off-mode boundaries and scores after
    # the last closer. Presence is diagnostic, never an analysis/release gate.
    return {'requests': 8, 'protocol_checked': True, 'violation_scored': False,
            'off_replies_with_thought_boundaries': off_boundaries}


class Connection:
    def __init__(self, row):
        self.host, self.port = row['ssh_host'], int(row['ssh_port'])

    def ssh(self, command, timeout=50):
        return run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                    '-o', 'StrictHostKeyChecking=accept-new', '-p', str(self.port),
                    'root@' + self.host, command], timeout=timeout, input='')

    def put(self, source, target):
        return run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                    '-o', 'StrictHostKeyChecking=accept-new', '-P', str(self.port),
                    str(source), 'root@' + self.host + ':' + target], timeout=60)

    def pull(self, local):
        """Pull each immutable completed chunk once, replace only small status/logs."""
        local = Path(local)
        local.mkdir(parents=True, exist_ok=True)
        listing = json.loads(self.ssh("python3 -c " + shlex.quote(
            "import json,pathlib,hashlib;d=pathlib.Path('/phase6/output');"
            "print(json.dumps([{'name':str(p.relative_to(d)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} "
            "for p in d.rglob('*') if p.is_file() and not p.name.endswith(('.tmp','.incoming'))]))")))
        for info in listing:
            name = info['name']
            rel = Path(name)
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('unsafe remote artifact name')
            dest = local / rel
            if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == info['sha256']:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            temporary = dest.with_name(dest.name + '.download')
            run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-P', str(self.port),
                 'root@' + self.host + ':' + REMOTE + '/output/' + name, str(temporary)], timeout=60)
            mutable = name == 'progress.json' or name.endswith('.metrics.json')
            if mutable:
                # Atomically replaced status snapshots may legitimately advance
                # between the listing and SCP. They must still be whole JSON.
                json.loads(temporary.read_text())
            elif hashlib.sha256(temporary.read_bytes()).hexdigest() != info['sha256']:
                temporary.unlink(missing_ok=True)
                raise RuntimeError('artifact changed during retrieval or transfer checksum mismatch')
            temporary.replace(dest)
        # Keep startup diagnostics outside the worker's fresh output directory.
        run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-P', str(self.port),
             'root@' + self.host + ':' + REMOTE + '/worker.log', str(local / 'worker.log')], timeout=60)

    def permit(self, name, decision, local):
        path = Path(local) / (name + '.permit.json')
        atomic(path, decision)
        self.put(path, REMOTE + '/output/' + path.name + '.incoming')
        self.ssh('mv ' + REMOTE + '/output/' + path.name + '.incoming ' + REMOTE + '/output/' + path.name)


def save(ledger):
    ledger['updated_unix'] = time.time()
    atomic(LEDGER, ledger)


def spent(ledger):
    return sum(float(a.get('charged_usd', 0)) if a.get('zero_verified') else
               (time.time() - a['started_unix']) * a['costs']['hourly_usd'] / 3600
               + a['costs']['network_envelope_usd'] for a in ledger['attempts'])


def watchdog(path):
    """A separate process destroys only this Phase 6 label on deadline/parent loss."""
    path = Path(path)
    observed_owned = False
    while True:
        data = json.loads(path.read_text())
        if data.get('finished'):
            return
        alive = True
        try:
            os.kill(data['controller_pid'], 0)
        except ProcessLookupError:
            alive = False
        if not alive or time.time() >= data['deadline_unix']:
            try:
                rows = instances()
                for row in rows:
                    if row.get('label') == data['label']:
                        observed_owned = True
                        cli('destroy', 'instance', row['id'], '--yes', mutation=True)
                remaining = instances()
                owned = [r for r in remaining if r.get('label') == data['label']]
                known_create = observed_owned or data.get('instance_id') is not None
                atomic(path.with_name(path.stem + '-fired.json'),
                       {'at_unix': time.time(), 'instances_after': remaining, 'parent_alive': alive,
                        'zero_verified': not owned and known_create,
                        'create_reconciled': known_create})
                if not owned and known_create:
                    return
            except (RuntimeError, subprocess.SubprocessError):
                # Retain the watchdog through transient API failures; a failed
                # request is never teardown evidence.
                pass
        time.sleep(5)


def start_watchdog(label, deadline, local):
    config = Path(local) / 'watchdog.json'
    atomic(config, {'controller_pid': os.getpid(), 'label': label,
                    'deadline_unix': deadline, 'finished': False})
    with (Path(local) / 'watchdog.log').open('ab') as log:
        child = subprocess.Popen([sys.executable, '-m', 'src.phase6_vast_control', '--watchdog', str(config)],
                                 cwd=ROOT, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                                 start_new_session=True)
    return config, child.pid


def bundle(target, requests):
    with tarfile.open(target, 'w:gz') as archive:
        for folder in ('src', 'prompts'):
            for path in (ROOT / folder).glob('*'):
                if path.is_file() and path.suffix in ('.py', '.json', '.txt'):
                    archive.add(path, arcname=str(path.relative_to(ROOT)))
        archive.add(requests, arcname='requests.json')
        archive.add(ROOT / 'DESIGN.md', arcname='DESIGN.md')


def execute(args):
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if LEDGER.exists():
        ledger = json.loads(LEDGER.read_text())
        if any(not a.get('zero_verified') for a in ledger['attempts']):
            raise RuntimeError('unresolved prior Phase 6 rental; reconcile before relaunch')
    else:
        ledger = {'phase': 'phase6', 'cap_usd': CAP, 'initial_account': account(),
                  'registration_commit': '8af0156', 'attempts': [], 'stage_decisions': {},
                  'completed_stages': [], 'excluded_machines': []}
        save(ledger)
    if instances():
        raise RuntimeError('Vast must have zero instances before Phase 6 starts')
    spec = json.loads(Path(args.requests).read_text())
    # This function is adapted only once the generator publishes its interface.
    model_spec = spec[args.model] if args.model in spec else spec['models'][args.model]
    request_rows = model_spec['requests']
    if args.model == '3.5' and not all(
            s in ledger['completed_stages'] or
            ledger['stage_decisions'].get(s, {}).get('decision') == 'drop' for s in STAGES[:2]):
        raise RuntimeError('earlier stages must be completed or prospectively dropped before 3.5')
    if args.model + '-main' in ledger['completed_stages']:
        raise RuntimeError('completed model cannot be generated again')
    from src.vast_generate_6 import validate_spec
    validated_rows, _ = validate_spec(spec, args.model)
    if request_rows != validated_rows:
        raise ValueError('controller requests differ from the validated frozen grid')
    available = min(CAP - spent(ledger), float(account()['credit']))
    if available <= RESERVE + .5:
        raise RuntimeError('insufficient remaining Phase 6 allowance for setup')
    fresh = cli('search', 'offers',
                'gpu_ram>=48 num_gpus=1 reliability>=0.98 rentable=true verified=true '
                'disk_space>=100 cuda_vers>=13.0 cpu_arch=amd64 dph<=1.30',
                '--storage', DISK, '--limit', 200, '--order', 'dph_total')
    offer, costs = choose_offer(fresh, ledger['excluded_machines'])
    keys = ('id', 'machine_id', 'gpu_name', 'gpu_ram', 'num_gpus', 'reliability',
            'dph_base', 'dph_total', 'storage_cost', 'inet_down_cost', 'inet_up_cost',
            'cuda_max_good', 'driver_version', 'verification', 'rentable')
    number = len(ledger['attempts']) + 1
    label = f'phase6-{args.model}-a{number}-20261021'
    local = ROOT / 'results/raw' / label
    local.mkdir(parents=True, exist_ok=False)
    now = time.time()
    seconds = math.floor((available - RESERVE - costs['network_envelope_usd']) * 3600 / costs['hourly_usd']) - 90
    if seconds < 1200:
        raise RuntimeError('less than 1200 seconds including setup available; no rental')
    deadline = now + seconds
    attempt = {'label': label, 'model': args.model, 'started_unix': now,
               'offer': {k: offer[k] for k in keys if k in offer}, 'costs': costs,
               'deadline_unix': deadline, 'zero_verified': False, 'status': 'reserved'}
    ledger['attempts'].append(attempt)
    save(ledger)
    watch, watchpid = start_watchdog(label, deadline, local)
    attempt['watchdog_pid'] = watchpid
    save(ledger)
    conn = None
    try:
        reply = cli('create', 'instance', offer['id'], '--image', IMAGE, '--disk', DISK,
                    '--label', label, '--ssh', '--direct', '--cancel-unavail',
                    '--env', '-e HF_HOME=/hf -e VLLM_USE_FLASHINFER_SAMPLER=0', mutation=True)
        attempt['create_response'] = {k: reply[k] for k in ('success', 'new_contract') if k in reply}
        save(ledger)
        if reply.get('success') is not True or not isinstance(reply.get('new_contract'), int):
            raise RuntimeError('create unsuccessful; inspect label before any new attempt')
        attempt['instance_id'] = reply['new_contract']
        atomic(watch, {'controller_pid': os.getpid(), 'label': label,
                       'deadline_unix': deadline, 'instance_id': attempt['instance_id'], 'finished': False})
        save(ledger)
        ready_limit = min(deadline, now + 1200)
        while time.time() < ready_limit:
            owned = [r for r in instances() if r.get('label') == label]
            if owned and owned[0].get('ssh_host') and owned[0].get('ssh_port'):
                possible = Connection(owned[0])
                try:
                    receipt = json.loads(possible.ssh("python3 -c " + shlex.quote(
                        "import json,sys,torch,vllm;print(json.dumps({'python':sys.version,"
                        "'torch':torch.__version__,'cuda':torch.version.cuda,'vllm':vllm.__version__,"
                        "'gpu':torch.cuda.get_device_name(0),'memory':torch.cuda.get_device_properties(0).total_memory}))"), timeout=45))
                    if receipt['vllm'] != '0.32.0' or not receipt['torch'].startswith('2.13.0'):
                        raise RuntimeError('pinned container version drift')
                    atomic(local / 'runtime-receipt.json', receipt)
                    conn = possible
                    break
                except (subprocess.SubprocessError, json.JSONDecodeError):
                    pass
            time.sleep(12)
        if conn is None:
            raise RuntimeError('image/SSH/import setup did not finish within 1200 seconds')
        attempt['ssh_ready_unix'] = time.time()
        save(ledger)
        archive = local / 'bundle.tar.gz'
        bundle(archive, args.requests)
        attempt['bundle_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
        conn.ssh('mkdir -p /phase6/output')
        conn.put(archive, '/phase6/bundle.tar.gz')
        conn.ssh('cd /phase6 && tar xzf bundle.tar.gz')
        seq = 256 if offer['gpu_ram'] >= 80000 else 64
        launch = ("import os,subprocess;log=open('/phase6/worker.log','ab');"
                  "env=dict(os.environ,HF_HOME='/hf',VLLM_USE_FLASHINFER_SAMPLER='0');"
                  "p=subprocess.Popen(" + repr(['python3', '-m', 'src.vast_generate_6',
                   '--requests', '/phase6/requests.json', '--model', args.model,
                   '--output-dir', '/phase6/output', '--max-num-seqs', str(seq),
                   '--deadline-unix', str(deadline)]) +
                  ",cwd='/phase6',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True);print(p.pid)")
        attempt['remote_supervisor_pid'] = int(conn.ssh('python3 -c ' + shlex.quote(launch)))
        guard_launch = ("import subprocess;log=open('/phase6/guard.log','ab');p=subprocess.Popen("
                        "['python3','-m','src.phase6_resource_guard','--output-dir','/phase6/output'],"
                        "cwd='/phase6',stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True);print(p.pid)")
        attempt['remote_transfer_guard_pid'] = int(conn.ssh('python3 -c ' + shlex.quote(guard_launch)))
        attempt['status'] = 'worker_started'
        save(ledger)
        next_stage = 'smoke'
        measurement_deadline = None
        last_progress = None
        while time.time() < deadline - 45:
            conn.pull(local)
            error = next((local / name for name in ('error.json', 'worker.error.json', 'worker.timeout.json')
                          if (local / name).exists()), None)
            if error:
                attempt['worker_error'] = json.loads(error.read_text())
                raise RuntimeError('worker failed; see pulled error/log artifacts')
            progress = (local / 'progress.json').read_text() if (local / 'progress.json').exists() else None
            if progress != last_progress:
                print(json.dumps({'model': args.model, 'progress': json.loads(progress) if progress else None,
                                  'spent_envelope_usd': spent(ledger)}), flush=True)
                last_progress = progress
            done = local / (next_stage + '.done.json')
            if done.exists():
                metrics = json.loads(done.read_text())
                atomic(EVIDENCE / f'{label}-{next_stage}.json', metrics)
                if next_stage == 'smoke':
                    if metrics['requests'] != 8:
                        raise ValueError('exact eight-prompt unscored smoke required')
                    attempt['smoke_protocol'] = smoke_protocol(local, metrics)
                    allowance = 900
                    if spent(ledger) + allowance * costs['hourly_usd'] / 3600 + RESERVE > CAP:
                        raise RuntimeError('full throughput measurement allowance does not fit')
                    conn.permit('throughput', {'decision': 'run'}, local)
                    attempt['measurement_started_unix'] = time.time()
                    measurement_deadline = time.time() + allowance
                    next_stage = 'throughput'
                elif next_stage == 'throughput':
                    metrics['wall_seconds'] = max(metrics['generation_seconds'],
                        time.time() - attempt['measurement_started_unix'])
                    measurement(metrics)
                    attempt['measurement'] = metrics
                    next_stage = 'main'
                    selected = [r for r in request_rows if r['arm'] == 'main']
                    decision = stage_gate(selected, metrics, remaining=CAP - spent(ledger),
                                          hourly=costs['hourly_usd'], seconds_left=deadline - time.time())
                    ledger['stage_decisions'][args.model + '-main'] = decision
                    save(ledger)
                    conn.permit('main', decision, local)
                    if decision['decision'] != 'run':
                        break
                elif next_stage == 'main':
                    if metrics['requests'] != 1600:
                        raise ValueError('main stage did not complete all 1600 replies')
                    ledger['completed_stages'].append(args.model + '-main')
                    save(ledger)
                    if args.model == '3.5':
                        break
                    next_stage = 'prefill'
                    selected = [r for r in request_rows if r['arm'] == 'prefill']
                    decision = stage_gate(selected, attempt['measurement'], remaining=CAP - spent(ledger),
                                          hourly=costs['hourly_usd'], seconds_left=deadline - time.time())
                    ledger['stage_decisions']['3.8-prefill'] = decision
                    save(ledger)
                    conn.permit('prefill', decision, local)
                    if decision['decision'] != 'run':
                        break
                else:
                    if metrics['requests'] != 1200:
                        raise ValueError('prefill stage did not complete all 1200 replies')
                    ledger['completed_stages'].append('3.8-prefill')
                    save(ledger)
                    break
            if spent(ledger) + RESERVE >= CAP:
                raise RuntimeError('independent Phase 6 spend stop')
            if next_stage == 'throughput' and time.time() >= measurement_deadline:
                raise RuntimeError('full-batch throughput measurement exceeded 900-second allowance')
            time.sleep(12)
        else:
            raise RuntimeError('absolute deadline before stage completion')
        attempt['status'] = 'completed_selected_stages'
    except BaseException as exc:
        attempt['status'] = 'failed'
        attempt['error_type'] = type(exc).__name__
        attempt['error'] = str(exc) if not isinstance(exc, subprocess.SubprocessError) else 'subprocess failure, inspect artifacts'
        ledger['excluded_machines'].append(offer['machine_id'])
        save(ledger)
        raise
    finally:
        if conn:
            try:
                conn.pull(local)
            except Exception:
                attempt['last_pull_failed'] = True
        # Teardown always owns only this unique label, never Phase 5 resources.
        for row in instances():
            if row.get('label') == label:
                cli('destroy', 'instance', row['id'], '--yes', mutation=True)
        zero = instances()
        owned = [r for r in zero if r.get('label') == label]
        if owned or not attempt.get('instance_id'):
            attempt['create_reconciled'] = bool(attempt.get('instance_id'))
            attempt['reserved_exposure_usd'] = available - RESERVE
            save(ledger)
            raise RuntimeError('Phase 6 teardown/create unresolved; independent watchdog stays armed')
        ended = time.time()
        attempt.update(ended_unix=ended, zero_verified=True,
                       elapsed_seconds=ended - now,
                       charged_usd=(ended - now) * costs['hourly_usd'] / 3600 + costs['network_envelope_usd'])
        atomic(watch, {'finished': True})
        ledger['conservative_spend_usd'] = spent(ledger)
        ledger['final_account'] = account()
        ledger['zero_resources'] = {'checked_unix': time.time(), 'instances': zero}
        save(ledger)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watchdog')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--requests', type=Path)
    parser.add_argument('--model', choices=('3.8', '3.5'))
    args = parser.parse_args()
    if args.watchdog:
        watchdog(args.watchdog)
    elif args.execute and args.requests and args.model:
        execute(args)
    else:
        parser.error('execution requires --execute --requests PATH --model 3.8|3.5')


if __name__ == '__main__':
    main()
