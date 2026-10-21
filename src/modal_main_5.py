"""One authorized Gemma main attempt, full-batch prospective gate, $3 lifetime cap.

No scoring. The historical pilot module/API remains untouched. Default is dry.
Run: uv run python -m src.modal_main_5 --execute --run-id FRESH_ID
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

from src.models_5 import GEMMA
from src.phase5_control import (BudgetError, BudgetLedger, LEDGER_PATH, ROOT,
                                atomic_json, positive, safe_id, watchdog_deadline)
from src.modal_app_5 import (image, hf_cache, results, RESULT_VOLUME,
                             modal_cli, stop_and_verify, ALL_IN_RATE)
import modal

MODEL_CAP = 3.0
PRIOR_PILOT = .282284
RESERVE = .30
BATCH_SIZE = 256
CHUNK_SIZE = 256
CONTROLS = ('gemma/prefill/neutral', 'gemma/prefill/irrelevant')
EVIDENCE = ROOT / 'results/phase5'
app = modal.App('grader-mode-phase5-gemma-main')


def cell_id(row):
    if row['model'] != GEMMA:
        raise ValueError('Gemma only')
    if row['arm'] == 'prefill':
        return f"gemma/prefill/{row['prefill']}"
    return f"gemma/main/{'on' if row['thinking'] else 'off'}/{row['request_type']}"


def model_prior_spend(snapshot):
    # Conservatively attribute every prior Phase 5 Modal charge to Gemma. No
    # credits, reserve releases, refunds or failed attempts reduce this floor.
    return max(PRIOR_PILOT, sum(a['charged_microusd'] for a in snapshot['attempts']
                              if a['provider'] == 'modal') / 1e6)


def deadline_allocation(prior_usd):
    prior = positive(prior_usd, 'prior spend', zero=True)
    # $0.075 stays outside the ticket for delayed/storage/egress charges.
    total_seconds = math.floor((MODEL_CAP - prior - .075) / ALL_IN_RATE)
    startup, teardown = 180, 120
    runtime = total_seconds - startup - teardown
    if runtime < 600:
        raise BudgetError('remaining model budget cannot cover startup, measurement and teardown')
    return dict(startup_seconds=startup, runtime_seconds=runtime,
                teardown_seconds=teardown, reserved_usd=total_seconds * ALL_IN_RATE,
                external_delayed_charge_buffer_usd=.075)


def project_gemma(progress, remaining, pilot_means, *, prior_usd, current_seconds):
    """Use timing/length metadata only, before any neutral/irrelevant generation."""
    if progress['requests'] < 256:
        raise ValueError('at least 256 full-batch main rows required')
    if any(progress['per_cell'].get(c, {}).get('count', 0) for c in CONTROLS):
        raise ValueError('control generation already began; prospective drop forbidden')
    speed = positive(progress['output_tokens'] / positive(progress['generation_seconds'], 'seconds'), 'throughput')
    observed = {c: v['output_tokens'] / v['count'] for c, v in progress['per_cell'].items()}
    on_floor = max((v for c, v in observed.items() if '/on/' in c), default=0)
    counts = Counter(cell_id(r) for r in remaining)
    expected = {f'gemma/main/{t}/{r}' for t in ('off','on') for r in ('plain','explicit')}
    expected |= {f'gemma/prefill/{p}' for p in ('empty','reminder','neutral','irrelevant')}
    if set(pilot_means) != expected:
        raise ValueError('exact eight registered pilot cell means required')
    means = {}
    for c in expected:
        pilot = positive(pilot_means[c], 'pilot mean', zero=True)
        value = max(pilot, observed.get(c, on_floor if '/prefill/' in c else 4096))
        if value > 4096:
            raise ValueError('length exceeds fixed output cap')
        means[c] = value
    prior = positive(prior_usd, 'prior spend', zero=True)
    elapsed_cost = positive(current_seconds, 'current elapsed', zero=True) * ALL_IN_RATE
    projections = []
    for drops in ((), CONTROLS):
        tokens = sum(n * means[c] for c,n in counts.items() if c not in drops)
        seconds = tokens / speed
        total = prior + elapsed_cost + seconds * 1.20 * ALL_IN_RATE + RESERVE
        projections.append(dict(dropped_cells=list(drops), remaining_output_tokens=tokens,
                                remaining_generation_seconds=seconds, projected_total_usd=total))
        if total <= MODEL_CAP:
            decision = 'full' if not drops else 'drop_controls'
            break
    else:
        decision = 'stop_noninferential'
    selected = projections[-1]
    return dict(decision=decision, cap_usd=MODEL_CAP, prior_gemma_usd=prior,
                current_attempt_elapsed_seconds=current_seconds,
                current_attempt_cost_bound_usd=elapsed_cost,
                measured_rows=progress['requests'], measured_output_tokens=progress['output_tokens'],
                measured_generation_seconds=progress['generation_seconds'],
                measured_tokens_per_second=speed, setup_seconds=progress['setup_seconds'],
                mean_tokens_by_cell=means, observed_means_by_cell=observed,
                pilot_means_by_cell=pilot_means, hourly_usd=ALL_IN_RATE * 3600,
                runtime_margin=1.20, teardown_and_failure_reserve_usd=RESERVE,
                dropped_cells=selected['dropped_cells'],
                remaining_rows=sum(n for c,n in counts.items() if c not in selected['dropped_cells']),
                projected_total_usd=selected['projected_total_usd'], projections=projections,
                controls_generated_before_decision=0, decision_unix=time.time(),
                inference_eligible=False, outcomes_read_or_scored=False)


def source_hashes():
    names = ('DESIGN.md', 'prompts/templates.json', 'prompts/templates_4.json',
             'prompts/system_tutor.txt', 'src/models_5.py', 'src/phase5_split.py',
             'src/vast_generate_5.py', 'src/modal_main_5.py', 'src/modal_app_5.py',
             'src/phase5_control.py')
    return {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}


def worker(payload_path):
    """Killable GPU child. Each JSONL flush precedes an explicit Volume commit."""
    from src.vast_generate_5 import generate
    payload = json.loads(Path(payload_path).read_text())
    directory = Path(payload_path).parent
    gate_path = directory / 'budget-projection.json'
    manifest = payload['manifest']
    manifest['stage'] = 'main'
    manifest['execution_source_sha256'] = payload['source_hashes']
    manifest['implementation_sha256'].update({k:v for k,v in payload['source_hashes'].items()
                                            if k.startswith('src/')})

    def before():
        if time.time() >= payload['work_deadline_unix'] - 20:
            raise TimeoutError('absolute Gemma work deadline; partial data are noninferential')

    def gate(progress, remaining):
        projection = project_gemma(progress, remaining, payload['pilot_means'],
                                  prior_usd=payload['prior_usd'],
                                  current_seconds=time.time() - payload['started_unix'])
        atomic_json(gate_path, projection)
        manifest['gemma_budget_gate'] = projection
        manifest['dropped_cells'] = projection['dropped_cells']
        atomic_json(directory / 'manifest.json', manifest)
        results.commit()
        print('GEMMA_MAIN_COST_PROJECTION ' + json.dumps(projection, allow_nan=False), flush=True)
        if projection['decision'] == 'stop_noninferential':
            raise BudgetError('minimum mandatory Gemma grid does not fit model-specific $3 cap')
        return projection['dropped_cells']

    atomic_json(directory / 'manifest.json', manifest)
    stats = generate(GEMMA, payload['requests'], payload['run_id'], manifest,
                     dict(output_dir=str(directory), batch_size=BATCH_SIZE, chunk_size=CHUNK_SIZE,
                          enforce_eager=False, ordering='main_priority', checkpoint=results.commit,
                          first_chunk_gate=gate, before_chunk=before, host='modal', gpu='L40S'))
    counts = {k:v['count'] for k,v in stats['per_cell'].items()}
    expected = set(payload['pilot_means']) - set(stats['dropped_cells'])
    if set(counts) != expected or any(n != 400 for n in counts.values()):
        raise RuntimeError('main grid incomplete; no inference permitted')
    atomic_json(directory / 'completion.json', dict(status='completed', per_cell_counts=counts,
                rows=stats['requests'], complete_selected_grid=True, inference_eligible=False,
                human_audit_pending=True, ended_unix=time.time()))
    results.commit()


@app.function(image=image, gpu='L40S', volumes={'/hf':hf_cache, '/phase5':results},
              timeout=3600, startup_timeout=180, retries=0, min_containers=0,
              buffer_containers=0, max_containers=1, single_use_containers=True,
              scaledown_window=2, cpu=(4,4), memory=(32768,32768))
def main_gpu(payload: dict, progress: modal.Dict):
    directory = Path('/phase5') / safe_id(payload['run_id'])
    directory.mkdir(exist_ok=False)
    atomic_json(directory/'input.json', payload)
    results.commit()
    hard_stop = threading.Timer(max(.01, payload['work_deadline_unix'] - time.time() + 5),
                                lambda: os._exit(124))
    hard_stop.daemon = True
    hard_stop.start()
    progress['state'] = dict(started=True, unix=time.time())
    state = dict(run_id=payload['run_id'], status='running', inference_eligible=False)
    process = None
    try:
        with (directory/'generation.log').open('xb') as log:
            process = subprocess.Popen([sys.executable, '-m', 'src.modal_main_5', '--worker',
                                        str(directory/'input.json')], cwd=ROOT,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            while process.poll() is None:
                if time.time() >= payload['work_deadline_unix'] - 10:
                    raise TimeoutError('remote absolute deadline')
                update = dict(started=True, unix=time.time())
                for name,key in (('gemma-4-12B-it.progress.json','generation'),
                                 ('budget-projection.json','projection')):
                    path = directory/name
                    if path.exists():
                        try:
                            item = json.loads(path.read_text())
                            update[key] = item if key=='projection' else {k:item[k] for k in
                                ('requests','output_tokens','generation_seconds','elapsed_seconds','per_cell')}
                        except json.JSONDecodeError:
                            pass
                progress['state'] = update
                time.sleep(2)
            state['returncode'] = process.returncode
            state['status'] = 'completed' if process.returncode == 0 else 'failed_noninferential'
    except BaseException as error:
        state.update(status='failed_noninferential', error=f'{type(error).__name__}: {error}')
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=8)
        state['ended_unix'] = time.time()
        # Operational extraction only. No final text, correctness or violation reads.
        log = (directory/'generation.log').read_text(errors='replace')
        state['kv_capacity_log'] = [s for s in log.splitlines() if
                                   re.search(r'KV cache size|Available KV cache|Maximum concurrency', s)]
        atomic_json(directory/'main-execution.json', state)
        results.commit()
        progress['state'] = dict(started=True, status=state['status'])
        hard_stop.cancel()
    return state


def watchdog(run_id):
    ledger = BudgetLedger()
    while True:
        ticket = ledger.attempt(run_id)
        if ticket['teardown_verified']:
            return
        deadline = watchdog_deadline(ticket)
        if time.time() < deadline:
            time.sleep(min(1, deadline-time.time()))
            continue
        zero, evidence = stop_and_verify(ticket['resource_id']) if ticket['resource_id'] else (False, {})
        ledger.note(run_id, gemma_main_watchdog=evidence)
        ledger.settle(run_id, actual_usd=ticket['reserved_usd'], teardown_verified=zero)
        if zero or not ticket['resource_id']:
            return
        time.sleep(2)


def execute(run_id):
    from src.vast_generate_5 import build
    safe_id(run_id)
    ledger = BudgetLedger()
    if ledger.path != LEDGER_PATH.resolve():
        raise BudgetError('single shared aggregate ledger required')
    snapshot = ledger.snapshot()
    prior = model_prior_spend(snapshot)
    allocation = deadline_allocation(prior)
    apps = json.loads(modal_cli('app','list','--json'))
    containers = json.loads(modal_cli('container','list','--json'))
    if containers or any(a['state'] not in ('stopped','disabled') for a in apps):
        raise BudgetError('existing Modal resources; no concurrent paid runs allowed')
    rows, manifest = build('gemma', 200)
    if len(rows) != 3200:
        raise ValueError('exact full 3200 registered grid required')
    frozen = source_hashes()
    pilot = json.loads((ROOT/'results/phase5-pilot/pilot-summary.json').read_text())
    pilot_means = pilot['models']['gemma']['mean_generated_tokens_by_cell']
    out = ROOT/'results/raw'/run_id
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    ticket = ledger.reserve(run_id, provider='modal', hourly_usd=ALL_IN_RATE*3600,
                            runtime_seconds=allocation['runtime_seconds'],
                            startup_seconds=allocation['startup_seconds'],
                            teardown_seconds=allocation['teardown_seconds'])
    # Local absolute deadline includes queue, image/model startup, failures and teardown.
    payload = dict(requests=rows, manifest=manifest, run_id=run_id, source_hashes=frozen,
                   pilot_means=pilot_means, prior_usd=prior, started_unix=ticket['started_unix'],
                   work_deadline_unix=ticket['work_deadline_unix'])
    atomic_json(EVIDENCE/'gemma-execution-launch.json', dict(run_id=run_id, ticket=ticket,
                model_cap_usd=MODEL_CAP, prior_gemma_usd=prior, allocation=allocation,
                source_sha256=frozen, modal_apps_before=apps, modal_containers_before=containers,
                registered_rows=len(rows), sampling=manifest['sampling'],
                engine=dict(batch_size=256, chunk_size=256, enforce_eager=False, max_model_len=8192)))
    with (out/'watchdog.log').open('ab') as log:
        child = subprocess.Popen([sys.executable, '-m', 'src.modal_main_5', '--watchdog',run_id],
                                 cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 start_new_session=True, close_fds=True)
    ledger.note(run_id, watchdog_pid=child.pid, model_cap_usd=MODEL_CAP,
                prior_gemma_usd=prior, no_retries=True)
    call = result = app_id = caught = None
    zero = pulled = False
    last_count = -1
    projection_seen = False
    try:
        with modal.enable_output(), app.run():
            app_id = app.app_id
            ledger.bind_resource(run_id, app_id)
            with modal.Dict.ephemeral() as progress:
                call = main_gpu.spawn(payload, progress)
                ledger.note(run_id, function_call_id=call.object_id)
                started = False
                while True:
                    now = time.time()
                    state = progress.get('state') or {}
                    if state.get('started') and not started:
                        ledger.note(run_id, startup_complete=True)
                        started = True
                    if not started and now >= ticket['startup_deadline_unix']:
                        raise TimeoutError('bounded Modal startup deadline')
                    if now >= ticket['work_deadline_unix'] - 15:
                        raise TimeoutError('local absolute Gemma deadline')
                    if state.get('projection') and not projection_seen:
                        projection = state['projection']
                        atomic_json(EVIDENCE/'gemma-execution-projection.json', projection)
                        print('GEMMA_MAIN_COST_PROJECTION ' + json.dumps(projection), flush=True)
                        ledger.note(run_id, prospective_budget_decision=projection)
                        projection_seen = True
                    count = state.get('generation',{}).get('requests',0)
                    if count != last_count:
                        print('GEMMA_MAIN_PROGRESS '+json.dumps(state), flush=True)
                        last_count = count
                    try:
                        result = call.get(timeout=0)
                        break
                    except TimeoutError:
                        time.sleep(2)
            if result['status'] != 'completed':
                raise RuntimeError('Gemma main did not complete; partial data are noninferential')
    except BaseException as error:
        caught = error
    finally:
        if call is not None:
            try:
                call.cancel(terminate_containers=True)
            except Exception as error:
                ledger.note(run_id, cancel_error=str(error))
        if app_id:
            zero, evidence = stop_and_verify(app_id)
            ledger.note(run_id, teardown=evidence)
        # Stop GPU first. Committed Volume artifacts can be downloaded after teardown.
        try:
            modal_cli('volume','get',RESULT_VOLUME,run_id,str(out),'--force',timeout=90)
            pulled = True
        except Exception as error:
            caught = caught or error
        ledger.settle(run_id, teardown_verified=zero, success=caught is None and result is not None)
        status = dict(run_id=run_id, app_id=app_id, result=result,
                      zero_containers_verified=zero, artifacts_pulled=pulled,
                      raw_directory=str(out), inference_eligible=False, human_audit_pending=True,
                      error=None if caught is None else str(caught), ended_unix=time.time())
        atomic_json(EVIDENCE/'gemma-execution.json',status)
        for name in ('budget-projection.json','completion.json','main-execution.json',
                     'gemma-4-12B-it.metrics.json','manifest.json'):
            matches = list(out.rglob(name))
            if matches:
                suffix = {'budget-projection.json':'projection','completion.json':'completion',
                          'main-execution.json':'remote','gemma-4-12B-it.metrics.json':'metrics',
                          'manifest.json':'manifest'}[name]
                atomic_json(EVIDENCE/f'gemma-execution-{suffix}.json',json.loads(matches[0].read_text()))
        atomic_json(EVIDENCE/'gemma-execution-zero-resources.json', dict(
            modal_apps=json.loads(modal_cli('app','list','--json')),
            modal_containers=json.loads(modal_cli('container','list','--json')),
            captured_unix=time.time(), source_hashes_unchanged=frozen == source_hashes()))
    if caught:
        raise caught
    if not zero:
        raise BudgetError('teardown unresolved; reservation retained')
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--run-id',default='phase5-gemma-main-20261021-a')
    parser.add_argument('--watchdog',help=argparse.SUPPRESS)
    parser.add_argument('--worker',help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.watchdog:
        return watchdog(args.watchdog)
    if args.worker:
        return worker(args.worker)
    if args.execute:
        return execute(args.run_id)
    snapshot = BudgetLedger().snapshot()
    print(json.dumps(dict(authorized_model='gemma', full_grid_rows=3200, cap_usd=MODEL_CAP,
          prior_usd=model_prior_spend(snapshot), allocation=deadline_allocation(model_prior_spend(snapshot)),
          dry_run=True, enforce_eager=False, max_num_seqs=256, chunk_size=256),indent=2))


if __name__ == '__main__':
    main()
