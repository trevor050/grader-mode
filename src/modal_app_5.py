"""Ephemeral Gemma12B BF16 pilot. Default command is entirely local/dry.

Inspect without provisioning: uv run python -m src.modal_app_5 --dry-run
Parent-authorized pilot: uv run python -m src.modal_app_5 --execute --run-id ID
Do not use `modal run`: the CPU entrypoint owns reservation before app startup.
Recover committed artifacts: uv run modal volume get grader-mode-phase5-results ID OUT
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.phase5_control import (BudgetError, BudgetLedger, CAP_USD, LEDGER_PATH, ROOT,
                                atomic_json, require_pilot, safe_id, watchdog_deadline)

PILOT_TIMEOUT = 900
STARTUP_TIMEOUT = 120
TEARDOWN_SECONDS = 120
# Deliberately conservative all-in bound for fixed 4 CPU / 32 GiB plus L40S.
# This is an exposure bound, not a claim about settled provider billing.
ALL_IN_RATE = .00075
RESULT_VOLUME = 'grader-mode-phase5-results'
PINS = ('vllm==0.32.0', 'torch==2.13.0', 'transformers==5.19.0',
        'tokenizers==0.23.3', 'huggingface-hub==1.31.0', 'safetensors==0.8.0',
        'numpy==2.3.5', 'jinja2==3.1.6', 'datasets==4.8.3')

# Definitions are inert until app.run(); importing/building this spec needs no
# token, provisioning, model download or GPU package on the local machine.
import modal

app = modal.App('grader-mode-phase5-pilot')
hf_cache = modal.Volume.from_name('grader-mode-phase5-hf-cache', create_if_missing=True)
results = modal.Volume.from_name(RESULT_VOLUME, create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
         .uv_pip_install(*PINS)
         .env({'HF_HOME': '/hf', 'HF_XET_HIGH_PERFORMANCE': '1',
               'PHASE5_HOST': 'modal', 'PHASE5_GPU': 'L40S',
               'PHASE5_RUNTIME_IMAGE': 'modal-debian-slim-python3.12-vllm0.32.0',
               'VLLM_USE_FLASHINFER_SAMPLER': '0', 'DO_NOT_TRACK': '1'})
         .add_local_python_source('src')
         .add_local_dir(ROOT / 'prompts', remote_path='/root/prompts')
         .add_local_file(ROOT / 'DESIGN.md', remote_path='/root/DESIGN.md'))


def dry_config():
    return dict(pilot_only=True, gpu='L40S', model='google/gemma-4-12B-it',
                dtype='bfloat16', python='3.12', pins=list(PINS),
                timeout_seconds=PILOT_TIMEOUT, startup_seconds=STARTUP_TIMEOUT,
                teardown_seconds=TEARDOWN_SECONDS, min_containers=0, buffer_containers=0,
                max_containers=1, single_use_containers=True, retries=0,
                ledger=str(LEDGER_PATH), lifetime_cap_usd=CAP_USD,
                reservation_usd=(PILOT_TIMEOUT + STARTUP_TIMEOUT + TEARDOWN_SECONDS) * ALL_IN_RATE,
                result_volume=RESULT_VOLUME,
                generation_command=['python', '-m', 'src.vast_generate_5', '--model', 'gemma',
                                    '--pilot', '--output-dir', '/phase5/<run-id>'],
                full_run_requires_new_authorization=True)


def generation_command(directory, run_id):
    return [sys.executable, '-m', 'src.vast_generate_5', '--model', 'gemma',
            '--pilot', '--run-id', safe_id(run_id), '--output-dir', str(directory)]


def run_generation(directory, run_id, deadline_unix, checkpoint, *, command=None):
    """Keep all heavy imports/CUDA in a killable child, commit partial artifacts.

    `command` is dependency injection for CPU-only lifecycle tests. Each pilot
    writes into a fresh directory; failed attempts cannot overwrite evidence.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    state = dict(run_id=safe_id(run_id), status='running', pilot=True, started_unix=time.time())
    process = None

    def save():
        atomic_json(directory / 'pilot-execution.json', state)
        checkpoint()

    try:
        save()
        if time.time() >= deadline_unix:
            raise TimeoutError('pilot deadline already expired')
        with (directory / 'generation.log').open('xb') as log:
            process = subprocess.Popen(command or generation_command(directory, run_id),
                                       stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       start_new_session=True, cwd=ROOT)
            while True:
                remaining = deadline_unix - time.time()
                if remaining <= 0:
                    raise TimeoutError('absolute pilot deadline')
                try:
                    code = process.wait(timeout=min(5, remaining))
                    break
                except subprocess.TimeoutExpired:
                    save()
            if code:
                raise RuntimeError(f'generation subprocess exited {code}; see remote generation.log')
        metrics = {str(path.relative_to(directory)): json.loads(path.read_text())
                   for path in directory.rglob('*.metrics.json')}
        state.update(status='completed', returncode=code, metrics=metrics,
                     artifact_paths=sorted(str(path.relative_to(directory))
                                           for path in directory.rglob('*') if path.is_file()))
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        try:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
        except BaseException as error:
            state.update(status='failed', child_cleanup_error=f'{type(error).__name__}: {error}')
        finally:
            state['ended_unix'] = time.time()
            # Failed generation returns a durable error envelope. The local caller
            # pulls it before raising; partial JSONL/logs survive a normal exception.
            save()
    return state


@app.function(image=image, gpu='L40S', volumes={'/hf': hf_cache, '/phase5': results},
              timeout=900, startup_timeout=120, retries=0, min_containers=0,
              buffer_containers=0, max_containers=1, single_use_containers=True,
              scaledown_window=2, cpu=(4, 4), memory=(32768, 32768))
def pilot(run_id: str, deadline_unix: float, progress: modal.Dict):
    # Independent last-resort timer covers an SDK commit/model/driver hang. The
    # normal child-process deadline saves results before this timer can fire.
    hard_stop = threading.Timer(max(.01, deadline_unix - time.time() + 10), lambda: os._exit(124))
    hard_stop.daemon = True
    hard_stop.start()
    progress['state'] = {'started': True, 'unix': time.time()}
    try:
        state = run_generation(Path('/phase5') / safe_id(run_id), run_id, deadline_unix,
                               checkpoint=results.commit)
        progress['state'] = {'started': True, 'status': state['status']}
        return state
    finally:
        # HF volume contains cached weights only. Results live in /phase5 and
        # were committed separately, including exceptions/partial completions.
        results.commit()


def modal_cli(*args, timeout=20):
    completed = subprocess.run([sys.executable, '-m', 'modal', *args], check=True,
                               stdin=subprocess.DEVNULL, capture_output=True, text=True,
                               timeout=timeout)
    return completed.stdout


def stop_and_verify(app_id):
    errors = []
    deadline = time.monotonic() + 40
    stop_confirmed = False
    rows = None
    for _ in range(5):
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if not stop_confirmed:
                try:
                    modal_cli('app', 'stop', '--yes', app_id, timeout=min(5, remaining))
                    stop_confirmed = True
                except Exception as error:
                    errors.append(str(error))
                    # The CLI exits nonzero for an already-stopped app. Require
                    # an explicit stopped state, never infer it from idle GPUs.
                    try:
                        apps = json.loads(modal_cli('app', 'list', '--json',
                                          timeout=min(5, max(.1, deadline-time.monotonic()))))
                        stop_confirmed = any(a.get('app_id') == app_id and
                                             a.get('state') == 'stopped' for a in apps)
                    except Exception as state_error:
                        errors.append(str(state_error))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            rows = json.loads(modal_cli('container', 'list', '--app-id', app_id, '--json',
                                       timeout=min(5, remaining)))
            if stop_confirmed and rows == []:
                return True, dict(app_id=app_id, app_stop_confirmed=True,
                                  containers_after=rows, stop_errors=errors)
        except Exception as error:
            errors.append(str(error))
        time.sleep(1)
    return False, dict(app_id=app_id, app_stop_confirmed=stop_confirmed,
                       containers_after=rows, stop_errors=errors)


def modal_watchdog(attempt_id):
    ledger = BudgetLedger()
    while True:
        ticket = ledger.attempt(attempt_id)
        if ticket['teardown_verified']:
            return
        deadline = watchdog_deadline(ticket)
        if time.time() < deadline:
            time.sleep(min(1, deadline - time.time()))
            continue
        if ticket['resource_id']:
            zero, evidence = stop_and_verify(ticket['resource_id'])
            ledger.note(attempt_id, watchdog=evidence)
        else:
            # No GPU spawn is allowed before app ID binding. An uncertain app
            # allocation still stays charged and blocked for human reconciliation.
            zero = False
        ledger.settle(attempt_id, actual_usd=ticket['reserved_usd'], teardown_verified=zero)
        if zero or not ticket['resource_id']:
            return
        time.sleep(2)


def execute_pilot(run_id, *, ledger=None):
    safe_id(run_id)
    ledger = ledger or BudgetLedger()
    if ledger.path != LEDGER_PATH.resolve():
        raise BudgetError('live Modal pilot must use the single shared aggregate ledger')
    # Before any app startup: generation adapter must actually be present.
    if not (ROOT / 'src/vast_generate_5.py').exists():
        raise BudgetError('Phase 5 generation adapter not available yet')
    ticket = ledger.reserve(run_id, provider='modal', hourly_usd=ALL_IN_RATE * 3600,
                            runtime_seconds=PILOT_TIMEOUT, startup_seconds=STARTUP_TIMEOUT,
                            teardown_seconds=TEARDOWN_SECONDS)
    out = ROOT / 'results/raw' / run_id
    if out.exists():
        ledger.settle(run_id, teardown_verified=True)
        raise FileExistsError(out)
    out.mkdir(parents=True)
    with (out / 'watchdog.log').open('ab') as log:
        watchdog = subprocess.Popen([sys.executable, '-m', 'src.modal_app_5', '--watchdog', run_id],
                                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                    start_new_session=True, close_fds=True)
    ledger.note(run_id, watchdog_pid=watchdog.pid)
    call = result = app_id = caught = None
    zero = False
    try:
        with modal.enable_output(), app.run():
            app_id = app.app_id
            ledger.bind_resource(run_id, app_id)
            try:
                with modal.Dict.ephemeral() as progress:
                    # Exactly one GPU function invocation, SDK retries disabled.
                    call = pilot.spawn(run_id, ticket['work_deadline_unix'] - 15, progress)
                    ledger.note(run_id, function_call_id=call.object_id)
                    startup_recorded = False
                    while True:
                        now = time.time()
                        state = progress.get('state')
                        if state and not startup_recorded:
                            ledger.note(run_id, startup_complete=True)
                            startup_recorded = True
                        if not state and now >= ticket['startup_deadline_unix']:
                            raise TimeoutError('Modal bounded startup deadline exceeded')
                        if now >= ticket['work_deadline_unix'] - 20:
                            raise TimeoutError('local Modal absolute pilot deadline')
                        try:
                            result = call.get(timeout=0)
                            break
                        except TimeoutError:
                            time.sleep(1)
                if result.get('status') != 'completed':
                    raise RuntimeError(f'pilot failed; committed output in {RESULT_VOLUME}/{run_id}')
            finally:
                try:
                    # Pull immediately on success, and committed partial outputs
                    # on exceptions; no analysis on paid GPU after generation.
                    modal_cli('volume', 'get', RESULT_VOLUME, run_id, str(out), '--force', timeout=30)
                finally:
                    if call is not None:
                        try:
                            call.cancel(terminate_containers=True)
                        except Exception as error:
                            ledger.note(run_id, cancel_error=str(error))
                    zero, evidence = stop_and_verify(app_id)
                    ledger.note(run_id, teardown=evidence)
    except BaseException as error:
        caught = error
    finally:
        if app_id and not zero:
            zero, evidence = stop_and_verify(app_id)
            ledger.note(run_id, final_teardown=evidence)
        ledger.settle(run_id, teardown_verified=zero, success=caught is None and result is not None)
        atomic_json(out / 'local-execution.json', dict(run_id=run_id, app_id=app_id, result=result,
                    zero_containers_verified=zero, error=None if caught is None else str(caught)))
    if caught is not None:
        raise caught
    if not zero:
        raise BudgetError('Modal teardown unresolved; reservation charged and future launches blocked')
    print(json.dumps(result, allow_nan=False))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--run-id', default='phase5-gemma-pilot')
    parser.add_argument('--watchdog', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    require_pilot(args.full)
    if args.watchdog:
        return modal_watchdog(args.watchdog)
    if args.execute and args.dry_run:
        raise ValueError('choose dry-run or execute')
    if args.execute:
        return execute_pilot(args.run_id)
    print(json.dumps(dry_config(), indent=2))


if __name__ == '__main__':
    main()
