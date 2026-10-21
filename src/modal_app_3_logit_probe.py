"""Explicit parent-reviewed launch only; imports never submit GPU work.

CPU preflight: uv run python -m src.phase3_logit_probe --run-id <fresh-id>
Launch AFTER explorer shutdown: uv run modal run -m src.modal_app_3_logit_probe --run-id <fresh-id>
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import modal

from src.modal_app_3 import image, hf_cache, volume, write_json
from src.modal_app_3_resume import require_remote_fresh, recover_artifacts
from src.models import MODELS
from src.phase3 import MODEL, RATE
from src.phase3_resume import validate_no_containers
from src.phase3_logit_probe import (SOURCE, EXPLORER, CELLS, PATHS, PROBE_USD,
    STARTUP_SECONDS, SHUTDOWN_SECONDS, HEADROOM, probe_limits, grid_fits,
    close_allocation, condition, digest, file_hash, last_position_hook,
    lifetime_seconds, load_package, prepare_prefix, log_probs, contrast_stats,
    require, validate_package, verify_hashes)

app = modal.App('grader-mode-phase3-logit-probe')


@app.function(image=image, gpu='L40S', volumes={'/hf': hf_cache, '/study': volume},
              timeout=int(PROBE_USD/RATE-STARTUP_SECONDS-SHUTDOWN_SECONDS),
              startup_timeout=STARTUP_SECONDS, retries=0, single_use_containers=True,
              min_containers=0, buffer_containers=0, max_containers=1,
              scaledown_window=2, cpu=4, memory=32768)
def probe(package: dict, manifest: dict, run_id: str, progress: modal.Dict,
          allocation_start: float, absolute_deadline_unix: float):
    started = time.monotonic()
    validate_package(package, rebuild_requests=False)
    limits = probe_limits(package['previous_seconds'], allocation_start)
    require(manifest['payload_sha256'] == package['payload_sha256']
            and absolute_deadline_unix == limits['absolute_deadline_unix'], 'manifest/deadline changed')
    require(run_id.startswith('phase3-logit-') and run_id == manifest['run_id'], 'separate logit run ID required')
    deadline = started+absolute_deadline_unix-time.time()
    startup = max(0., time.time()-allocation_start)
    directory = Path('/study')/run_id
    directory.mkdir(exist_ok=False)
    stats = dict(schema_version=1, run_id=run_id, source_run_id=SOURCE, explorer_run_id=EXPLORER,
                 exploratory=True, analysis_excluded=True, status='running', stage='setup',
                 model=MODEL, revision=MODELS[MODEL], gpu='L40S', dtype='bfloat16', attention='sdpa',
                 manifest=manifest, cells=list(CELLS), completed_forwards=0, planned_forwards=None,
                 positions='only last sequence token input to down_proj; generated response token',
                 sampling=False, autoregressive_generation=False,
                 previous_seconds=package['previous_seconds'], additional_cap_usd=PROBE_USD,
                 lifetime_cap_usd=7., rate_usd_per_gpu_second=RATE,
                 startup_seconds=startup, shutdown_reserve_seconds=SHUTDOWN_SECONDS,
                 timing_headroom=HEADROOM, exclusions=[], completed_contrasts=[], partial=True)
    rows, arrays, durations = [], {}, []

    def remaining():
        return max(0., limits['exposure_seconds']-startup-(time.monotonic()-started))

    def check(seconds=5.):
        if time.monotonic()+seconds >= deadline:
            raise TimeoutError('Absolute logit forward deadline')

    def checkpoint(stage):
        import numpy as np
        stats.update(stage=stage, function_seconds=time.monotonic()-started,
                     remaining_exposure_seconds=remaining(), completed_forwards=len(rows),
                     completed_contrasts=[dict(probe_index=r['probe_index'], cell=r['cell']) for r in rows])
        write_json(directory/'metrics.json', stats)
        temporary = directory/'logprobs.npz.tmp'
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(directory/'logprobs.npz')
        write_json(directory/'contrasts.json', rows)
        volume.commit()
        progress['state'] = {k: stats[k] for k in ('stage', 'completed_forwards', 'planned_forwards', 'function_seconds')}

    try:
        import importlib.metadata
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.modal_app import render_prompt

        check()
        for relative, expected in manifest['probe_source_sha256'].items():
            require(file_hash(Path(__file__).parent/Path(relative).name) == expected, 'shipped probe source changed')
        for name in ('src/phase3.py', 'src/modal_app_3.py', 'src/hf_decode.py', 'src/models.py', 'src/score.py'):
            require(file_hash(Path(__file__).parent/Path(name).name)
                    == package['original']['source_manifest']['source_sha256'][name], 'shipped frozen source changed')
        write_json(directory/'probe-manifest.json', manifest)
        stats['versions'] = {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'numpy', 'accelerate')}
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODELS[MODEL], use_fast=True)
        prefixes, evidence = [], []
        for i, (source, baseline) in enumerate(zip(package['original']['probes'], package['baselines'])):
            check()
            prefix = prepare_prefix(tokenizer, render_prompt(tokenizer, source), baseline['response'],
                                    baseline['rendered_prompt_sha256'])
            prefix.update(probe_index=i, source_request_id=source['id'],
                          explorer_baseline_id=baseline['id'], baseline_row_sha256=digest(baseline))
            evidence.append(prefix)
            if prefix['eligible']:
                prefixes.append(prefix)
            else:
                stats['exclusions'].append({k: prefix[k] for k in ('probe_index', 'source_request_id', 'exclusion')})
        write_json(directory/'prefixes.json', evidence)
        require(bool(prefixes), 'all three baseline boundaries excluded; no forwards planned')
        stats['planned_forwards'] = len(prefixes)*len(CELLS)
        check(60.)
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODELS[MODEL],
            torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa').eval()
        hf_cache.commit()
        layers = model.model.layers
        require((len(layers), model.config.intermediate_size) == (40, 17408)
                and model.dtype == torch.bfloat16 and 'L40S' in torch.cuda.get_device_name()
                and model.config._attn_implementation == 'sdpa', 'pinned GPU/model/dtype/attention changed')
        # Exercise the exact hook in CUDA BF16 without changing any model weights.
        x = torch.arange(30, device='cuda', dtype=torch.bfloat16).reshape(2, 3, 5)
        for mode in ('add', 'zero'):
            indices = torch.tensor([1, 3], device='cuda')
            values = torch.tensor([.0031, -3.0031], device='cuda', dtype=torch.float32)
            y = last_position_hook(indices, values, mode)(None, (x,))[0]
            expected = x.float().clone()
            expected[:, -1, indices] = 0 if mode == 'zero' else expected[:, -1, indices]+values
            require(torch.equal(y, expected.to(x.dtype)) and torch.equal(y[:, :-1], x[:, :-1]),
                    'runtime last-position BF16 hook check failed')
        del x, y, expected
        stats['hook_math_runtime_passed'] = True
        checkpoint('setup_complete')

        for prefix in prefixes:
            encoded = torch.tensor([prefix['input_ids']], device='cuda', dtype=torch.long)
            require(encoded.shape[1] <= model.config.max_position_embeddings, 'prefix exceeds model context')
            base = None
            for cell in CELLS:
                check(max(durations, default=5.)*HEADROOM)
                # The first baseline/push pair measures actual full-prefix costs.
                # Every later forward requires the ENTIRE remaining grid to fit.
                if len(durations) >= 2 and not grid_fits(durations,
                        stats['planned_forwards']-len(rows), remaining()):
                    raise TimeoutError('Measured full-grid projection exceeds remaining probe budget')
                spec, handles = condition(cell, package['original']['neurons']), []
                tick = time.monotonic()
                try:
                    for li in sorted({n['layer'] for n in spec['coordinates']}):
                        group = [n for n in spec['coordinates'] if n['layer'] == li]
                        indices = torch.tensor([n['neuron'] for n in group], device='cuda')
                        values = torch.tensor([n['amount'] for n in group], device='cuda', dtype=torch.float32)
                        handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(
                            last_position_hook(indices, values, spec['mode'])))
                    with torch.inference_mode():
                        output = model(input_ids=encoded, attention_mask=torch.ones_like(encoded),
                                       use_cache=False, logits_to_keep=1)
                        probs = log_probs(output.logits[0, -1])
                    del output
                    torch.cuda.synchronize()
                finally:
                    for handle in handles:
                        handle.remove()
                seconds = time.monotonic()-tick
                durations.append(seconds)
                if cell == 'baseline':
                    base = probs
                key = f'p{prefix["probe_index"]}__{cell}'
                arrays[key] = probs
                row = dict(run_id=run_id, probe_index=prefix['probe_index'],
                    source_request_id=prefix['source_request_id'], exploratory=True, analysis_excluded=True,
                    token_prefix_sha256=prefix['token_prefix_sha256'], npz_key=key, seconds=seconds,
                    **spec, **contrast_stats(base, probs, prefix['actual_next_token_id'],
                        lambda ids: tokenizer.decode(ids, skip_special_tokens=False,
                                                     clean_up_tokenization_spaces=False)))
                rows.append(row)
                # Persist every completed contrast, including a timing-gate refusal.
                checkpoint('forwards')
                check()
                if len(durations) == 2:
                    stats['initial_timing_gate'] = dict(baseline_seconds=durations[0], push_seconds=durations[1],
                        forwards_left=stats['planned_forwards']-2, headroom=HEADROOM,
                        remaining_exposure_seconds=remaining(),
                        fits=grid_fits(durations, stats['planned_forwards']-2, remaining()))
                    if not stats['initial_timing_gate']['fits']:
                        raise TimeoutError('Initial measured full-grid projection exceeds remaining probe budget')
            del encoded
        check()
        require(len(rows) == stats['planned_forwards'], 'incomplete planned grid')
        stats.update(status='completed', partial=False)
    except BaseException as error:
        stats.update(status='budget_stopped' if isinstance(error, TimeoutError) else 'failed',
                     stop_reason=f'{type(error).__name__}: {error}')
        import traceback
        traceback.print_exc()
    finally:
        checkpoint('finished')
        stats['accounting'] = dict(function_seconds=time.monotonic()-started,
            allocation_seconds_at_return=startup+time.monotonic()-started,
            final='same local lifetime ledger through verified zero authoritative')
        stats['artifacts_sha256'] = {p.name: file_hash(p) for p in directory.iterdir()
                                    if p.is_file() and not p.name.endswith('.tmp') and p.name != 'probe-result.json'}
        write_json(directory/'probe-result.json', stats)
        volume.commit()
    return dict(status=stats['status'], function_seconds=time.monotonic()-started, partial=stats['partial'])


def container_snapshot():
    import subprocess
    rows = json.loads(subprocess.check_output(['uv', 'run', 'modal', 'container', 'list', '--json'], timeout=5))
    return dict(checked_unix=time.time(), containers=rows)


@app.local_entrypoint()
def main(run_id: str, source_run_id: str = SOURCE, explorer_run_id: str = EXPLORER):
    import fcntl
    from src.build_prompts import ROOT
    out = ROOT/'results/phase3'
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = out/'gpu-attempts.json'
        attempts = json.loads(ledger.read_text())
        package = load_package(ROOT, source_run_id, explorer_run_id, run_id, attempts)
        previous = lifetime_seconds(attempts)
        before = container_snapshot()
        validate_no_containers(before['containers'])
        require_remote_fresh(run_id)
        original = package['original']
        manifest = dict(schema_version=1, run_id=run_id, source_run_id=SOURCE, explorer_run_id=EXPLORER,
            exploratory=True, analysis_excluded=True, payload_sha256=package['payload_sha256'],
            ledger_sha256=package['ledger_sha256'], original=original,
            baselines=package['baselines'], explorer_artifacts_sha256=package['explorer_artifacts_sha256'],
            probe_source_sha256={p: file_hash(ROOT/p) for p in PATHS}, cells=list(CELLS),
            additional_cap_usd=PROBE_USD, lifetime_cap_usd=7., rate_usd_per_gpu_second=RATE)
        verify_hashes(ROOT, original['source_manifest']['source_sha256'])
        verify_hashes(ROOT, original['source_artifacts_sha256'])
        verify_hashes(ROOT, package['explorer_artifacts_sha256'])
        rawdir = ROOT/'results/raw'/run_id
        rawdir.mkdir(parents=True)
        write_json(rawdir/'launch-manifest.json', manifest)
        write_json(rawdir/'containers-before.json', before)
        call, submitted, zero, caught = None, False, False, None
        with modal.Dict.ephemeral() as progress:
            tick, allocation_start = time.monotonic(), time.time()
            limits = probe_limits(previous, allocation_start)
            attempt = dict(run_id=run_id, source_run_id=SOURCE, explorer_run_id=EXPLORER,
                exploratory=True, analysis_excluded=True, status='running', previous_seconds=previous,
                gpu='L40S', rate_usd_per_hour=RATE*3600, started_unix=allocation_start, **limits)
            attempts.append(attempt)
            write_json(ledger, attempts)
            try:
                submitted = True
                call = probe.with_options(timeout=limits['timeout_seconds'], retries=0).spawn(
                    package, manifest, run_id, progress, allocation_start, limits['absolute_deadline_unix'])
                attempt['call_id'] = call.object_id
                write_json(ledger, attempts)
                seen = None
                while True:
                    if time.monotonic()-tick >= limits['exposure_seconds']-SHUTDOWN_SECONDS:
                        raise TimeoutError('Local logit/lifetime cancellation guard')
                    state = progress.get('state')
                    if state and state != seen:
                        write_json(rawdir/'local-progress.json', state); seen = state
                        print('LOCAL_LOGIT '+json.dumps(state), flush=True)
                    try:
                        result = call.get(timeout=0)
                    except TimeoutError:
                        time.sleep(2)
                        continue
                    attempt.update(status=result['status'], function_seconds=result['function_seconds'], partial=result['partial'])
                    break
            except BaseException as error:
                caught = error
                attempt.update(status='failed_or_cancelled' if call is not None else 'submission_unresolved',
                               error=f'{type(error).__name__}: {error}')
            finally:
                if call is not None:
                    try:
                        call.cancel(terminate_containers=True)
                    except Exception as error:
                        attempt['cancel_error'] = f'{type(error).__name__}: {error}'
                for _ in range(15):
                    try:
                        evidence = container_snapshot()
                        write_json(rawdir/'containers-after.json', evidence)
                        validate_no_containers(evidence['containers'])
                        zero = True
                        attempt['zero_containers_verified_unix'] = evidence['checked_unix']
                        break
                    except Exception as error:
                        attempt['shutdown_check_error'] = f'{type(error).__name__}: {error}'
                        if time.monotonic()-tick+7 >= limits['exposure_seconds']:
                            break
                        time.sleep(2)
                elapsed = max(time.monotonic()-tick, time.time()-allocation_start)
                attempt.update(close_allocation(call_known=call is not None, elapsed=elapsed,
                    previous_seconds=previous, status=attempt['status'], zero_containers=zero), ended_unix=time.time())
                write_json(ledger, attempts)
                recovery_errors = recover_artifacts(rawdir, run_id) if submitted else []
                write_json(rawdir/'local-execution.json', dict(attempt=attempt,
                    lifetime_gpu_usd_upper=sum(a['wall_seconds'] for a in attempts)*RATE,
                    probe_gpu_usd_upper=attempt['wall_seconds']*RATE, recovery_errors=recovery_errors))
        print(json.dumps(dict(run_id=run_id, status=attempt['status'], raw_directory=str(rawdir),
                              exploratory=True, analysis_excluded=True)), flush=True)
        if caught is not None:
            raise caught
        require(zero and not recovery_errors and attempt['status'] == 'completed',
                'probe incomplete or requires shutdown/artifact reconciliation')
