"""Explicitly launched, single-use exploratory allocation. No registered outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import modal

from src.modal_app_3 import image, hf_cache, volume, write_json
from src.modal_app_3_resume import require_remote_fresh, recover_artifacts
from src.models import MODELS
from src.phase3 import MODEL, RATE, CAP_SECONDS, score_generation
from src.phase3_resume import validate_no_containers
from src.phase3_explore import (CELLS, EXPLORATION_USD, STARTUP_SECONDS, SHUTDOWN_SECONDS,
    MIN_CELL_SECONDS, EXPLORER_PATHS, SAMPLING, cells_fit, close_allocation, condition,
    digest, exploration_limits, file_hash, input_hook, lifetime_seconds, load_package,
    make_generation_row, require, summarize, validate_package, verify_hashes)

app = modal.App('grader-mode-phase3-explore')


@app.function(image=image, gpu='L40S', volumes={'/hf': hf_cache, '/study': volume},
              timeout=int(EXPLORATION_USD/RATE-STARTUP_SECONDS-SHUTDOWN_SECONDS),
              startup_timeout=STARTUP_SECONDS, retries=0, single_use_containers=True,
              min_containers=0, buffer_containers=0, max_containers=1, scaledown_window=2,
              cpu=4, memory=32768)
def explore(package: dict, manifest: dict, run_id: str, progress: modal.Dict,
            allocation_start: float, absolute_deadline_unix: float):
    started = time.monotonic()
    validate_package(package, rebuild_requests=False)
    require(manifest['payload_sha256'] == package['payload_sha256'], 'manifest/payload mismatch')
    limits = exploration_limits(package['previous_seconds'], allocation_start)
    require(absolute_deadline_unix == limits['absolute_deadline_unix'], 'allocation deadline changed')
    require(run_id.startswith('phase3-explore-') and run_id != package['source_run_id'], 'separate exploration ID required')
    startup = max(0., time.time()-allocation_start)
    deadline = started+absolute_deadline_unix-time.time()
    directory = Path('/study')/run_id
    directory.mkdir(exist_ok=False)
    rows, durations = [], []
    stats = dict(schema_version=1, run_id=run_id, source_run_id=package['source_run_id'],
                 exploratory=True, analysis_excluded=True, status='running', stage='setup',
                 model=MODEL, revision=MODELS[MODEL], gpu='L40S', sampling=SAMPLING,
                 manifest=manifest, batches=[], completed_cells=[], output_rows=0,
                 previous_seconds=package['previous_seconds'], startup_seconds=startup,
                 seed_method='reset torch.manual_seed(first original Phase 1b input seed) each cell',
                 push_positions='cached generated input positions only; prefill unchanged',
                 additional_cap_usd=EXPLORATION_USD, lifetime_cap_usd=7.,
                 rate_usd_per_gpu_second=RATE, shutdown_reserve_seconds=SHUTDOWN_SECONDS)
    raw = (directory/'generations.jsonl').open('x')

    def remaining():
        return max(0., limits['exposure_seconds']-startup-(time.monotonic()-started))

    def check(seconds=5.):
        if time.monotonic()+seconds >= deadline:
            raise TimeoutError('Exploration absolute generation deadline')

    def checkpoint(stage):
        stats.update(stage=stage, output_rows=len(rows), function_seconds=time.monotonic()-started,
                     allocation_seconds=startup+time.monotonic()-started,
                     remaining_exposure_seconds=remaining(), **summarize(rows, completed_cells=stats['completed_cells']))
        write_json(directory/'metrics.json', stats)
        write_json(directory/'summary.json', summarize(rows, completed_cells=stats['completed_cells']))
        volume.commit()
        progress['state'] = {k: stats[k] for k in ('stage', 'completed_cells', 'output_rows', 'function_seconds')}

    try:
        import gc
        import importlib.metadata
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.hf_decode import compact_generate
        from src.modal_app import render_prompt

        srcdir = Path(__file__).parent
        for relative, expected in manifest['explorer_sha256'].items():
            require(file_hash(srcdir/Path(relative).name) == expected, 'shipped exploration source changed')
        for name in ('src/phase3.py', 'src/modal_app_3.py', 'src/hf_decode.py', 'src/models.py', 'src/score.py'):
            require(file_hash(srcdir/Path(name).name) == package['source_manifest']['source_sha256'][name],
                    f'shipped original source changed: {name}')
        write_json(directory/'exploration-manifest.json', manifest)
        write_json(directory/'source-selection.json', package['source_selection'])
        stats['versions'] = {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'numpy', 'accelerate')}
        check()
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODELS[MODEL], padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        probes = package['probes']
        rendered = [render_prompt(tokenizer, r) for r in probes]
        hashes = [hashlib.sha256(p.encode()).hexdigest() for p in rendered]
        require(hashes == [package['prompt_hashes'][r['id']] for r in probes], 'original rendered prompt changed')
        check()
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODELS[MODEL],
            torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa').eval()
        hf_cache.commit()
        layers = model.model.layers
        require((len(layers), model.config.intermediate_size) == (40, 17408)
                and model.dtype == torch.bfloat16 and 'L40S' in torch.cuda.get_device_name(), 'GPU/model shape changed')
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos)
        require(len(eos) == 2 and set(eos) == {151643, 151645}, 'both pinned EOS IDs required')
        stats['eos_token_ids'] = eos

        def encode():
            return tokenizer(rendered, padding=True, add_special_tokens=False, return_tensors='pt').to('cuda')

        # One cheap deterministic cache-compaction parity check before sampling.
        check()
        encoded = encode()
        with torch.inference_mode():
            reference = model.generate(**encoded, do_sample=False, max_new_tokens=8,
                                       logits_to_keep=1, pad_token_id=tokenizer.pad_token_id)
            actual, meta = compact_generate(model, encoded, eos_token_id=eos, max_new_tokens=8,
                                            deadline=deadline, sample=False)
        require(not meta['budget_terminated'] and torch.equal(reference[:, encoded.input_ids.shape[1]:], actual),
                'greedy compact decoder parity failed')
        del encoded, reference, actual
        # Exercise the ACTUAL shared add wrapper and zero hook in CUDA BF16.
        for mode in ('add', 'zero'):
            state = dict(generated=False)
            linear = torch.nn.Linear(5, 3, bias=False, device='cuda', dtype=torch.bfloat16)
            x = torch.arange(20, device='cuda', dtype=torch.bfloat16).reshape(2, 2, 5)
            weights = linear.weight.detach().clone()
            indices = torch.tensor([1, 3], device='cuda')
            values = torch.tensor([.0031, -3.0031], device='cuda', dtype=torch.float32)
            handle = linear.register_forward_pre_hook(input_hook(indices, values, state, mode))
            try:
                require(torch.equal(linear(x), torch.nn.functional.linear(x, weights)), 'hook modified prefill')
                state['generated'] = True
                expected = x.float()
                expected[..., indices] = 0 if mode == 'zero' else expected[..., indices]+values
                require(torch.equal(linear(x), torch.nn.functional.linear(expected.to(x.dtype), weights))
                        and torch.equal(linear.weight, weights), 'runtime hook math/weights failed')
            finally:
                handle.remove()
        del linear, x, weights, indices, values, expected
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        stats.update(greedy_cache_parity_passed=True, hook_math_runtime_passed=True)
        checkpoint('setup_complete')

        for cell_index, cell in enumerate(CELLS):
            # After baseline, require TWO additional full cells to fit initially.
            needed = 2 if cell_index == 1 else 1
            estimates = durations or [MIN_CELL_SECONDS]
            if not cells_fit(estimates, remaining_seconds=remaining(), cells=needed):
                stats['stop_reason'] = 'measured_cell_projection_exceeds_remaining_exploration_budget'
                break
            check()
            seed = probes[0]['seed']
            torch.manual_seed(seed)
            state, handles = dict(generated=False), []
            spec = condition(cell, package['neurons'])
            for li in sorted({n['layer'] for n in spec['coordinates']}):
                group = [n for n in spec['coordinates'] if n['layer'] == li]
                indices = torch.tensor([n['neuron'] for n in group], device='cuda')
                values = torch.tensor([n['amount'] for n in group], device='cuda', dtype=torch.float32)
                handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(
                    input_hook(indices, values, state, spec['mode'])))

            def forward(**kwargs):
                state['generated'] = kwargs.get('past_key_values') is not None
                return model(**kwargs)

            def decode_progress(step, active):
                progress['state'] = dict(stage='generation', cell=cell, decode_step=step,
                                         active_rows=active, output_rows=len(rows))

            tick = time.monotonic()
            try:
                encoded = encode()
                with torch.inference_mode():
                    generated, meta = compact_generate(forward, encoded, eos_token_id=eos,
                        max_new_tokens=3072, deadline=deadline, progress=decode_progress)
                torch.cuda.synchronize()
            finally:
                for handle in handles:
                    handle.remove()
            seconds = time.monotonic()-tick
            incomplete = meta['budget_terminated']
            cell_rows = []
            for source, tokens, prompt_hash in zip(probes, generated.cpu().tolist(), hashes):
                end = next((i for i, token in enumerate(tokens) if token in eos), len(tokens))
                response = tokenizer.decode(tokens[:end], skip_special_tokens=True)
                end_id = tokens[end] if end < len(tokens) else None
                final = score_generation(response, source['gold'], 'stop' if end_id is not None else 'length')['final_reply']
                cell_rows.append(make_generation_row(source, cell=cell, neurons=package['neurons'],
                    source_run_id=package['source_run_id'], run_id=run_id, response=response,
                    output_tokens=end, end_token_id=end_id, batch_seed=seed,
                    rendered_prompt_sha256=prompt_hash, final_reply_tokens=len(tokenizer.encode(final, add_special_tokens=False)),
                    incomplete=incomplete))
            raw.write(''.join(json.dumps(r, allow_nan=False)+'\n' for r in cell_rows)); raw.flush()
            rows.extend(cell_rows)
            stats['batches'].append(dict(cell=cell, n=3, seed=seed, seconds=seconds,
                source_ids=[r['id'] for r in probes], **meta))
            if not incomplete:
                durations.append(seconds)
                stats['completed_cells'].append(cell)
            checkpoint('generation')
            del encoded, generated
            if incomplete:
                raise TimeoutError('Deadline-truncated batch retained only as incomplete exploratory rows')
        # Completed means bounded allocation succeeded; partial flag is separate.
        stats['status'] = 'completed'
    except BaseException as error:
        stats.update(status='budget_stopped' if isinstance(error, TimeoutError) else 'failed',
                     stop_reason=f'{type(error).__name__}: {error}')
        import traceback
        traceback.print_exc()
    finally:
        raw.close()
        checkpoint('finished')
        stats['accounting'] = dict(function_seconds=time.monotonic()-started,
            allocation_seconds_at_return=startup+time.monotonic()-started,
            previous_seconds=package['previous_seconds'], final='local lifetime ledger through verified zero authoritative')
        stats['artifacts_sha256'] = {p.name: file_hash(p) for p in directory.iterdir()
                                    if p.is_file() and not p.name.endswith('.tmp') and p.name != 'exploration-result.json'}
        write_json(directory/'exploration-result.json', stats)
        volume.commit()
    return dict(status=stats['status'], function_seconds=time.monotonic()-started,
                partial_exploration=stats['partial_exploration'])


def container_snapshot():
    """Bound each CLI check so 15 checks + 2s pauses fit the shutdown reserve."""
    import subprocess
    rows = json.loads(subprocess.check_output(['uv', 'run', 'modal', 'container', 'list', '--json'], timeout=5))
    return dict(checked_unix=time.time(), containers=rows)


@app.local_entrypoint()
def main(source_run_id: str, run_id: str):
    """Explicit parent-approved launch; the CPU module never calls this."""
    import fcntl
    from src.build_prompts import ROOT
    out = ROOT/'results/phase3'
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = out/'gpu-attempts.json'
        attempts = json.loads(ledger.read_text())
        package = load_package(ROOT, source_run_id, run_id, attempts)
        previous = lifetime_seconds(attempts)
        before = container_snapshot()
        validate_no_containers(before['containers'])
        require_remote_fresh(run_id)
        rawdir = ROOT/'results/raw'/run_id
        manifest = dict(schema_version=1, run_id=run_id, source_run_id=source_run_id,
            exploratory=True, analysis_excluded=True, source_manifest=package['source_manifest'],
            source_selection=package['source_selection'], source_artifacts_sha256=package['source_artifacts_sha256'],
            payload_sha256=package['payload_sha256'], ledger_sha256=package['ledger_sha256'],
            explorer_sha256={p: file_hash(ROOT/p) for p in EXPLORER_PATHS}, cells=list(CELLS),
            probes=package['probes'], prompt_hashes=package['prompt_hashes'], sampling=SAMPLING,
            additional_cap_usd=EXPLORATION_USD, lifetime_cap_usd=7., rate_usd_per_gpu_second=RATE)
        verify_hashes(ROOT, package['source_manifest']['source_sha256'])
        verify_hashes(ROOT, package['source_artifacts_sha256'])
        rawdir.mkdir(parents=True)
        write_json(rawdir/'launch-manifest.json', manifest)
        write_json(rawdir/'containers-before.json', before)
        call, submitted, zero, caught = None, False, False, None
        recovery_errors = []
        with modal.Dict.ephemeral() as progress:
            tick, allocation_start = time.monotonic(), time.time()
            limits = exploration_limits(previous, allocation_start)
            attempt = dict(run_id=run_id, source_run_id=source_run_id, exploratory=True,
                analysis_excluded=True, status='running', previous_seconds=previous,
                gpu='L40S', rate_usd_per_hour=RATE*3600, started_unix=allocation_start, **limits)
            attempts.append(attempt)
            write_json(ledger, attempts)
            try:
                submitted = True
                call = explore.with_options(timeout=limits['timeout_seconds'], retries=0).spawn(
                    package, manifest, run_id, progress, allocation_start, limits['absolute_deadline_unix'])
                attempt['call_id'] = call.object_id
                write_json(ledger, attempts)
                seen = None
                while True:
                    if time.monotonic()-tick >= limits['exposure_seconds']-SHUTDOWN_SECONDS:
                        raise TimeoutError('Local exploration/lifetime cancellation guard')
                    state = progress.get('state')
                    if state and state != seen:
                        write_json(rawdir/'local-progress.json', state); seen = state
                        print('LOCAL_EXPLORE '+json.dumps(state), flush=True)
                    try:
                        result = call.get(timeout=0)
                    except TimeoutError:
                        time.sleep(2)
                        continue
                    attempt.update(status=result['status'], function_seconds=result['function_seconds'],
                                   partial_exploration=result['partial_exploration'])
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
                # Also snapshot a lost/uncertain spawn; a known handle remains
                # mandatory for reconciliation even if this snapshot shows zero.
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
                    exploration_gpu_usd_upper=attempt['wall_seconds']*RATE, recovery_errors=recovery_errors))
        print(json.dumps(dict(run_id=run_id, status=attempt['status'], raw_directory=str(rawdir),
                              exploratory=True, analysis_excluded=True)), flush=True)
        if caught is not None:
            raise caught
        require(zero and not recovery_errors, 'exploration requires shutdown/artifact reconciliation')
