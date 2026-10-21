"""Optional Phase 3 continuation, never imported/executed by the active runner.

See src.phase3_resume's module docstring for the required parent integration.
No merge, analysis, dose reselection, or original-output mutation occurs here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import modal

from src.modal_app_3 import image, hf_cache, volume, push_hook, write_json
from src.models import MODELS
from src.phase3 import MODEL, RATE, CAP_SECONDS, CELLS
from src.phase3_resume import (SAMPLING, TIMING_METHOD, WORK_RESERVE, allocation_limits,
    close_allocation, digest, exact_batch_capacity, file_hash, ledger_seconds,
    load_source_package, make_generation_row, make_test_batches, project_test_plan,
    require, safe_run_id, validate_no_containers, validate_payload, verify_hashes)

app = modal.App('grader-mode-phase3-resume')


@app.function(image=image, gpu='L40S', volumes={'/hf': hf_cache, '/study': volume},
              timeout=12600, startup_timeout=120, retries=0, single_use_containers=True,
              min_containers=0, buffer_containers=0, max_containers=1, scaledown_window=2,
              cpu=4, memory=32768)
def resume_study(package: dict, resume_manifest: dict, allocation_run_id: str,
                 progress: modal.Dict, previous_seconds: float,
                 allocation_started_unix: float, absolute_deadline_unix: float):
    """One metered model load, one dev-only timing batch, then the fixed test grid."""
    started = time.monotonic()
    # Modal ships src, not the prompts directory; local preflight rebuilds requests.
    # Remote verifies the immutable transport digest and all rendered prompt hashes.
    validate_payload(package, rebuild_requests=False)
    require(resume_manifest['payload_sha256'] == package['payload_sha256'], 'manifest/payload mismatch')
    require(previous_seconds == package['eligibility']['previous_seconds'], 'lifetime spent changed')
    limits = allocation_limits(previous_seconds, allocation_started_unix)
    require(absolute_deadline_unix == limits['absolute_deadline_unix'], 'absolute deadline exceeds remaining exposure')
    safe_run_id(allocation_run_id)
    require(allocation_run_id != package['source_run_id'], 'allocation ID must be fresh')
    deadline = started+absolute_deadline_unix-time.time()
    require(deadline > started and allocation_started_unix <= time.time(), 'invalid absolute allocation clock')
    # Convert startup/provisioning into a fixed elapsed offset; monotonic thereafter.
    startup_elapsed = max(0., time.time()-allocation_started_unix)
    directory = Path('/study')/allocation_run_id
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    stats = dict(schema_version=1, run_id=package['source_run_id'], allocation_run_id=allocation_run_id,
                 status='running', stage='setup', model=MODEL, revision=MODELS[MODEL], gpu='L40S',
                 source_manifest=package['source_manifest'], source_selection=package['source_selection'],
                 resume_manifest=resume_manifest, sampling=SAMPLING.copy(), batches=[], output_rows=0,
                 planning_pilot=None, test_generation_started=False,
                 parent_validation_required=True, timing_method=TIMING_METHOD,
                 previous_seconds=previous_seconds, startup_elapsed_seconds=startup_elapsed,
                 rate_usd_per_gpu_second=RATE, cap_usd=7.,
                 seed_method='torch.manual_seed(first Phase 1b request seed per ordered batch); CUDA global stream; EOS cache compaction',
                 push_positions='cached generated inputs only, including final reply; unmodified prefill',
                 work_reserve_bytes=WORK_RESERVE, capacity_rounding='exact positive integer, capped at 64',
                 oom_policy='fail closed, no split/rebatch after timing pilot',
                 allocation_started_unix=allocation_started_unix,
                 absolute_deadline_unix=absolute_deadline_unix)
    raw = (directory/'generations.jsonl').open('x')
    pilot_file = (directory/'planning-pilot.jsonl').open('x')

    def spent():
        return previous_seconds+startup_elapsed+time.monotonic()-started

    def check():
        if time.monotonic()+60 >= deadline:
            raise TimeoutError('Resume absolute GPU budget deadline')

    def checkpoint(stage, **extra):
        stats.update(extra, stage=stage, function_seconds=time.monotonic()-started,
                     lifetime_gpu_seconds_at_checkpoint=spent(), estimated_gpu_usd=spent()*RATE)
        write_json(directory/'metrics.json', stats)
        volume.commit()
        state = dict(stage=stage, function_seconds=stats['function_seconds'],
                     output_rows=stats['output_rows'], batches=len(stats['batches']))
        progress['state'] = state
        print('PHASE3_RESUME '+json.dumps(state), flush=True)

    try:
        import gc
        import importlib.metadata
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.hf_decode import compact_generate
        from src.modal_app import render_prompt
        from src.phase3 import score_generation

        stats['versions'] = {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'numpy', 'accelerate')}
        stats['versions']['modal'] = modal.__version__
        # Bind the shipped continuation and the shared decoder/hook sources too.
        srcdir = Path(__file__).parent
        for relative, expected in resume_manifest['continuation_sha256'].items():
            require(file_hash(srcdir/Path(relative).name) == expected, 'shipped continuation source changed')
        for name in ('src/phase3.py', 'src/modal_app_3.py', 'src/hf_decode.py', 'src/models.py', 'src/score.py'):
            require(file_hash(srcdir/Path(name).name) == package['source_manifest']['source_sha256'][name],
                    f'shipped original source changed: {name}')
        write_json(directory/'resume-manifest.json', resume_manifest)
        write_json(directory/'source-manifest.json', package['source_manifest'])
        write_json(directory/'source-selection.json', package['source_selection'])
        write_json(directory/'neurons.json', package['neurons'])
        check()
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODELS[MODEL], padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        requests = package['dev_requests']+package['test_requests']
        rendered = {r['id']: render_prompt(tokenizer, r) for r in requests}
        hashes = {rid: hashlib.sha256(text.encode()).hexdigest() for rid, text in rendered.items()}
        require(all(hashes[r['id']] == package['prompt_hashes'][r['id']] for r in requests),
                'rendered prompt differs from original development/tracking')
        stats['original_rendered_prompt_hashes_verified'] = len(requests)
        check()
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODELS[MODEL],
            torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa').eval()
        hf_cache.commit()
        layers = model.model.layers
        require((len(layers), model.config.intermediate_size) == (40, 17408), 'unexpected neuron dimensions')
        require(model.dtype == torch.bfloat16 and 'L40S' in torch.cuda.get_device_name(), 'unexpected GPU/dtype')
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos)
        require(set(eos) == {151643, 151645} and len(eos) == 2, 'both pinned EOS IDs required')
        stats['eos_token_ids'] = eos
        stats['cuda_device'] = torch.cuda.get_device_name()

        def encode(rows):
            return tokenizer([rendered[r['id']] for r in rows], padding=True,
                             add_special_tokens=False, return_tensors='pt').to('cuda')

        # Cheap exact decoder parity; the timing pilot resets the Phase 1b seed.
        check()
        encoded = encode(package['dev_requests'][:2])
        with torch.inference_mode():
            reference = model.generate(**encoded, do_sample=False, max_new_tokens=8,
                                       logits_to_keep=1, pad_token_id=tokenizer.pad_token_id)
            actual, meta = compact_generate(model, encoded, eos_token_id=eos, max_new_tokens=8,
                                            deadline=deadline, sample=False)
        require(not meta['budget_terminated'] and torch.equal(reference[:, encoded.input_ids.shape[1]:], actual),
                'greedy compact decoder parity failed')
        del encoded, reference, actual
        # Exercise the actual shared hook with float32 additions restored to BF16.
        state = {'generated': False}
        linear = torch.nn.Linear(5, 3, bias=False, device='cuda', dtype=torch.bfloat16)
        x = torch.zeros(2, 2, 5, device='cuda', dtype=torch.bfloat16)
        indices = torch.tensor([1, 3], device='cuda')
        values = torch.tensor([.0031, -3.0031], device='cuda', dtype=torch.float32)
        handle = linear.register_forward_pre_hook(push_hook(indices, values, state))
        try:
            require(torch.equal(linear(x), linear.weight.new_zeros(2, 2, 3)), 'hook modified prefill')
            state['generated'] = True
            expected = x.float(); expected[..., indices] += values
            require(torch.equal(linear(x), torch.nn.functional.linear(expected.to(x.dtype), linear.weight)),
                    'float32 delta to BF16 hook math failed')
        finally:
            handle.remove()
        del linear, x, expected, indices, values
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        stats.update(greedy_cache_parity_passed=True, hook_math_runtime_passed=True)
        kv = (2*len(layers)*model.config.num_key_value_heads
              *getattr(model.config, 'head_dim', model.config.hidden_size//model.config.num_attention_heads)*2)
        max_prompt = max(len(tokenizer.encode(t, add_special_tokens=False)) for t in rendered.values())
        free, total = torch.cuda.mem_get_info()
        batch_size = exact_batch_capacity(free, kv, max_prompt)
        stats.update(batch_size=batch_size, kv_bytes_per_token=kv, max_prompt_tokens=max_prompt,
                     capacity_free_bytes=free, capacity_total_bytes=total,
                     setup_seconds=time.monotonic()-started)
        checkpoint('planning_pilot')
        neurons = package['neurons']

        def generate_batch(batch, cell, c, *, pilot=False):
            check()
            seed = batch[0]['seed']
            torch.manual_seed(seed)
            state = {'generated': False}
            handles = []
            if cell != 'baseline':
                control = int(cell.split('_')[-1]) if cell.startswith('control_') else None
                group = neurons['controls'][control] if control is not None else neurons['tutor']
                amounts = [c*abs(r['delta'])*neurons['control_signs'][control][i] if control is not None
                           else c*r['delta'] for i, r in enumerate(neurons['tutor'])]
                for li in sorted({r['layer'] for r in group}):
                    indices = torch.tensor([r['neuron'] for r in group if r['layer'] == li], device='cuda')
                    values = torch.tensor([v for r, v in zip(group, amounts) if r['layer'] == li],
                                          device='cuda', dtype=torch.float32)
                    handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(push_hook(indices, values, state)))

            def forward(**kwargs):
                state['generated'] = kwargs.get('past_key_values') is not None
                return model(**kwargs)

            def decode_progress(step, active):
                progress['state'] = dict(stage='planning_pilot' if pilot else 'test', cell=cell,
                    decode_step=step, active_rows=active, output_rows=stats['output_rows'],
                    function_seconds=time.monotonic()-started)

            torch.cuda.synchronize()
            tick = time.monotonic()
            try:
                encoded = encode(batch)
                with torch.inference_mode():
                    generated, decode_meta = compact_generate(forward, encoded, eos_token_id=eos,
                        max_new_tokens=3072, deadline=deadline, progress=decode_progress)
                torch.cuda.synchronize()
            except BaseException as error:
                stats.setdefault('incomplete_batches', []).append(dict(cell=cell, c=c, pilot=pilot,
                    split=batch[0]['split'], n=len(batch), seed=seed, first_id=batch[0]['id'],
                    last_id=batch[-1]['id'], seconds=time.monotonic()-tick,
                    error=f'{type(error).__name__}: {error}'))
                raise
            finally:
                for handle in handles:
                    handle.remove()
            seconds = time.monotonic()-tick
            record = dict(cell=cell, c=c, split=batch[0]['split'], n=len(batch), seed=seed,
                          first_id=batch[0]['id'], last_id=batch[-1]['id'], seconds=seconds,
                          prompt_width=encoded.input_ids.shape[1], **decode_meta)
            if decode_meta['budget_terminated']:
                stats.setdefault('incomplete_batches', []).append(record)
                raise TimeoutError('Absolute deadline: incomplete batch excluded from study rows')
            rows = []
            for r, ids in zip(batch, generated.cpu().tolist()):
                end = next((i for i, token in enumerate(ids) if token in eos), len(ids))
                text = tokenizer.decode(ids[:end], skip_special_tokens=True)
                final = score_generation(text, r['gold'], 'stop' if end < len(ids) else 'length')['final_reply']
                row = make_generation_row(r, cell=cell, c=c, source_run_id=package['source_run_id'],
                    allocation_run_id=allocation_run_id, response=text, output_tokens=end,
                    end_token_id=ids[end] if end < len(ids) else None, batch_seed=seed,
                    rendered_prompt_sha256=hashes[r['id']],
                    final_reply_tokens=len(tokenizer.encode(final, add_special_tokens=False)))
                if pilot:
                    row.update(planning_only=True, analysis_excluded=True, pilot_kind=TIMING_METHOD)
                rows.append(row)
            record.update(output_tokens=sum(r['output_tokens'] for r in rows),
                          max_output_tokens=max(r['output_tokens'] for r in rows),
                          final_reply_tokens=sum(r['final_reply_tokens'] for r in rows))
            target = pilot_file if pilot else raw
            target.write(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows)); target.flush()
            del encoded, generated
            return record

        # Exactly ONE full-capacity baseline batch, using DEVELOPMENT template 0.
        # Its outputs cannot influence original dose choice or enter test analysis.
        pilot = generate_batch(package['dev_requests'][:batch_size], 'baseline', 0., pilot=True)
        stats['planning_pilot'] = pilot
        checkpoint('pilot_completed')
        check()
        try:
            plan = project_test_plan(pilot['seconds'], batch_size, spent())
        except ValueError:
            stats['planning_refusal'] = True
            raise
        selection = {**package['source_selection'], **plan, 'test_generation_started': False,
                     'c_star': package['source_selection']['c_star'],
                     'pilot_rows_excluded_from_study': pilot['n']}
        stats['selection'] = selection
        write_json(directory/'selection.json', selection)
        checkpoint('pretest_planned')
        # Commit overhead is charged too. Never begin if the persisted plan no
        # longer fits; do not change templates after seeing any test generation.
        require(spent()+plan['projected_test_seconds']+300 <= CAP_SECONDS,
                'Projected test grid no longer fits after plan checkpoint')
        batches = make_test_batches(package['test_requests'], plan['test_templates'], batch_size)
        stats['test_batch_source_ids'] = [[r['id'] for r in batch] for batch in batches]
        selection['test_generation_started'] = True
        stats['test_generation_started'] = True
        write_json(directory/'selection.json', selection)
        checkpoint('test')
        require(spent()+plan['projected_test_seconds']+300 <= CAP_SECONDS,
                'Projected test grid no longer fits before first test batch')
        for cell in CELLS:
            for batch_index, batch in enumerate(batches):
                record = generate_batch(batch, cell, 0. if cell == 'baseline' else selection['c_star'])
                record['batch_index'] = batch_index
                stats['batches'].append(record)
                stats['output_rows'] += len(batch)
                checkpoint('test')
        require(stats['output_rows'] == 100*plan['test_templates']*7, 'incomplete test grid')
        stats['status'] = 'completed'
    except BaseException as error:
        stats.update(status='budget_stopped' if isinstance(error, TimeoutError) or stats.get('planning_refusal') else 'failed',
                     stop_reason=f'{type(error).__name__}: {error}')
        import traceback
        traceback.print_exc()
    finally:
        raw.close(); pilot_file.close()
        checkpoint('finished')
        stats['artifacts_sha256'] = {p.name: file_hash(p) for p in directory.iterdir()
                                    if p.is_file() and not p.name.endswith('.tmp') and p.name != 'resume-result.json'}
        stats['accounting'] = dict(previous_seconds=previous_seconds, startup_elapsed_seconds=startup_elapsed,
                                  function_seconds=time.monotonic()-started,
                                  lifetime_gpu_seconds_at_return=spent(), rate_usd_per_gpu_second=RATE,
                                  cap_usd=7., final_allocation_wall_time='local lifetime ledger authoritative')
        write_json(directory/'resume-result.json', stats)
        volume.commit()
    return dict(status=stats['status'], function_seconds=time.monotonic()-started,
                artifacts=[f'{allocation_run_id}/{p.name}' for p in directory.iterdir() if p.is_file()])


def container_evidence():
    """Called only by a later authorized local entry point, never on import."""
    import subprocess
    result = json.loads(subprocess.check_output(['uv', 'run', 'modal', 'container', 'list', '--json'], timeout=30))
    validate_no_containers(result)
    return dict(checked_unix=time.time(), containers=result)


def require_remote_fresh(run_id):
    # Listing a nonexistent path can raise an RPC error; listing the root also
    # detects an empty existing directory, which must never be reused.
    entries = volume.listdir('/')
    require(not any(str(entry.path).strip('/').split('/')[0] == run_id for entry in entries),
            'remote allocation path already exists')


def recover_artifacts(rawdir, run_id):
    """Recover partial continuation outputs only; never target results/phase3."""
    errors = []
    try:
        entries = list(volume.listdir(run_id))
    except Exception as error:
        return [f'list: {type(error).__name__}: {error}']
    for entry in entries:
        name = Path(entry.path).name
        if name.endswith('.tmp'):
            continue
        try:
            target = rawdir/name
            require(not target.exists(), f'recovery target already exists: {name}')
            partial = rawdir/(name+'.download-part')
            with partial.open('xb') as handle:
                for chunk in volume.read_file(entry.path):
                    handle.write(chunk)
            partial.replace(target)
        except Exception as error:
            errors.append(f'{name}: {type(error).__name__}: {error}')
    return errors


@app.local_entrypoint()
def main(source_run_id: str, run_id: str):
    """Explicit launch only. Requires the original run to have stopped at its gate."""
    import fcntl
    from src.build_prompts import ROOT
    safe_run_id(source_run_id); safe_run_id(run_id)
    require(run_id != source_run_id, 'fresh allocation run ID required')
    out = ROOT/'results/phase3'
    # Existing shared lock serializes this controller with the original runner.
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = out/'gpu-attempts.json'
        attempts = json.loads(ledger.read_text())
        previous = ledger_seconds(attempts)
        require(run_id not in {a['run_id'] for a in attempts}, 'allocation ID already in lifetime ledger')
        package = load_source_package(ROOT, source_run_id, attempts)
        validate_payload(package)
        before = container_evidence()
        rawdir = ROOT/'results/raw'/run_id
        require(not rawdir.exists(), 'refuse to overwrite continuation raw directory')
        # Check remote uniqueness BEFORE any spawn. Same original results Volume.
        require_remote_fresh(run_id)
        continuation = {name: file_hash(ROOT/name) for name in
                        ('src/phase3_resume.py', 'src/modal_app_3_resume.py',
                         'src/phase3_validation.py', 'src/analyze_3.py')}
        resume_manifest = dict(schema_version=1, source_run_id=source_run_id, allocation_run_id=run_id,
            source_manifest=package['source_manifest'], source_artifacts_sha256=package['source_artifacts_sha256'],
            continuation_sha256=continuation, payload_sha256=package['payload_sha256'],
            timing_method=TIMING_METHOD, rate_usd_per_gpu_second=RATE, cap_usd=7.,
            parent_validation_required=True)
        # Recheck frozen files and stopped-source bytes immediately before submission.
        verify_hashes(ROOT, package['source_manifest']['source_sha256'])
        verify_hashes(ROOT, package['source_artifacts_sha256'])
        rawdir.mkdir(parents=True)
        write_json(rawdir/'launch-manifest.json', resume_manifest)
        write_json(rawdir/'containers-before.json', before)
        attempt = dict(run_id=run_id, source_run_id=source_run_id, status='running',
                       previous_seconds=previous, gpu='L40S', rate_usd_per_hour=RATE*3600)
        call, submitted, zero = None, False, False
        caught = None
        # Dict allocation/serialization before metering cannot leave a GPU submission.
        with modal.Dict.ephemeral() as progress:
            tick = time.monotonic(); allocation_start = time.time()
            limits = allocation_limits(previous, allocation_start)
            attempt.update(started_unix=allocation_start, **limits)
            attempts.append(attempt)
            write_json(ledger, attempts)
            try:
                submitted = True
                call = resume_study.with_options(timeout=limits['timeout_seconds'], retries=0).spawn(
                    package, resume_manifest, run_id, progress, previous, allocation_start,
                    limits['absolute_deadline_unix'])
                attempt['call_id'] = call.object_id
                write_json(ledger, attempts)
                seen = None
                while True:
                    if previous+time.monotonic()-tick >= CAP_SECONDS-60:
                        raise TimeoutError('Local Phase 3 lifetime allocation cap')
                    state = progress.get('state')
                    if state and state != seen:
                        write_json(rawdir/'local-progress.json', state); seen = state
                        print('LOCAL_RESUME '+json.dumps(state), flush=True)
                    try:
                        result = call.get(timeout=0)
                    except TimeoutError:
                        time.sleep(2)
                        continue
                    attempt.update(status=result['status'], function_seconds=result['function_seconds'])
                    break
            except BaseException as error:
                caught = error
                attempt['status'] = 'failed_or_cancelled' if call is not None else 'submission_unresolved'
            finally:
                # A successful response can precede container teardown. Terminate
                # explicitly and meter through zero-container verification.
                if call is not None:
                    try:
                        call.cancel(terminate_containers=True)
                    except Exception as error:
                        attempt['cancel_error'] = f'{type(error).__name__}: {error}'
                    # Bounded shutdown check; a failed/unknown check reserves the
                    # full remaining exposure and blocks every later retry.
                    for _ in range(15):
                        try:
                            evidence = container_evidence()
                            write_json(rawdir/'containers-after.json', evidence)
                            zero = True
                            attempt['zero_containers_verified_unix'] = evidence['checked_unix']
                            break
                        except Exception as error:
                            attempt['shutdown_check_error'] = f'{type(error).__name__}: {error}'
                            if previous+time.monotonic()-tick >= CAP_SECONDS-10:
                                break
                            time.sleep(2)
                elapsed = time.monotonic()-tick
                attempt.update(close_allocation(call_known=call is not None, elapsed=elapsed,
                               previous_seconds=previous, status=attempt['status'], zero_containers=zero),
                               ended_unix=time.time())
                write_json(ledger, attempts)
                # Recover even if the remote call was cancelled or failed. A lost
                # spawn handle can still have written this unique Volume path.
                recovery_errors = recover_artifacts(rawdir, run_id) if submitted else []
                write_json(rawdir/'local-execution.json', dict(attempt=attempt,
                           allocation_gpu_usd_upper=sum(a['wall_seconds'] for a in attempts)*RATE,
                           recovery_errors=recovery_errors, parent_validation_required=True))
        print(json.dumps(dict(run_id=source_run_id, allocation_run_id=run_id, status=attempt['status'],
                              raw_directory=str(rawdir), parent_validation_required=True)), flush=True)
        if caught is not None:
            raise caught
        require(zero and not recovery_errors, 'continuation requires shutdown/artifact reconciliation')
