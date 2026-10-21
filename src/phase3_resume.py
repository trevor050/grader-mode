"""Fail-closed CPU contracts for an optional, separate Phase 3 allocation.

Preparation only; importing this module never contacts Modal. The safe local
preflight is ``uv run python -m src.phase3_resume --source-run-id <source>``.
Only a later, explicitly authorized ``modal run -m src.modal_app_3_resume``
allocates a GPU. Neither entry point merges into results/phase3.

Parent integration, ONLY if a resume is actually used:
1. Verify resume-result.json status=completed, zero-container evidence, and the
   SAME lifetime gpu-attempts.json. Retain source_manifest/source_selection and
   all artifact hashes. Never include planning-pilot.jsonl in study generations.
2. Combine original development rows with new generations.jsonl test rows;
   use original tracking/trajectories/neurons unchanged. New test run_id is the
   original logical study; allocation_run_id identifies the continuation.
3. Extend phase3_validation._validate_selection for timing_method below: check
   pilot provenance/tokens, exact safe capacity, lifetime spent including startup,
   ceil(100*templates/B)*7 full-cost batches * 1.2, 300s reserve, and 2-then-1
   choice BEFORE any test row. Validate identical boundaries/seeds in seven cells.
   The existing validator intentionally does NOT certify this timing method.
4. Produce a NEW integration manifest/metrics/summary only after parent review;
   do not replace frozen source hashes with hashes of these continuation modules.

resume-result.json schema v1: run_id (logical), allocation_run_id, status,
selection (original dose fields plus timing fields), planning_pilot, batches,
output_rows, source_manifest, source_selection, resume_manifest, sampling,
versions, accounting, artifacts_sha256, parent_validation_required. An incomplete
status retains complete batches only and must never be analyzed as a full grid.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

from src.models import MODELS
from src.phase3 import MODEL, RATE, CAP_SECONDS, build_requests, score_generation

TIMING_METHOD = 'resume_full_batch_pilot_v1'
WORK_RESERVE = 4*1024**3
SAMPLING = dict(enable_thinking=True, dtype='bfloat16', temperature=.7, top_p=.95,
                top_k=0, max_new_tokens=3072, left_padding=True)
CLOSED = {'completed', 'failed', 'budget_stopped', 'failed_or_cancelled'}
FROZEN_PATHS = ('DESIGN.md', 'prompts/system_tutor.txt', 'prompts/system_neutral.txt',
                'prompts/templates_1b.json', 'src/models.py', 'src/phase3.py',
                'src/modal_app_3.py', 'src/hf_decode.py', 'src/score.py',
                'results/raw/phase1b-full/Qwen3-14B.jsonl',
                'results/raw/phase1b-full/scored.jsonl')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify_hashes(root, hashes):
    root = Path(root).resolve()
    require(isinstance(hashes, dict) and bool(hashes), 'source hashes required')
    for name, expected in hashes.items():
        require(isinstance(name, str) and not Path(name).is_absolute(), 'invalid source path')
        path = (root/name).resolve()
        require(path.is_relative_to(root), 'source path escapes project')
        require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'invalid SHA256')
        require(file_hash(path) == expected, f'frozen source changed: {name}')


def safe_run_id(run_id):
    require(isinstance(run_id, str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9-]{0,99}', run_id), 'unsafe run ID')
    return run_id


def ledger_seconds(attempts):
    require(isinstance(attempts, list) and bool(attempts), 'existing lifetime ledger required')
    seen = set()
    for a in attempts:
        require(isinstance(a, dict) and a.get('status') in CLOSED, 'reconcile running/unresolved allocation before resume')
        require(isinstance(a.get('run_id'), str) and a['run_id'] not in seen, 'invalid/duplicate allocation ID')
        seen.add(a['run_id'])
        require(number(a.get('wall_seconds')) and number(a.get('ended_unix'), positive=True), 'closed allocation time required')
        start, zero = a.get('started_unix'), a.get('zero_containers_verified_unix')
        require(number(start, positive=True) and number(zero, positive=True)
                and start <= zero <= a['ended_unix']+.001, 'prior allocation shutdown must be verified')
        require(a['wall_seconds']+.001 >= zero-start, 'prior allocation shutdown tail is not fully charged')
    total = sum(a['wall_seconds'] for a in attempts)
    require(total < CAP_SECONDS-300, 'lifetime GPU budget exhausted')
    return total


def allocation_limits(previous_seconds, now_unix):
    require(number(previous_seconds) and number(now_unix, positive=True), 'invalid allocation clock')
    timeout = int(CAP_SECONDS-previous_seconds-120-180)
    require(timeout >= 60, 'GPU budget exhausted')
    return dict(timeout_seconds=timeout, startup_timeout_seconds=120,
                absolute_deadline_unix=now_unix+CAP_SECONDS-previous_seconds-180,
                local_cancel_unix=now_unix+CAP_SECONDS-previous_seconds-60)


def close_allocation(*, call_known, elapsed, previous_seconds, status, zero_containers):
    require(number(elapsed) and number(previous_seconds), 'invalid elapsed accounting')
    resolved = call_known and zero_containers
    return dict(status=status if resolved else ('shutdown_unresolved' if call_known else 'submission_unresolved'),
                observed_wall_seconds=elapsed,
                wall_seconds=elapsed if resolved else max(0., CAP_SECONDS-previous_seconds),
                accounting_basis='allocation wall time through verified shutdown' if resolved
                else 'entire remaining exposure reserved pending explicit reconciliation')


def validate_no_containers(rows):
    """Modal 1.6.2 container-list JSON schema, verified from installed CLI source."""
    require(isinstance(rows, list), 'container list must be JSON array')
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get('app_name'), str)
                and isinstance(row.get('app_id'), str), 'unknown container evidence schema')
        require(not row['app_name'].startswith('grader-mode-phase3'), 'Phase 3 container still present')


def exact_batch_capacity(free_bytes, kv_bytes_per_token, max_prompt_tokens):
    require(type(free_bytes) is int and free_bytes > 0 and type(kv_bytes_per_token) is int
            and kv_bytes_per_token > 0 and type(max_prompt_tokens) is int and max_prompt_tokens > 0,
            'invalid capacity inputs')
    raw = (free_bytes-WORK_RESERVE)//(kv_bytes_per_token*(max_prompt_tokens+3072))
    require(raw >= 1, '4 GiB work reserve leaves no safe batch capacity')
    return min(64, raw)


def project_test_plan(full_batch_seconds, batch_size, spent_seconds):
    require(number(full_batch_seconds, positive=True) and number(spent_seconds)
            and type(batch_size) is int and 1 <= batch_size <= 64, 'invalid pilot/budget inputs')
    alternatives = []
    for templates in (2, 1):
        batches = math.ceil(100*templates/batch_size)
        seconds = batches*7*full_batch_seconds*1.2
        total = spent_seconds+seconds+300
        alternatives.append(dict(test_templates=templates, batches_per_cell=batches,
                                 projected_test_seconds=seconds, projected_gpu_usd=total*RATE,
                                 fits=total <= CAP_SECONDS))
        if total <= CAP_SECONDS:
            return dict(timing_method=TIMING_METHOD, test_templates=templates, batch_size=batch_size,
                        full_batch_seconds=full_batch_seconds, headroom=1.2, reserve_seconds=300,
                        spent_gpu_seconds_at_test_gate=spent_seconds, batches_per_cell=batches,
                        projected_test_seconds=seconds, projected_gpu_usd=total*RATE,
                        timing_template_reduction=templates == 1, alternatives=alternatives)
    raise ValueError('Reduced test grid exceeds GPU budget: '+json.dumps(alternatives))


def make_test_batches(rows, templates, batch_size):
    require(type(templates) is int and templates in (1, 2) and type(batch_size) is int
            and 1 <= batch_size <= 64, 'invalid test batch plan')
    require(len(rows) == 200 and len({r['id'] for r in rows}) == 200
            and {(r['selection_rank'], r['template_idx']) for r in rows}
            == {(i, t) for i in range(101, 201) for t in (0, 1)}
            and all(r['split'] == 'test' for r in rows), 'complete untouched test request grid required')
    ordered = sorted((r for r in rows if r['template_idx'] < templates),
                     key=lambda r: (r['selection_rank'], r['template_idx']))
    return [ordered[i:i+batch_size] for i in range(0, len(ordered), batch_size)]


def make_generation_row(source, *, cell, c, source_run_id, allocation_run_id, response,
                        output_tokens, end_token_id, batch_seed, rendered_prompt_sha256,
                        final_reply_tokens):
    finish = 'stop' if end_token_id is not None else 'length'
    require(type(output_tokens) is int and 0 <= output_tokens <= 3072
            and (finish != 'length' or output_tokens == 3072), 'incomplete decoder output')
    require(end_token_id is None or end_token_id in (151643, 151645), 'unexpected EOS')
    return {**source, 'id': f'{source["id"]}:{cell}:c{c:g}', 'intervention': cell,
            'generation_cell': cell, 'c': c, 'model': MODEL, 'model_revision': MODELS[MODEL],
            'run_id': source_run_id, 'allocation_run_id': allocation_run_id,
            'response': response, 'output_tokens': output_tokens, 'finish_reason': finish,
            'end_token_id': end_token_id, 'batch_seed': batch_seed, 'budget_terminated': False,
            'rendered_prompt_sha256': rendered_prompt_sha256, 'final_reply_tokens': final_reply_tokens,
            **score_generation(response, source['gold'], finish)}


def validate_eligibility(*, source_run_id, requests, generation_rows, tracking_rows, neurons,
                         selection, metrics, manifest, attempts):
    """Only the exact original pre-test projection failure may be continued."""
    from src.phase3_validation import _neurons, _validate_batches, SOURCE_FIELDS
    from src.analyze_3 import _validate_tracking
    previous = ledger_seconds(attempts)
    require(not any(a.get('source_run_id') == source_run_id for a in attempts),
            'continuation is single-use for this logical source; reconcile existing continuation')
    source = [a for a in attempts if a['run_id'] == source_run_id]
    require(len(source) == 1, 'source allocation absent/ambiguous in lifetime ledger')
    require(metrics.get('run_id') == source_run_id and metrics.get('status') in ('failed', 'budget_stopped')
            and source[0]['status'] == metrics['status'] and metrics.get('stage') == 'finished', 'source not stopped at final checkpoint')
    require(metrics.get('stop_reason') == 'ValueError: Reduced test grid exceeds GPU budget',
            'only explicit original pre-test budget-projection failure is eligible')
    require(metrics.get('manifest') == manifest and manifest.get('phase') == 'phase3'
            and manifest.get('preregistration_commit') == '0e6a330' and manifest.get('cap_usd') == 7., 'original metadata changed')
    require(metrics.get('model') == MODEL and metrics.get('revision') == MODELS[MODEL]
            and metrics.get('gpu') == 'L40S' and metrics.get('sampling') == SAMPLING, 'original model/sampling changed')
    require(metrics.get('screen_prompts_per_system') == [400, 400] and metrics.get('tracking_rows') == 400,
            'full original screening/tracking required')
    require(selection.get('test_generation_started') is False and selection.get('test_templates') is None
            and not metrics.get('test_generation_started'), 'test generation already planned/started')
    require(metrics.get('selection') is None or metrics['selection'] == selection, 'selection metadata changed')
    require(requests == build_requests(manifest['sample']), 'original request metadata changed')
    require(isinstance(generation_rows, list) and bool(generation_rows)
            and all(r.get('split') == 'dev' and r.get('selection_rank', 999) <= 100 for r in generation_rows),
            'resume requires zero existing test rows')
    require(metrics.get('n_test_rows', 0) == 0 and metrics.get('output_rows') == len(generation_rows), 'test/output row accounting mismatch')
    _neurons(neurons)
    _validate_tracking(tracking_rows)
    require(len(tracking_rows) == 400, 'require 400 tracking rows')
    expected = {r['id']: r for r in requests}
    trials = selection.get('dose_trials')
    require(isinstance(trials, list) and 1 <= len(trials) <= 4
            and all(type(t.get('c')) is int for t in trials)
            and [t['c'] for t in trials] == [1, 2, 4, 8][:len(trials)], 'ordered original dose grid required')
    cells = {'baseline', *(f'dose_{t["c"]}' for t in trials)}
    require({r.get('generation_cell') for r in generation_rows} == cells, 'unexpected/missing development cells')
    groups = defaultdict(list)
    seen = set()
    prompt_hashes = {}
    eos = metrics.get('eos_token_ids')
    require(isinstance(eos, list) and set(eos) == {151643, 151645} and len(eos) == 2, 'both pinned EOS IDs required')
    for row in generation_rows:
        src = expected.get(row.get('phase1b_id'))
        require(src is not None, 'unknown original source ID')
        require(all(key in row and type(row[key]) is type(src[key]) and row[key] == src[key]
                    for key in SOURCE_FIELDS), 'original row source metadata changed')
        cell, c = row['generation_cell'], row.get('c')
        require(number(c) and c == (0 if cell == 'baseline' else int(cell[5:])), 'invalid development dose')
        require(row.get('intervention') == ('baseline' if cell == 'baseline' else 'tutor'), 'invalid development intervention')
        require(row.get('id') == f'{src["id"]}:{cell}:c{c:g}' and row['id'] not in seen, 'invalid/duplicate original row ID')
        seen.add(row['id'])
        require((row.get('run_id'), row.get('model'), row.get('model_revision'))
                == (source_run_id, MODEL, MODELS[MODEL]), 'original run/model mismatch')
        require(row.get('budget_terminated') is False and type(row.get('output_tokens')) is int
                and 0 <= row['output_tokens'] <= 3072, 'invalid/incomplete development generation')
        prompt_hash = row.get('rendered_prompt_sha256')
        require(isinstance(prompt_hash, str) and re.fullmatch('[0-9a-f]{64}', prompt_hash)
                and prompt_hashes.setdefault(src['id'], prompt_hash) == prompt_hash,
                'original development prompt hashes inconsistent')
        finish = row.get('finish_reason')
        require(finish in ('stop', 'length') and (finish != 'length' or row['output_tokens'] == 3072)
                and ((finish == 'length' and row.get('end_token_id') is None)
                     or (finish == 'stop' and row.get('end_token_id') in eos)), 'invalid finish/EOS')
        scored = score_generation(row['response'], row['gold'], finish)
        require(all(key in row and type(row[key]) is type(value) and row[key] == value
                    for key, value in scored.items()), 'development score changed')
        groups[cell].append(row)
    def rate(cell):
        rows = groups[cell]
        require(len(rows) == 100 and {(r['selection_rank'], r['template_idx']) for r in rows}
                == {(i, 0) for i in range(1, 101)}, 'each dev cell requires exactly 100 template-0 problems')
        valid = [r for r in rows if not r['excluded_from_rates']]
        require(bool(valid), 'development cell has no valid final replies')
        return sum(r['leak'] for r in valid)/len(valid), len(valid)
    base, _ = rate('baseline')
    require(selection.get('dev_baseline_leak') == base, 'baseline leak changed')
    chosen, met = 8, False
    for i, trial in enumerate(trials):
        leak, valid = rate(f'dose_{trial["c"]}')
        reduction = base-leak
        require(trial.get('n') == 100 and trial.get('n_valid') == valid
                and math.isclose(trial.get('leak', -1), leak, abs_tol=1e-12)
                and math.isclose(trial.get('reduction', -1), reduction, abs_tol=1e-12), 'original dose summary changed')
        if reduction >= .1-1e-12:
            chosen, met = trial['c'], True
            require(i == len(trials)-1, 'dose selection continued after first qualifying dose')
            break
    require(met or len(trials) == 4, 'fallback eight requires all four complete doses')
    require(type(selection.get('c_star')) is int and selection['c_star'] == chosen
            and selection.get('dose_threshold_met') is met, 'c_star changed')
    require(all(b.get('split') == 'dev' for b in metrics['batches']), 'test generation batch already exists')
    _validate_batches(generation_rows, metrics['batches'], expected)
    return dict(c_star=chosen, n_dev_rows=len(generation_rows), n_tracking_rows=400,
                n_test_rows=0, previous_seconds=previous, source_run_id=source_run_id)


def load_source_package(root, source_run_id, attempts):
    """Read/verify stopped source artifacts, without rewriting any source/output."""
    import numpy as np
    from src.score import read_rows
    from src.phase3 import choose_neurons, validate_tracking_sources, reasoning_span
    from src.phase3_validation import _local_sources, validate_trajectories
    root = Path(root).resolve()
    safe_run_id(source_run_id)
    rawdir, out = root/'results/raw'/source_run_id, root/'results/phase3'
    artifacts = {}
    def locate(name):
        candidates = [p for p in (rawdir/name, out/name) if p.is_file()]
        require(bool(candidates), f'missing original artifact: {name}')
        require(len({file_hash(p) for p in candidates}) == 1, f'conflicting source artifacts: {name}')
        path = candidates[0]
        artifacts[str(path.relative_to(root))] = file_hash(path)
        return path
    metadata = {name: json.loads(locate(name+'.json').read_text())
                for name in ('manifest', 'metrics', 'selection', 'neurons')}
    manifest = metadata['manifest']
    require(set(FROZEN_PATHS) <= manifest['source_sha256'].keys(), 'incomplete frozen source hashes')
    verify_hashes(root, manifest['source_sha256'])
    requests = build_requests(manifest['sample'])
    rows = list(read_rows(locate('generations.jsonl')))
    tracking = list(read_rows(locate('tracking.jsonl')))
    eligibility = validate_eligibility(source_run_id=source_run_id, requests=requests,
                    generation_rows=rows, tracking_rows=tracking, attempts=attempts, **metadata)
    source_raw, source_scored, _ = _local_sources(root, manifest)
    sources = {r['id']: r for r in validate_tracking_sources(requests, source_raw, source_scored)}
    for row in tracking:
        source = sources.get(row['id'])
        require(source is not None, 'tracking source ID mismatch')
        for key in ('problem_idx', 'selection_rank', 'template_idx', 'gold', 'leak', 'excluded_from_rates'):
            require(key in row and type(row[key]) is type(source[key]) and row[key] == source[key], 'tracking source metadata changed')
        require(row['source_response_sha256'] == hashlib.sha256(source['response'].encode()).hexdigest()
                and row['closed_reasoning'] == (reasoning_span(source['response']) is not None), 'tracking source response changed')
        old = source.get('source_rendered_prompt_sha256')
        require(old is None or old == row['rendered_prompt_sha256'], 'tracking prompt hash changed')
    validate_trajectories(tracking, locate('trajectories.npz'))
    with np.load(locate('screen-statistics.npz'), allow_pickle=False) as archive:
        stats = {k: archive[k] for k in ('d', 'delta', 'tutor_mean', 'neutral_mean', 'tutor_sd', 'neutral_sd')}
        require(all(a.shape == (40, 17408) for a in stats.values()), 'screen statistic shape changed')
        stats['n_prompts'] = 400
        require(choose_neurons(stats) == metadata['neurons'], 'neurons differ from original screen statistics')
    dev = sorted((r for r in requests if r['split'] == 'dev' and r['template_idx'] == 0),
                 key=lambda r: r['selection_rank'])
    test = sorted((r for r in requests if r['split'] == 'test' and r['template_idx'] < 2),
                  key=lambda r: (r['selection_rank'], r['template_idx']))
    prompt_hashes = {r['phase1b_id']: r['rendered_prompt_sha256'] for r in rows}
    prompt_hashes.update({r['id']: r['rendered_prompt_sha256'] for r in tracking})
    package = dict(source_run_id=source_run_id, source_manifest=manifest,
                   source_selection=metadata['selection'], source_metrics=metadata['metrics'],
                   neurons=metadata['neurons'], dev_requests=dev, test_requests=test,
                   prompt_hashes=prompt_hashes, eligibility=eligibility,
                   source_artifacts_sha256=artifacts)
    package['payload_sha256'] = digest(package)
    return package


def validate_payload(package, *, rebuild_requests=True):
    """Reject transport mutation and rederive exact dev/test input grids remotely."""
    from src.phase3_validation import _neurons
    require(package.get('payload_sha256') == digest({k: v for k, v in package.items() if k != 'payload_sha256'}), 'resume payload changed')
    requests = build_requests(package['source_manifest']['sample']) if rebuild_requests else None
    for key, split, templates in (('dev_requests', 'dev', 1), ('test_requests', 'test', 2)):
        if requests is not None:
            expected = sorted((r for r in requests if r['split'] == split and r['template_idx'] < templates),
                              key=lambda r: (r['selection_rank'], r['template_idx']))
            require(package[key] == expected, 'resume request source metadata changed')
        start = 1 if split == 'dev' else 101
        rows = package[key]
        require(len(rows) == 100*templates and len({r['id'] for r in rows}) == len(rows)
                and [(r['selection_rank'], r['template_idx']) for r in rows]
                == [(i, t) for i in range(start, start+100) for t in range(templates)]
                and all(r['split'] == split and r['thinking'] is True and r['cue'] == 'none'
                        and r['format'] == 'boxed' and r['arm'] == 'tutor' for r in rows), 'invalid resume input grid')
    _neurons(package['neurons'])
    require(package['source_selection']['test_generation_started'] is False
            and package['source_selection']['c_star'] == package['eligibility']['c_star']
            and package['eligibility']['n_test_rows'] == 0, 'resume selection changed')


def main():
    from src.build_prompts import ROOT
    parser = argparse.ArgumentParser(description='Read-only CPU preflight; never launches GPU work')
    parser.add_argument('--source-run-id', required=True)
    args = parser.parse_args()
    attempts = json.loads((ROOT/'results/phase3/gpu-attempts.json').read_text())
    package = load_source_package(ROOT, args.source_run_id, attempts)
    validate_payload(package)
    print(json.dumps(package['eligibility'], indent=2))


if __name__ == '__main__':
    main()
