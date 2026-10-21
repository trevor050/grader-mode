"""Separate, descriptive three-input neuron exploration; CPU preflight by default.

No registered study rows, selections, summaries, prompts, or findings are written.
Only the explicit Modal entry point may allocate, after parent launch approval.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from src.models import MODELS
from src.phase3 import MODEL, RATE, CAP_SECONDS, build_requests, score_generation
from src.phase3_resume import (SAMPLING, close_allocation, digest, file_hash,
    ledger_seconds, number, require, safe_run_id, verify_hashes)

CELLS = ('baseline', 'top1_plus', 'top1_minus', 'top1_zero', 'tutor20_minus')
C_STAR = 8
EXPLORATION_USD = 1.00
STARTUP_SECONDS = 120
SHUTDOWN_SECONDS = 180
MIN_SETUP_SECONDS = 120
MIN_CELL_SECONDS = 60  # Admission floor, not a claimed throughput measurement.
HEADROOM = 1.3
TOP1 = dict(id='22.5128', layer=22, neuron=5128,
            d=4.38622263324684, delta=.1027524962792331)
EXPLORER_PATHS = ('src/phase3_explore.py', 'src/modal_app_3_explore.py',
                  'src/phase3_resume.py', 'src/modal_app_3_resume.py', 'src/modal_app.py')


def lifetime_seconds(attempts):
    """All original, failed, continuation and exploration charges, through zero."""
    return ledger_seconds(attempts)


def exploration_limits(previous_seconds, now_unix):
    require(number(previous_seconds) and previous_seconds < CAP_SECONDS
            and number(now_unix, positive=True), 'invalid lifetime budget/clock')
    exposure = min(EXPLORATION_USD/RATE, CAP_SECONDS-previous_seconds)
    minimum = STARTUP_SECONDS+MIN_SETUP_SECONDS+2*MIN_CELL_SECONDS*HEADROOM+SHUTDOWN_SECONDS
    require(exposure >= minimum, 'remaining budget cannot cover minimum setup and one paired probe')
    return dict(exposure_seconds=exposure, exposure_usd=exposure*RATE,
                startup_timeout_seconds=STARTUP_SECONDS,
                timeout_seconds=int(exposure-STARTUP_SECONDS-SHUTDOWN_SECONDS),
                absolute_deadline_unix=now_unix+exposure-SHUTDOWN_SECONDS,
                local_cancel_unix=now_unix+exposure-SHUTDOWN_SECONDS)


def cells_fit(durations, *, remaining_seconds, cells=1):
    require(bool(durations) and all(number(d, positive=True) for d in durations)
            and number(remaining_seconds) and type(cells) is int and cells >= 1,
            'invalid measured cell budget')
    return max(durations)*HEADROOM*cells+SHUTDOWN_SECONDS <= remaining_seconds


def select_probes(requests):
    probes = sorted((r for r in requests if r['selection_rank'] <= 3
                     and r['template_idx'] == 0), key=lambda r: r['selection_rank'])
    require(len(probes) == 3 and [r['selection_rank'] for r in probes] == [1, 2, 3]
            and len({r['phase1b_id'] for r in probes}) == 3
            and all(r['split'] == 'dev' and r['thinking'] is True and r['arm'] == 'tutor'
                    and r['cue'] == 'none' and r['format'] == 'boxed' for r in probes),
            'exact first three development template-0 requests required')
    return probes


def condition(cell, neurons):
    require(cell in CELLS, 'unknown exploration condition')
    coefficient = {'baseline': 0, 'top1_plus': 8, 'top1_minus': -8,
                   'top1_zero': 0, 'tutor20_minus': -8}[cell]
    group = [] if cell == 'baseline' else neurons['tutor'][:1 if cell != 'tutor20_minus' else 20]
    return dict(cell=cell, coefficient=coefficient, c_star=C_STAR,
                mode='zero' if cell == 'top1_zero' else 'add',
                coordinates=[{k: n[k] for k in ('id', 'layer', 'neuron', 'delta')}
                             | dict(amount=coefficient*n['delta']) for n in group])


def input_hook(indices, values, state, mode):
    """Shared clone/add hook in float32, cast back; zero only chosen inputs.

    Works on NumPy in CPU tests and Torch BF16 in the actual GPU runner. The
    shared hook receives promoted activations so its in-place indexed addition
    cannot round a small delta to BF16 before adding it. Prefill is untouched.
    """
    require(mode in ('add', 'zero'), 'invalid coordinate mode')
    from src.modal_app_3 import push_hook
    add = push_hook(indices, values, state)

    def hook(module, args):
        x = args[0]
        if not state['generated']:
            return args
        if mode == 'zero':
            y = x.clone() if hasattr(x, 'clone') else x.copy()
            y[..., indices] = 0
        elif hasattr(x, 'float'):
            y = add(module, (x.float(), *args[1:]))[0].to(x.dtype)
        else:
            import numpy as np
            y = add(module, (x.astype(np.float32), *args[1:]))[0].astype(x.dtype)
        return (y, *args[1:])
    return hook


def make_package(original, attempts):
    previous = lifetime_seconds(attempts)
    probes = select_probes(original['dev_requests'])
    package = {k: original[k] for k in ('source_run_id', 'source_manifest', 'source_selection',
                                       'neurons', 'source_artifacts_sha256')}
    package.update(schema_version=1, exploratory=True, analysis_excluded=True,
                   probes=probes, prompt_hashes={r['id']: original['prompt_hashes'][r['id']] for r in probes},
                   previous_seconds=previous, ledger_sha256=digest(attempts), sampling=SAMPLING.copy(),
                   cells=list(CELLS), c_star=C_STAR)
    package['payload_sha256'] = digest(package)
    return package


def validate_package(package, *, rebuild_requests=True):
    require(package.get('payload_sha256') == digest({k: v for k, v in package.items() if k != 'payload_sha256'}),
            'exploration transport changed')
    require(package.get('schema_version') == 1 and package.get('exploratory') is True
            and package.get('analysis_excluded') is True and package.get('cells') == list(CELLS)
            and package.get('sampling') == SAMPLING and type(package.get('c_star')) is int
            and package['c_star'] == C_STAR, 'fixed exploration settings changed')
    selection = package['source_selection']
    require(type(selection.get('c_star')) is int and selection['c_star'] == C_STAR
            and selection.get('dose_threshold_met') is False
            and [(d.get('c'), d.get('n')) for d in selection.get('dose_trials', [])]
            == [(c, 100) for c in (1, 2, 4, 8)], 'immutable full-grid fallback c_star changed')
    tutor = package['neurons']['tutor']
    require(len(tutor) == 20 and all(tutor[0].get(k) == v for k, v in TOP1.items()), 'original strongest neuron changed')
    require(len({n['id'] for n in tutor}) == 20
            and all(type(n['layer']) is int and 0 <= n['layer'] < 40
                    and type(n['neuron']) is int and 0 <= n['neuron'] < 17408
                    and n['id'] == f'{n["layer"]}.{n["neuron"]}' and number(n['delta'], positive=True)
                    for n in tutor), 'invalid original coordinates')
    require(select_probes(package['probes']) == package['probes'], 'probe order changed')
    if rebuild_requests:
        require(package['probes'] == select_probes(build_requests(package['source_manifest']['sample'])),
                'probe differs from original manifest selection')
    require(set(package['prompt_hashes']) == {r['id'] for r in package['probes']}
            and all(isinstance(h, str) and len(h) == 64 and all(c in '0123456789abcdef' for c in h)
                    for h in package['prompt_hashes'].values()), 'original prompt hashes required')
    require(number(package.get('previous_seconds')) and package['previous_seconds'] < CAP_SECONDS,
            'invalid previous lifetime exposure')


def load_package(root, source_run_id, run_id, attempts):
    """CPU-only provenance verification. No remote imports, calls, or writes."""
    from src.phase3_resume import load_source_package, validate_payload
    safe_run_id(source_run_id); safe_run_id(run_id)
    require(run_id.startswith('phase3-explore-') and run_id != source_run_id, 'unique phase3-explore- ID required')
    previous = lifetime_seconds(attempts)
    require(run_id not in {a['run_id'] for a in attempts}, 'exploration ID already attempted')
    root = Path(root)
    require(not (root/'results/raw'/run_id).exists(), 'exploration output already exists')
    # The resumer's source validator forbids another *registered continuation*.
    # Validate the original allocation's source evidence independently; ALL ledger
    # entries above and below remain charged in the exploration budget/digest.
    source_attempts = [a for a in attempts if a['run_id'] == source_run_id]
    original = load_source_package(root, source_run_id, source_attempts)
    validate_payload(original)
    package = make_package(original, attempts)
    require(package['previous_seconds'] == previous, 'lifetime accounting changed')
    validate_package(package)
    exploration_limits(previous, 1.)  # Refuse insufficient setup/pair budget locally.
    return package


def make_generation_row(source, *, cell, neurons, source_run_id, run_id, response,
                        output_tokens, end_token_id, batch_seed, rendered_prompt_sha256,
                        final_reply_tokens, incomplete=False):
    require(type(output_tokens) is int and 0 <= output_tokens <= 3072
            and (end_token_id in (151643, 151645) or end_token_id is None)
            and (incomplete or end_token_id is not None or output_tokens == 3072),
            'incomplete decoder output must be explicitly excluded')
    require(type(final_reply_tokens) is int and final_reply_tokens >= 0, 'invalid final token count')
    finish = 'stop' if end_token_id is not None else 'length'
    return {**source, 'phase': 'phase3-exploration', 'id': f'{run_id}:{source["id"]}:{cell}',
            'source_request_id': source['id'], 'source_run_id': source_run_id, 'run_id': run_id,
            'allocation_run_id': run_id, 'exploratory': True, 'analysis_excluded': True,
            'cell': cell, 'generation_cell': cell, **condition(cell, neurons),
            'model': MODEL, 'model_revision': MODELS[MODEL], 'response': response,
            'output_tokens': output_tokens, 'final_reply_tokens': final_reply_tokens,
            'end_token_id': end_token_id, 'finish_reason': finish,
            'batch_seed': batch_seed, 'rendered_prompt_sha256': rendered_prompt_sha256,
            'budget_terminated': incomplete, 'cell_complete': not incomplete,
            **score_generation(response, source['gold'], finish)}


def summarize(rows, *, completed_cells):
    require(list(completed_cells) == list(CELLS[:len(completed_cells)]), 'completed cells must be ordered prefix')
    groups = defaultdict(list)
    for row in rows:
        if row['cell_complete'] and row['cell'] in completed_cells:
            groups[row['cell']].append(row)
    result = dict(exploratory=True, analysis_excluded=True,
                  total_toy_examples=sum(map(len, groups.values())),
                  incomplete_rows=sum(not r['cell_complete'] for r in rows),
                  completed_cells=list(completed_cells),
                  skipped_cells=[c for c in CELLS if c not in completed_cells],
                  partial_exploration=len(completed_cells) != len(CELLS),
                  cells={}, pairs_against_baseline={})
    for cell in completed_cells:
        group = groups[cell]
        require(len(group) == 3 and len({r['source_request_id'] for r in group}) == 3, 'full cell requires three toy examples')
        valid = [r for r in group if not r['excluded_from_rates']]
        counts = dict(n_toy_examples=3, n_valid=len(valid))
        counts.update({key: sum(bool(r[key]) for r in valid) for key in ('leak', 'commit', 'correct_commit')})
        counts.update({key: sum(bool(r[key]) for r in group) for key in
                       ('empty', 'repetition', 'truncated', 'truncation_without_final', 'tutor_mention')})
        for key in ('output_tokens', 'final_reply_tokens'):
            counts['mean_'+key] = statistics.mean(r[key] for r in group)
            counts['median_'+key] = statistics.median(r[key] for r in group)
        result['cells'][cell] = counts
        if cell == 'baseline':
            continue
        base = {r['source_request_id']: r for r in groups['baseline'] if not r['excluded_from_rates']}
        pairs = [(base[r['source_request_id']], r) for r in valid if r['source_request_id'] in base]
        result['pairs_against_baseline'][cell] = dict(n_matched_nonempty=len(pairs), **{
            key+'_difference_count': sum(int(b[key])-int(a[key]) for a, b in pairs)
            for key in ('leak', 'commit', 'correct_commit', 'tutor_mention')})
    return result


def main():
    from src.build_prompts import ROOT
    parser = argparse.ArgumentParser(description='Read-only CPU exploration preflight; no Modal calls')
    parser.add_argument('--source-run-id', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    attempts = json.loads((ROOT/'results/phase3/gpu-attempts.json').read_text())
    package = load_package(ROOT, args.source_run_id, args.run_id, attempts)
    print(json.dumps(dict(schema_version=1, run_id=args.run_id, source_run_id=args.source_run_id,
        exploratory=True, analysis_excluded=True, probe_ids=[r['id'] for r in package['probes']],
        c_star=C_STAR, cells=CELLS, payload_sha256=package['payload_sha256'],
        previous_gpu_usd=package['previous_seconds']*RATE,
        limits=exploration_limits(package['previous_seconds'], 1.),
        live_container_check='required again by launch controller; not performed by CPU preflight',
        gpu_throughput='unverified'), indent=2))


if __name__ == '__main__':
    main()
