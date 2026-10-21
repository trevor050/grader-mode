"""Descriptive, single-position next-token probe. CPU preflight never allocates.

This is separate from persistent generated-token interventions and registered
analyses. Only the three fresh explorer baseline responses may supply prefixes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.models import MODELS
from src.phase3 import MODEL, RATE, CAP_SECONDS
from src.phase3_resume import (close_allocation, digest, file_hash, ledger_seconds,
    number, require, safe_run_id, verify_hashes)

SOURCE = 'phase3-20261020-v2'
EXPLORER = 'phase3-explore-20261021-a'
CELLS = ('baseline', 'top1_plus8delta', 'top1_minus8delta', 'top1_zero',
         'tutor20_plus8delta', 'tutor20_minus8delta', 'tutor20_zero',
         'top1_plus64delta', 'top1_minus64delta', 'control0_plus8')
PROBE_USD = .60
STARTUP_SECONDS, SHUTDOWN_SECONDS, MIN_SETUP_SECONDS = 120, 180, 120
HEADROOM = 1.3
PATHS = ('src/phase3_logit_probe.py', 'src/modal_app_3_logit_probe.py',
         'src/phase3_explore.py', 'src/phase3_resume.py',
         'src/modal_app_3_resume.py', 'src/modal_app.py')


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def lifetime_seconds(attempts):
    return ledger_seconds(attempts)


def probe_limits(previous_seconds, now_unix):
    require(number(previous_seconds) and previous_seconds < CAP_SECONDS
            and number(now_unix, positive=True), 'invalid lifetime exposure/clock')
    exposure = min(PROBE_USD/RATE, CAP_SECONDS-previous_seconds)
    require(exposure >= STARTUP_SECONDS+MIN_SETUP_SECONDS+SHUTDOWN_SECONDS+60,
            'remaining budget cannot cover setup, timing pair and shutdown')
    return dict(exposure_seconds=exposure, exposure_usd=exposure*RATE,
                startup_timeout_seconds=STARTUP_SECONDS,
                timeout_seconds=int(exposure-STARTUP_SECONDS-SHUTDOWN_SECONDS),
                absolute_deadline_unix=now_unix+exposure-SHUTDOWN_SECONDS,
                local_cancel_unix=now_unix+exposure-SHUTDOWN_SECONDS)


def grid_fits(durations, forwards_left, remaining_seconds):
    require(bool(durations) and all(number(d, positive=True) for d in durations)
            and type(forwards_left) is int and forwards_left >= 0
            and number(remaining_seconds), 'invalid forward timing gate')
    return max(durations)*HEADROOM*forwards_left+SHUTDOWN_SECONDS <= remaining_seconds


def condition(cell, neurons):
    require(cell in CELLS, 'unknown logit condition')
    mode = 'zero' if cell.endswith('_zero') else 'add'
    coefficient = (-1 if 'minus' in cell else 1)*(64 if '64' in cell else 8)
    if mode == 'zero' or cell == 'baseline':
        coefficient = 0
    tutor = neurons['tutor']
    group = ([] if cell == 'baseline' else neurons['controls'][0]
             if cell == 'control0_plus8' else tutor[:1] if cell.startswith('top1') else tutor)
    values = ([8*abs(t['delta'])*s for t, s in zip(tutor, neurons['control_signs'][0])]
              if cell == 'control0_plus8' else [coefficient*n['delta'] for n in group])
    return dict(cell=cell, mode=mode, coefficient=coefficient,
                coordinates=[dict(id=n['id'], layer=n['layer'], neuron=n['neuron'], amount=v)
                             for n, v in zip(group, values)])


def last_position_hook(indices, values, mode):
    """Clone float32, edit only x[:, -1, selected], cast back to input dtype."""
    require(mode in ('add', 'zero'), 'invalid intervention mode')
    def hook(module, args):
        x = args[0]
        require(x.ndim == 3 and x.shape[1] > 0, 'expected batch/sequence/MLP activations')
        y = x.float().clone()
        y[:, -1, indices] = 0 if mode == 'zero' else y[:, -1, indices]+values
        return (y.to(x.dtype), *args[1:])
    return hook


def prepare_prefix(tokenizer, rendered, response, expected_prompt_hash):
    """Cut at first non-whitespace final character using full-text fast offsets.

    Re-tokenizing the exact prefix must reproduce the full-text token prefix;
    prompt tokens must also be identical. A crossing token is excluded, never
    moved to a neighboring position. No generated IDs are available in explorer
    artifacts, so this explicitly verifies a reconstructed teacher-forced prefix.
    """
    require(tokenizer.is_fast, 'fast tokenizer offsets required')
    require(text_hash(rendered) == expected_prompt_hash, 'original rendered prompt hash mismatch')
    result = dict(eligible=False, rendered_prompt_sha256=expected_prompt_hash,
                  response_sha256=text_hash(response))
    end = response.find('</think>')
    if end < 0:
        return dict(result, exclusion='missing_closed_think')
    start = end+len('</think>')
    while start < len(response) and response[start].isspace():
        start += 1
    if start == len(response):
        return dict(result, exclusion='empty_final')
    boundary = len(rendered)+start
    full = tokenizer(rendered+response, add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids = tokenizer(rendered, add_special_tokens=False)['input_ids']
    ids, offsets = full['input_ids'], full['offset_mapping']
    require(ids[:len(prompt_ids)] == prompt_ids, 'original prompt token prefix changed')
    if any(a < boundary < b for a, b in offsets):
        return dict(result, exclusion='boundary_inside_token', final_char_offset=start)
    candidates = [i for i, (a, b) in enumerate(offsets) if a == boundary and b > a]
    if len(candidates) != 1:
        return dict(result, exclusion='ambiguous_token_boundary', final_char_offset=start)
    cut = candidates[0]
    if cut <= len(prompt_ids) or offsets[cut-1][0] < len(rendered):
        return dict(result, exclusion='last_input_not_generated', final_char_offset=start)
    prefix = rendered+response[:start]
    prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
    require(prefix_ids == ids[:cut] and tokenizer.decode(prefix_ids, skip_special_tokens=False,
            clean_up_tokenization_spaces=False) == prefix, 'exact response token prefix changed')
    return dict(result, eligible=True, final_char_offset=start, full_char_offset=boundary,
                response_prefix=response[:start], input_ids=prefix_ids,
                response_prefix_sha256=text_hash(response[:start]),
                full_prefix_sha256=text_hash(prefix), token_prefix_sha256=digest(prefix_ids),
                token_prefix_verified=True, prompt_token_count=len(prompt_ids),
                input_token_count=cut, last_input_offset=list(offsets[cut-1]),
                actual_next_token_id=ids[cut])


def log_probs(logits):
    import torch
    require(logits.ndim == 1 and bool(torch.isfinite(logits).all()), 'finite vocabulary logits required')
    return torch.log_softmax(logits.float(), dim=-1).cpu().numpy()


def contrast_stats(base, push, actual_id, decode):
    import numpy as np
    base, push = np.asarray(base, dtype=np.float32), np.asarray(push, dtype=np.float32)
    require(base.ndim == 1 and base.shape == push.shape and base.size > 1
            and np.isfinite(base).all() and np.isfinite(push).all()
            and type(actual_id) is int and 0 <= actual_id < base.size, 'invalid log-probability tensors')
    require(abs(float(np.exp(base).sum())-1) < 1e-4 and abs(float(np.exp(push).sum())-1) < 1e-4,
            'normalized full vocabulary log probabilities required')
    change = push-base
    def token(i):
        return dict(id=int(i), text_repr=repr(decode([int(i)])), baseline_logp=float(base[i]),
                    pushed_logp=float(push[i]), logp_change=float(change[i]),
                    baseline_probability=float(np.exp(base[i])), pushed_probability=float(np.exp(push[i])))
    n = min(10, base.size)
    baseline_top = [token(i) for i in np.argsort(-base, kind='stable')[:n]]
    pushed_top = [token(i) for i in np.argsort(-push, kind='stable')[:n]]
    return dict(most_increased=[token(i) for i in np.argsort(-change, kind='stable')[:n]],
                most_decreased=[token(i) for i in np.argsort(change, kind='stable')[:n]],
                baseline_top10=baseline_top, pushed_top10=pushed_top,
                baseline_argmax=baseline_top[0], pushed_argmax=pushed_top[0],
                actual_next_token=token(actual_id),
                kl_push_to_base=float(np.sum(np.exp(push).astype(np.float64)*change)),
                entropy_base=float(-np.sum(np.exp(base).astype(np.float64)*base)),
                entropy_push=float(-np.sum(np.exp(push).astype(np.float64)*push)))


def validate_package(package, *, rebuild_requests=True):
    from src.phase3_explore import validate_package as validate_exploration
    from src.phase3_validation import _neurons
    require(package.get('payload_sha256') == digest({k: v for k, v in package.items() if k != 'payload_sha256'}),
            'logit transport changed')
    require(package.get('schema_version') == 1 and package.get('source_run_id') == SOURCE
            and package['original']['source_run_id'] == SOURCE and package.get('explorer_run_id') == EXPLORER
            and package.get('analysis_excluded') is True and package.get('exploratory') is True
            and package.get('cells') == list(CELLS), 'fixed logit design changed')
    validate_exploration(package['original'], rebuild_requests=rebuild_requests)
    _neurons(package['original']['neurons'])
    require(package['previous_seconds'] == package['original']['previous_seconds'], 'lifetime exposure changed')
    probes = package['original']['probes']
    require(len(package['baselines']) == 3, 'exactly three explorer baseline responses required')
    for source, row in zip(probes, package['baselines']):
        require(row['source_request_id'] == source['id'] and row['cell'] == 'baseline'
                and row['run_id'] == EXPLORER and row['source_run_id'] == SOURCE
                and row['cell_complete'] is True and row['budget_terminated'] is False
                and row['model'] == MODEL and row['model_revision'] == MODELS[MODEL]
                and row['rendered_prompt_sha256'] == package['original']['prompt_hashes'][source['id']]
                and all(row[k] == v for k, v in source.items() if k not in ('id', 'phase')),
                'fresh explorer baseline provenance changed')
        require(isinstance(row['response'], str), 'baseline response must be text')
    require(bool(package.get('explorer_artifacts_sha256')), 'explorer evidence hashes required')


def load_package(root, source_run_id, explorer_run_id, run_id, attempts):
    """Read-only CPU provenance, with all attempts closed through verified zero."""
    from src.phase3_resume import load_source_package, validate_payload
    from src.phase3_explore import make_package, CELLS as EXPLORER_CELLS, EXPLORER_PATHS
    root = Path(root)
    require(source_run_id == SOURCE and explorer_run_id == EXPLORER, 'fixed original/explorer sources required')
    safe_run_id(run_id)
    require(run_id.startswith('phase3-logit-') and run_id not in {a['run_id'] for a in attempts}
            and not (root/'results/raw'/run_id).exists(), 'fresh phase3-logit- allocation ID required')
    require(not any(str(a.get('run_id', '')).startswith('phase3-logit-') for a in attempts),
            'one logit allocation only; the additional $0.60 allowance cannot reset on retry')
    previous = lifetime_seconds(attempts)
    probe_limits(previous, 1.)
    original = load_source_package(root, SOURCE, [a for a in attempts if a['run_id'] == SOURCE])
    validate_payload(original)
    original = make_package(original, attempts)
    directory = root/'results/raw'/EXPLORER
    hashes = {}
    def read(name):
        path = directory/name
        hashes[str(path.relative_to(root))] = file_hash(path)
        return json.loads(path.read_text())
    result, launch, execution = read('exploration-result.json'), read('launch-manifest.json'), read('local-execution.json')
    entries = [a for a in attempts if a['run_id'] == EXPLORER]
    require(len(entries) == 1 and entries[0]['status'] == 'completed'
            and execution['attempt'] == entries[0] and not execution['recovery_errors'],
            'explorer allocation must be closed and reconciled')
    require(result['status'] == 'completed' and result['stage'] == 'finished'
            and result['partial_exploration'] is False and result['completed_cells'] == list(EXPLORER_CELLS)
            and result['output_rows'] == 15 and result['manifest'] == launch
            and launch['run_id'] == EXPLORER and launch['source_run_id'] == SOURCE
            and launch['source_manifest'] == original['source_manifest']
            and launch['source_selection'] == original['source_selection']
            and launch['source_artifacts_sha256'] == original['source_artifacts_sha256']
            and launch['probes'] == original['probes'] and launch['prompt_hashes'] == original['prompt_hashes']
            and set(EXPLORER_PATHS) <= launch['explorer_sha256'].keys(), 'completed original explorer evidence required')
    verify_hashes(root, launch['explorer_sha256'])
    require({'generations.jsonl', 'metrics.json', 'exploration-manifest.json', 'summary.json'}
            <= result['artifacts_sha256'].keys(), 'incomplete explorer artifact hashes')
    for name, expected in result['artifacts_sha256'].items():
        require(Path(name).name == name and file_hash(directory/name) == expected, 'explorer artifact bytes changed')
        hashes[str((directory/name).relative_to(root))] = expected
    rows = [json.loads(line) for line in (directory/'generations.jsonl').read_text().splitlines()]
    require(len(rows) == 15 and {(r['source_request_id'], r['cell']) for r in rows}
            == {(r['id'], c) for r in original['probes'] for c in EXPLORER_CELLS}, 'explorer generation grid changed')
    baselines = sorted((r for r in rows if r['cell'] == 'baseline'), key=lambda r: r['selection_rank'])
    package = dict(schema_version=1, source_run_id=SOURCE, explorer_run_id=EXPLORER,
                   original=original, baselines=baselines, previous_seconds=previous,
                   explorer_artifacts_sha256=hashes, cells=list(CELLS),
                   exploratory=True, analysis_excluded=True, ledger_sha256=digest(attempts))
    package['payload_sha256'] = digest(package)
    validate_package(package)
    return package


def main():
    from src.build_prompts import ROOT
    parser = argparse.ArgumentParser(description='Read-only logit preflight; never launches Modal')
    parser.add_argument('--source-run-id', default=SOURCE)
    parser.add_argument('--explorer-run-id', default=EXPLORER)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    package = load_package(ROOT, args.source_run_id, args.explorer_run_id, args.run_id,
                           json.loads((ROOT/'results/phase3/gpu-attempts.json').read_text()))
    print(json.dumps(dict(payload_sha256=package['payload_sha256'], cells=CELLS,
        analysis_excluded=True, limits=probe_limits(package['previous_seconds'], 1.),
        live_container_check='required by explicit launch controller',
        token_boundary_eligibility='requires pinned tokenizer at runtime'), indent=2))


if __name__ == '__main__':
    main()
