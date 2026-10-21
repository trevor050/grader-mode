"""CPU-only validation of persisted Phase 3 provenance and execution metadata.

The CLI verifies local frozen source files. Synthetic callers may instead supply
the raw/scored Phase 1b rows explicitly. Prompt hashes establish consistency and
match stored Phase 1b hashes where available; they do not replay the tokenizer.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from src.models import MODELS
from src.phase3 import (MODEL, RATE, build_requests, choose_test_templates,
                        reasoning_span, summarize_trajectory, validate_tracking_sources)
from src.score import read_rows

RAW_SOURCE = 'results/raw/phase1b-full/Qwen3-14B.jsonl'
SCORED_SOURCE = 'results/raw/phase1b-full/scored.jsonl'
SOURCE_FIELDS = ('phase', 'phase1b_id', 'problem_idx', 'selection_rank', 'split',
                 'template_idx', 'format', 'cue', 'arm', 'thinking', 'question',
                 'gold', 'messages', 'seed')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return type(value) is int and value >= 0


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _close(actual, expected, context):
    _require(_number(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12), context)


def _neurons(neurons):
    _require(neurons.get('k') == 20 and type(neurons['k']) is int, 'neurons require K=20')
    _require(neurons.get('random_seed') == 0 and type(neurons['random_seed']) is int,
             'neurons random_seed must be 0')
    _require(neurons.get('n_prompts_per_system') == 400, 'neurons require 400 development prompts per system')
    tutor, controls, signs = (neurons.get(k) for k in ('tutor', 'controls', 'control_signs'))
    _require(isinstance(tutor, list) and len(tutor) == 20, 'require 20 tutor neurons')
    _require(isinstance(controls, list) and len(controls) == 5, 'require five control sets')
    _require(isinstance(signs, list) and len(signs) == 5, 'require five control sign sets')
    groups = [tutor, *controls]
    for index, group in enumerate(groups):
        _require(isinstance(group, list) and len(group) == 20, 'each neuron set must contain 20 neurons')
        seen = set()
        for rank, neuron in enumerate(group):
            _require(isinstance(neuron, dict), 'neuron must be an object')
            l, n = neuron.get('layer'), neuron.get('neuron')
            _require(_integer(l) and l < 40 and _integer(n) and n < 17408, 'invalid neuron layer/index')
            _require(neuron.get('id') == f'{l}.{n}' and (l, n) not in seen, 'invalid/duplicate neuron ID')
            seen.add((l, n))
            keys = ('d', 'delta', 'tutor_mean', 'neutral_mean', 'tutor_sd', 'neutral_sd')
            _require(all(_number(neuron.get(k)) for k in keys), 'neuron statistics must be finite')
            _require(neuron['tutor_sd'] > 0 and neuron['neutral_sd'] >= 0, 'invalid neuron SD')
            delta = neuron['tutor_mean']-neuron['neutral_mean']
            _close(neuron['delta'], delta, 'neuron delta mismatch')
            _close(neuron['d'], delta/math.sqrt((neuron['tutor_sd']**2+neuron['neutral_sd']**2)/2),
                   'neuron d mismatch')
            if index == 0:
                _require(neuron['d'] > 0 and delta > 0, 'tutor neuron d/delta must be positive')
            else:
                _require(l == tutor[rank]['layer'], 'control neurons must match tutor layers in rank order')
                _require(neuron['id'] not in {r['id'] for r in tutor}, 'control includes tutor neuron')
        if index:
            s = signs[index-1]
            _require(isinstance(s, list) and len(s) == 20
                     and all(type(v) is int and v in (-1, 1) for v in s), 'invalid control signs')
    ordering = [(-r['d'], r['layer'], r['neuron']) for r in tutor]
    _require(ordering == sorted(ordering), 'tutor neurons not sorted by positive d, layer, neuron')


def _local_sources(root, manifest):
    root = Path(root).resolve()
    hashes = manifest.get('source_sha256')
    _require(isinstance(hashes, dict) and {RAW_SOURCE, SCORED_SOURCE} <= hashes.keys(),
             'manifest must hash both local Phase 1b source files')
    for name, digest in hashes.items():
        path = (root/name).resolve()
        _require(path.is_relative_to(root), 'source hash path escapes project')
        with path.open('rb') as handle:
            actual = hashlib.file_digest(handle, 'sha256').hexdigest()
        _require(actual == digest, f'local source_sha256 mismatch: {name}')
    original = json.loads((root/'results/raw/phase1-stage2/manifest.json').read_text())
    for key in ('dataset', 'config', 'split', 'revision', 'sample_seed', 'sample'):
        _require(manifest.get(key) == original.get(key), f'Phase 1 manifest {key} mismatch')
    return read_rows(root/RAW_SOURCE), read_rows(root/SCORED_SOURCE), len(hashes)


def validate_trajectories(tracking_rows, path):
    """Recompute every saved summary from finite ragged per-token six-set scores."""
    with np.load(path, allow_pickle=False) as raw:
        _require({'scores', 'token_ids', 'offsets', 'ids'} <= set(raw.files), 'trajectory archive missing arrays')
        scores, tokens, offsets, ids = (raw[k] for k in ('scores', 'token_ids', 'offsets', 'ids'))
        _require(scores.ndim == 2 and scores.shape[1] == 6 and np.isfinite(scores).all(),
                 'trajectory scores must be finite tokens x six sets')
        _require(tokens.ndim == 1 and np.issubdtype(tokens.dtype, np.integer)
                 and len(tokens) == len(scores) and (tokens >= 0).all(), 'invalid trajectory token IDs')
        _require(offsets.ndim == 1 and np.issubdtype(offsets.dtype, np.integer)
                 and len(offsets) == len(tracking_rows)+1 and offsets[0] == 0
                 and offsets[-1] == len(scores) and (np.diff(offsets) >= 0).all(), 'invalid trajectory offsets')
        _require(ids.ndim == 1 and len(ids) == len(tracking_rows)
                 and len(set(ids.tolist())) == len(ids), 'invalid trajectory trace IDs')
        indexed = {r['id']: r for r in tracking_rows}
        _require(set(ids.tolist()) == set(indexed), 'trajectory IDs do not match tracking')
        for i, rid in enumerate(ids):
            row = indexed[rid]
            values = scores[offsets[i]:offsets[i+1]]
            _require(len(values) == row['n_reasoning_tokens'], f'trajectory token count mismatch: {rid}')
            expected = summarize_trajectory(values)
            for key, value in expected.items():
                actual = row[key]
                if value is None:
                    _require(actual is None, f'trajectory {key} mismatch: {rid}')
                elif key == 'deciles':
                    for a, b in zip(actual, value):
                        _require((a is None and b is None) or (a is not None and b is not None
                                 and np.allclose(a, b, rtol=1e-6, atol=1e-7)), f'trajectory {key} mismatch: {rid}')
                else:
                    _require(np.allclose(actual, value, rtol=1e-6, atol=1e-7), f'trajectory {key} mismatch: {rid}')
    return len(tracking_rows)


def _validate_batches(rows, batches, expected):
    _require(isinstance(batches, list) and bool(batches), 'generation batches required')
    groups = defaultdict(list)
    for row in rows:
        groups[(row['split'], row['generation_cell'], row['c'])].append(row)
    for group in groups.values():
        group.sort(key=lambda r: (r['selection_rank'], r['template_idx']))
    cursors = defaultdict(int)
    test_started = False
    for batch in batches:
        _require(isinstance(batch, dict), 'invalid generation batch')
        _require(batch.get('split') in ('dev', 'test') and isinstance(batch.get('cell'), str)
                 and _number(batch.get('c')), 'invalid batch split/cell/dose')
        key = (batch.get('split'), batch['cell'], batch['c'])
        _require(not (test_started and key[0] == 'dev'), 'development batch follows test generation')
        test_started = test_started or key[0] == 'test'
        n, start = batch.get('n'), cursors[key]
        _require(_integer(n) and n > 0 and key in groups, 'invalid generation batch count/cell')
        group = groups[key][start:start+n]
        _require(len(group) == n, 'generation batch exceeds saved rows')
        _require(not batch.get('budget_terminated'), 'budget-shortened generation batch')
        _require(_number(batch.get('seconds')) and batch['seconds'] > 0, 'invalid batch seconds')
        _require(batch.get('first_id') == group[0]['phase1b_id']
                 and batch.get('last_id') == group[-1]['phase1b_id'], 'generation batch source IDs mismatch')
        seed = expected[group[0]['phase1b_id']]['seed']
        _require(type(batch.get('seed')) is int and batch['seed'] == seed
                 and all(r['batch_seed'] == seed for r in group), 'generation batch_seed mismatch')
        steps = batch.get('decode_steps')
        _require(_integer(steps) and 0 < steps <= 3072, 'invalid batch decode_steps')
        _require(all(r['output_tokens']+(r['finish_reason'] == 'stop') <= steps for r in group),
                 'output_tokens exceeds generation batch steps')
        cursors[key] += n
    _require(all(cursors[key] == len(group) for key, group in groups.items()), 'unaccounted generation rows in batches')


def _validate_selection(rows, selection, batches):
    trials = selection.get('dose_trials')
    _require(isinstance(trials, list) and 1 <= len(trials) <= 4, 'ordered dose_trials required')
    _require(all(isinstance(t, dict) and type(t.get('c')) is int for t in trials), 'invalid dose trial')
    _require([t['c'] for t in trials] == [1, 2, 4, 8][:len(trials)], 'dose_trials must be ordered 1,2,4,8')
    dev = [r for r in rows if r['split'] == 'dev']
    cells = {'baseline', *(f'dose_{t["c"]}' for t in trials)}
    _require({r['generation_cell'] for r in dev} == cells, 'development cells differ from dose_trials')
    def rate(cell):
        group = [r for r in dev if r['generation_cell'] == cell]
        _require(len(group) == 100 and {(r['selection_rank'], r['template_idx']) for r in group}
                 == {(rank, 0) for rank in range(1, 101)}, 'development requires 100 problems at template 0 per trial')
        valid = [r for r in group if not r['excluded_from_rates']]
        _require(bool(valid), 'development trial has no valid final replies')
        return sum(r['leak'] for r in valid)/len(valid), len(valid)
    base, _ = rate('baseline')
    _close(selection.get('dev_baseline_leak'), base, 'development baseline leak mismatch')
    chosen, met = 8, False
    for i, trial in enumerate(trials):
        leak, valid = rate(f'dose_{trial["c"]}')
        _require(type(trial.get('n')) is int and trial['n'] == 100
                 and type(trial.get('n_valid')) is int and trial['n_valid'] == valid, 'dose trial denominator mismatch')
        _close(trial.get('leak'), leak, 'dose trial leak mismatch')
        reduction = base-leak
        _close(trial.get('reduction'), reduction, 'dose trial reduction mismatch')
        if reduction >= .1-1e-12:
            chosen, met = trial['c'], True
            _require(i == len(trials)-1, 'dose trials continued after smallest qualifying dose')
            break
    _require(met or len(trials) == 4, 'fallback 8 requires all four dose trials')
    _require(type(selection.get('c_star')) is int and selection['c_star'] == chosen, 'c_star mismatch')
    _require(type(selection.get('dose_threshold_met')) is bool and selection['dose_threshold_met'] == met,
             'dose_threshold_met mismatch')
    templates = selection.get('test_templates')
    _require(type(templates) is int and templates in (1, 2), 'test_templates must be 1 or 2')
    _require(selection.get('test_generation_started') is True, 'test_generation_started must be true')
    _require(type(selection.get('timing_template_reduction')) is bool
             and selection['timing_template_reduction'] == (templates == 1), 'timing_template_reduction mismatch')
    timings = defaultdict(lambda: [0., 0])
    for batch in batches:
        if batch['split'] == 'dev':
            timings[batch['cell']][0] += batch['seconds']
            timings[batch['cell']][1] += batch['n']
    spr = max(seconds/n for seconds, n in timings.values())*1.2
    _close(selection.get('timing_seconds_per_request'), spr, 'timing_seconds_per_request mismatch')
    spent = selection.get('spent_gpu_seconds_at_test_gate')
    _require(_number(spent) and spent >= 0, 'invalid pretest timing spent seconds')
    _require(spent+1e-6 >= sum(b['seconds'] for b in batches if b['split'] == 'dev'),
             'pretest spent seconds omit development generation')
    _require(choose_test_templates(spr, spent) == templates, 'test_templates inconsistent with pretest timing gate')
    _close(selection.get('projected_test_seconds'), 100*templates*7*spr, 'projected_test_seconds mismatch')
    _close(selection.get('projected_gpu_usd'), (spent+100*templates*7*spr+300)*RATE, 'projected_gpu_usd mismatch')


def validate_execution(tracking_rows, generation_rows, neurons, selection, manifest, metrics, *,
                       source_raw=None, source_scored=None, source_root=None, trajectories=None,
                       allow_tracking_only=False):
    """Reject incomplete/invalid persisted runs before reporting execution verification.

    No generations is permitted only with explicit opt-in and budget_stopped
    metrics after all 400 tracking rows. Such verification never means completed.
    Structural grids and independent text rescoring reuse the analyzer checks.
    """
    from src.analyze_3 import _validate_generations, _validate_tracking
    for name, value in (('neurons', neurons), ('selection', selection), ('manifest', manifest), ('metrics', metrics)):
        _require(isinstance(value, dict), f'{name} must be an object')
    _require(type(allow_tracking_only) is bool, 'allow_tracking_only must be boolean')
    tracking_rows = list(tracking_rows)
    generation_rows = None if generation_rows is None else list(generation_rows)
    tracking_only = not generation_rows
    _require(metrics.get('status') == ('budget_stopped' if tracking_only and allow_tracking_only else 'completed'),
             'metrics.status must be completed (or explicit tracking-only budget_stopped)')
    _require(not metrics.get('budget_terminated') or tracking_only, 'metrics budget_terminated')
    _require(not tracking_only or allow_tracking_only, 'full run requires generations; use explicit tracking-only')
    _require(metrics.get('model') == MODEL and metrics.get('revision') == MODELS[MODEL], 'metrics model/revision must match pinned 14B')
    run_id = metrics.get('run_id')
    _require(isinstance(run_id, str) and bool(run_id.strip()), 'metrics run_id required')
    _require(manifest.get('phase') == 'phase3' and manifest.get('preregistration_commit') == '0e6a330'
             and manifest.get('cap_usd') == 7., 'manifest Phase 3 preregistration/cap mismatch')
    _require(metrics.get('manifest') == manifest, 'metrics manifest mismatch')
    hashes = manifest.get('source_sha256')
    _require(isinstance(hashes, dict) and bool(hashes)
             and all(isinstance(p, str) and p and _sha(h) for p, h in hashes.items()), 'invalid source_sha256')
    sample = manifest.get('sample')
    _require(isinstance(sample, list) and len(sample) == 200 and all(isinstance(p, dict)
             and _integer(p.get('problem_idx')) and type(p.get('gold')) is int
             and isinstance(p.get('question'), str) for p in sample), 'invalid manifest sample')
    requests = build_requests(sample)
    expected = {r['id']: r for r in requests}
    source_files_verified = 0
    if source_root is not None:
        _require(source_raw is None and source_scored is None, 'choose local source root or source data')
        source_raw, source_scored, source_files_verified = _local_sources(source_root, manifest)
    _require(source_raw is not None and source_scored is not None, 'matching Phase 1b raw/scored source data required')
    sources = {r['id']: r for r in validate_tracking_sources(requests, source_raw, source_scored)}
    _neurons(neurons)
    _validate_tracking(tracking_rows)
    _require(metrics.get('tracking_rows') == 400, 'metrics tracking_rows must be 400')
    prompt_hashes = {}
    def prompt_hash(source, digest):
        _require(_sha(digest), 'invalid rendered_prompt_sha256')
        key = json.dumps(source['messages'], sort_keys=True)
        _require(prompt_hashes.setdefault(key, digest) == digest, 'inconsistent rendered_prompt_sha256')
    for row in tracking_rows:
        rid = row['id']
        _require(rid in sources, 'unknown tracking Phase 1b source ID')
        source = sources[rid]
        for key in ('problem_idx', 'selection_rank', 'template_idx', 'gold', 'leak', 'excluded_from_rates'):
            _require(key in row and type(row[key]) is type(source[key]) and row[key] == source[key], f'tracking source {key} mismatch')
        _require(row.get('source_response_sha256') == hashlib.sha256(source['response'].encode()).hexdigest(),
                 'tracking source_response_sha256 mismatch')
        prompt_hash(source, row.get('rendered_prompt_sha256'))
        old_hash = source.get('source_rendered_prompt_sha256')
        _require(old_hash is None or old_hash == row['rendered_prompt_sha256'], 'Phase 1b rendered_prompt_sha256 mismatch')
        _require(row['closed_reasoning'] == (reasoning_span(source['response']) is not None), 'tracking closed_reasoning mismatch')
    n_trajectories = validate_trajectories(tracking_rows, trajectories) if trajectories is not None else 0
    if tracking_only:
        _require(metrics.get('output_rows', 0) == 0 and not metrics.get('batches'),
                 'tracking-only metrics contain generation batches/rows')
        _require(not selection and not metrics.get('selection'), 'tracking-only run contains generation selection')
    else:
        _require(metrics.get('selection') == selection, 'metrics selection mismatch')
        _require(type(metrics.get('output_rows')) is int and metrics['output_rows'] == len(generation_rows), 'metrics output_rows mismatch')
        _require(isinstance(metrics.get('sampling'), dict) and metrics['sampling'].get('max_new_tokens') == 3072,
                 'sampling max_new_tokens must be 3072')
        eos = metrics.get('eos_token_ids')
        _require(isinstance(eos, list) and bool(eos) and all(_integer(t) for t in eos)
                 and len(set(eos)) == len(eos), 'invalid generation eos_token_ids')
        templates = selection.get('test_templates')
        _require(type(templates) is int and templates in (1, 2), 'test_templates must be 1 or 2')
        _validate_generations(generation_rows, templates)
        cells = {'baseline', 'tutor', *(f'control_{i}' for i in range(5))}
        for row in generation_rows:
            source_id = row.get('phase1b_id')
            _require(isinstance(source_id, str), 'generation Phase 1b source ID must be a string')
            source = expected.get(source_id)
            _require(source is not None, 'unknown generation Phase 1b source ID')
            for key in SOURCE_FIELDS:
                _require(key in row and type(row[key]) is type(source[key]) and row[key] == source[key], f'generation source {key} mismatch')
            _require((row.get('run_id'), row.get('model'), row.get('model_revision'))
                     == (run_id, MODEL, MODELS[MODEL]), 'generation run/model/revision mismatch')
            cell, c = row.get('generation_cell'), row.get('c')
            _require(isinstance(cell, str) and _number(c), 'invalid generation cell/c')
            _require(row.get('id') == f'{source["id"]}:{cell}:c{c:g}', 'generation intervention ID mismatch')
            if row['split'] == 'test':
                _require(cell in cells and row['intervention'] == cell, 'invalid test intervention identity')
                _require(c == (0 if cell == 'baseline' else selection.get('c_star')), 'test c differs from fixed selection')
            else:
                _require((cell == 'baseline' and c == 0 and row['intervention'] == 'baseline')
                         or (cell in ('dose_1', 'dose_2', 'dose_4', 'dose_8') and c == int(cell[5:])
                             and row['intervention'] == 'tutor'), 'invalid development intervention identity')
            _require(not row.get('budget_terminated'), 'budget-shortened generation row')
            finish, tokens = row.get('finish_reason'), row['output_tokens']
            _require(finish in ('stop', 'length') and tokens <= 3072, 'invalid finish_reason/output_tokens')
            _require((finish == 'length') == row['truncated'], 'finish_reason/truncated mismatch')
            _require(finish != 'length' or tokens == 3072, 'length finish before 3072 tokens (budget shortening)')
            end = row.get('end_token_id')
            _require((finish == 'length' and end is None)
                     or (finish == 'stop' and _integer(end) and end in eos), 'finish_reason/end_token_id mismatch')
            _require(_integer(row.get('batch_seed')), 'invalid batch_seed')
            prompt_hash(source, row.get('rendered_prompt_sha256'))
        batches = metrics.get('batches')
        _validate_batches(generation_rows, batches, expected)
        _validate_selection(generation_rows, selection, batches)
    return {'completed': not tracking_only, 'h14_missing': tracking_only,
            'n_tracking_rows': len(tracking_rows), 'n_source_labels_verified': len(sources),
            'n_source_files_verified': source_files_verified, 'n_trajectories_verified': n_trajectories,
            'n_dev_rows': sum(r['split'] == 'dev' for r in generation_rows or []),
            'n_test_rows': sum(r['split'] == 'test' for r in generation_rows or [])}
