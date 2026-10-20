"""Validate persisted execution/provenance, without scoring text or testing grids.

Development screens and alpha trials need not use the final test selection.
Rendered hashes attest consistency across identical prompts, not tokenizer replay.
"""
from __future__ import annotations

import json
import math
import re

from src.models import MODELS
from src.phase2 import build_requests, test_cells


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return type(value) is int and value >= 0


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-fA-F]{64}', value) is not None


def validate_execution(rows, selection, manifest, metrics):
    """Raise ValueError for invalid raw rows; return numerical verification counts.

    The embedded metrics manifest must match the supplied manifest, including
    optional source hashes. Seeds are Phase 1 identities; batch seeds only need
    to be nonnegative integers. Complete statistical grids belong to analyze_2.
    """
    for name, value in (('selection', selection), ('manifest', manifest), ('metrics', metrics)):
        _require(isinstance(value, dict), f'{name} must be a dict')
    _require(metrics.get('status') == 'completed', 'metrics.status must be completed')
    _require(not metrics.get('budget_terminated'), 'metrics budget_terminated')
    run_id, model, revision = (metrics.get(key) for key in ('run_id', 'model', 'revision'))
    _require(isinstance(run_id, str) and bool(run_id.strip()), 'metrics run_id required')
    _require(isinstance(model, str) and model in MODELS, 'metrics model is not registered')
    _require(revision == MODELS[model], 'metrics revision must match pinned MODELS revision')
    _require(metrics.get('manifest') == manifest, 'metrics manifest identity/source hashes mismatch')
    if 'source_sha256' in manifest:
        hashes = manifest['source_sha256']
        _require(isinstance(hashes, dict) and all(
            isinstance(path, str) and path and _sha256(digest)
            for path, digest in hashes.items()), 'invalid manifest source_sha256')
    _require(_integer(selection.get('layer')), 'selection.layer must be a nonnegative integer')
    _require(_number(selection.get('alpha')) and selection['alpha'] >= 0,
             'selection.alpha must be finite and nonnegative')
    if 'selection' in metrics:
        saved = metrics['selection']
        _require(isinstance(saved, dict), 'metrics selection must be a dict')
        for key in ('layer', 'alpha', 'test_templates'):
            if key in selection or key in saved:
                _require(key in selection and key in saved and saved[key] == selection[key],
                         f'metrics selection {key} mismatch')

    sample = manifest.get('sample')
    _require(isinstance(sample, list) and len(sample) == 200,
             'manifest.sample must contain the Phase 1 200 problems')
    for problem in sample:
        _require(isinstance(problem, dict) and _integer(problem.get('problem_idx'))
                 and type(problem.get('gold')) is int and isinstance(problem.get('question'), str),
                 'invalid manifest.sample problem metadata')
    expected = {row['phase1_id']: row for row in build_requests(sample)}
    try:
        rows = list(rows)
    except TypeError as error:
        raise ValueError('rows must be persisted dicts') from error
    _require(bool(rows), 'empty execution rows')
    if 'output_rows' in metrics:
        _require(_integer(metrics['output_rows']) and metrics['output_rows'] == len(rows),
                 'metrics output_rows mismatch')

    required = {'id', 'phase1_id', 'phase', 'model', 'model_revision', 'run_id',
                'problem_idx', 'selection_rank', 'split', 'template_idx', 'format',
                'cue', 'arm', 'question', 'gold', 'messages', 'seed', 'batch_seed',
                'intervention', 'layer', 'alpha', 'output_tokens', 'finish_reason',
                'rendered_prompt_sha256'}
    source_fields = ('phase1_id', 'phase', 'problem_idx', 'selection_rank', 'split',
                     'template_idx', 'format', 'cue', 'arm', 'question', 'gold', 'messages', 'seed')
    cells = set(test_cells())
    prompt_hashes, base_ids = {}, set()
    n_dev = 0
    for index, row in enumerate(rows):
        _require(isinstance(row, dict), f'row {index} must be a persisted dict')
        _require(not (required - row.keys()), f'row {index} missing fields: {sorted(required - row.keys())}')
        context = f'row {index} ({row["id"]})'
        _require((row['run_id'], row['model'], row['model_revision']) == (run_id, model, revision),
                 f'{context}: mixed run_id/model/revision or metrics mismatch')
        source_id = row['phase1_id']
        _require(isinstance(source_id, str) and source_id in expected,
                 f'{context}: unknown Phase 1 source ID')
        source = expected[source_id]
        for key in ('problem_idx', 'selection_rank', 'template_idx', 'seed'):
            _require(_integer(row[key]), f'{context}: {key} must be a nonnegative integer')
        _require(type(row['gold']) is int, f'{context}: gold must be an integer')
        for key in source_fields:
            _require(row[key] == source[key], f'{context}: {key} differs from build_requests(manifest.sample)')
        arm, layer, alpha = row['intervention'], row['layer'], row['alpha']
        _require(isinstance(arm, str), f'{context}: intervention must be a string')
        _require(row['id'] == f'{source_id}:{arm}', f'{context}: intervention ID mismatch')
        _require(_number(alpha), f'{context}: alpha must be finite')
        if arm == 'baseline':
            _require((row['format'], arm) in cells and layer is None and alpha == 0,
                     f'{context}: baseline requires an allowed cell, layer None and alpha 0')
        else:
            _require(_integer(layer), f'{context}: steered layer must be a nonnegative integer')
            if row['split'] == 'test':
                _require((row['format'], arm) in cells, f'{context}: invalid test intervention cell')
                _require(layer == selection['layer'], f'{context}: test layer differs from selection')
                expected_alpha = selection['alpha'] if arm.startswith('add_') else 0
                _require(alpha == expected_alpha, f'{context}: test alpha differs from selection')
            else:
                ablation = re.fullmatch(r'(?:screen|confirm)_L([0-9]+)', arm)
                addition = re.fullmatch(r'alpha_m([0-9]+)', arm)
                _require(bool(
                    (ablation and row['format'] == 'boxed'
                     and layer == int(ablation[1]) and alpha == 0)
                    or (addition and int(addition[1]) > 0
                        and row['format'] == 'none' and alpha >= 0)),
                    f'{context}: invalid development intervention metadata')
        _require(not row.get('budget_terminated'), f'{context}: budget_terminated')
        _require(row['finish_reason'] in ('stop', 'length'), f'{context}: invalid finish_reason')
        _require(_integer(row['output_tokens']) and row['output_tokens'] <= 768,
                 f'{context}: output_tokens must be an integer in 0..768')
        _require(row['finish_reason'] != 'length' or row['output_tokens'] == 768,
                 f'{context}: length finish before 768 output_tokens (deadline shortening)')
        _require(_integer(row['batch_seed']), f'{context}: batch_seed must be a nonnegative integer')
        digest = row['rendered_prompt_sha256']
        _require(_sha256(digest), f'{context}: invalid rendered_prompt_sha256')
        prompt = json.dumps(source['messages'], sort_keys=True)
        _require(prompt_hashes.setdefault(prompt, digest.lower()) == digest.lower(),
                 f'{context}: inconsistent rendered_prompt_sha256 for identical base prompts')
        base_ids.add(source_id)
        n_dev += row['split'] == 'dev'
    return {'n_rows': len(rows), 'n_dev_rows': n_dev, 'n_test_rows': len(rows) - n_dev,
            'n_base_prompts': len(base_ids)}
