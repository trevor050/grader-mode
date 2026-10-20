"""Execution checks use persisted row shapes, without GPU or response scoring."""
from copy import deepcopy
import hashlib
import json

import pytest

from src.models import MODELS
from src.phase2 import MODEL, build_requests, test_cells as cells


@pytest.fixture(scope='module')
def execution():
    sample = [{'problem_idx': 1000 - 3*i, 'question': f'{i+2} plus 3?', 'gold': 99}
              for i in range(200)]
    # Different source IDs can render exactly the same base prompt.
    sample[101]['question'] = sample[100]['question']
    manifest = {'phase': 'phase2', 'dataset': 'openai/gsm8k', 'config': 'main',
                'split': 'test', 'revision': 'dataset-revision', 'sample_seed': 0,
                'sample': sample, 'source_sha256': {'prompts/templates.json': 'a'*64}}
    selection = {'layer': 7, 'alpha': 4., 'test_templates': 2}
    rows = []

    def append(request, intervention, layer=None, alpha=0.):
        prompt = json.dumps(request['messages'], sort_keys=True)
        rows.append({**deepcopy(request), 'id': f"{request['id']}:{intervention}",
                     'intervention': intervention, 'layer': layer, 'alpha': alpha,
                     'run_id': 'synthetic-phase2', 'model': MODEL,
                     'model_revision': MODELS[MODEL], 'batch_seed': 123,
                     'rendered_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                     'output_tokens': 23, 'finish_reason': 'stop'})

    for request in build_requests(sample):
        if request['selection_rank'] in (101, 102) and request['template_idx'] < 2:
            for fmt, intervention in cells():
                if request['format'] == fmt:
                    append(request, intervention,
                           None if intervention == 'baseline' else selection['layer'],
                           selection['alpha'] if intervention.startswith('add_') else 0.)
        if request['selection_rank'] == 1 and request['template_idx'] == 0:
            if request['format'] == 'boxed':
                append(request, 'baseline')
                append(request, 'screen_L2', 2)
                append(request, 'confirm_L3', 3)
            elif request['format'] == 'none':
                append(request, 'baseline')
                append(request, 'alpha_m2', 3, 1.)
                append(request, 'alpha_m4', 3, 2.)
    metrics = {'status': 'completed', 'run_id': 'synthetic-phase2', 'model': MODEL,
               'revision': MODELS[MODEL], 'manifest': deepcopy(manifest),
               'output_rows': len(rows), 'selection': deepcopy(selection)}
    return rows, selection, manifest, metrics


@pytest.fixture
def case(execution):
    return deepcopy(execution)


def validate(case):
    from src.phase2_validation import validate_execution
    return validate_execution(*case)


def test_persisted_rows_and_partial_development_are_valid(case):
    result = validate(case)
    assert result['n_rows'] == 86
    assert result['n_dev_rows'] == 6
    assert result['n_test_rows'] == 80
    assert result['n_base_prompts'] == 22
    assert all(type(value) in (int, float) for value in result.values())
    assert all(row['batch_seed'] != row['seed'] for row in case[0])
    assert all('response' not in row for row in case[0])


def test_grid_completeness_is_left_to_analyzer(case):
    case[0][:] = case[0][:1]
    case[3]['output_rows'] = 1
    assert validate(case)['n_rows'] == 1


def test_zero_projection_scale_can_produce_zero_development_alpha(case):
    for row in case[0]:
        if row['intervention'].startswith('alpha_m'):
            row['alpha'] = 0.
    validate(case)


@pytest.mark.parametrize('field,value', [
    ('run_id', 'other-run'), ('model', 'Qwen/Qwen3-4B'),
    ('model_revision', 'other-revision'), ('phase1_id', 'unknown-source'),
    ('id', 'other-id'), ('split', 'dev'), ('selection_rank', 102),
    ('problem_idx', 99999), ('template_idx', 3), ('phase', 'phase1'),
    ('cue', 'human'), ('arm', 'ceiling'), ('format', 'boxed'),
    ('question', 'changed question'), ('gold', 100), ('seed', 123),
    ('gold', True), ('selection_rank', 101.), ('seed', True),
    ('batch_seed', -1), ('batch_seed', 1.5), ('batch_seed', True),
    ('batch_seed', None), ('rendered_prompt_sha256', 'x'*64),
    ('rendered_prompt_sha256', 'a'*63), ('finish_reason', 'deadline'),
    ('finish_reason', None), ('budget_terminated', True),
    ('output_tokens', -1), ('output_tokens', True), ('output_tokens', 769),
])
def test_bad_persisted_row_is_rejected(case, field, value):
    row = next(r for r in case[0] if r['split'] == 'test')
    row[field] = value
    with pytest.raises(ValueError):
        validate(case)


def test_altered_messages_are_rejected(case):
    case[0][0]['messages'][-1]['content'] += ' changed'
    with pytest.raises(ValueError, match='messages'):
        validate(case)


@pytest.mark.parametrize('intervention,field,value', [
    ('ablate_d', 'layer', 8), ('add_d', 'layer', None),
    ('ablate_r4', 'layer', 8), ('add_r0', 'layer', True),
    ('baseline', 'layer', 7), ('baseline', 'alpha', 4.),
    ('ablate_d', 'alpha', 1.), ('add_d', 'alpha', 2.),
    ('add_r4', 'alpha', 0.), ('add_d', 'alpha', float('nan')),
])
def test_test_parameters_match_selection(case, intervention, field, value):
    row = next(r for r in case[0] if r['split'] == 'test'
               and r['intervention'] == intervention)
    row[field] = value
    with pytest.raises(ValueError):
        validate(case)


@pytest.mark.parametrize('split,fmt,intervention,layer,alpha', [
    ('test', 'boxed', 'add_d', 7, 4.),
    ('test', 'none', 'ablate_d', 7, 0.),
    ('test', 'boxed', 'ablate_r5', 7, 0.),
    ('test', 'boxed', 'screen_L7', 7, 0.),
    ('dev', 'boxed', 'ablate_d', 7, 0.),
    ('dev', 'none', 'screen_L2', 2, 0.),
    ('dev', 'boxed', 'alpha_m2', 2, 1.),
    ('dev', 'boxed', 'screen_L2', 3, 0.),
    ('dev', 'boxed', 'confirm_L3', 3, 1.),
    ('dev', 'none', 'unknown', 3, 1.),
])
def test_intervention_cells_and_development_metadata(case, split, fmt, intervention, layer, alpha):
    row = next(r for r in case[0] if r['split'] == split and r['format'] == fmt)
    row.update(intervention=intervention, id=f"{row['phase1_id']}:{intervention}",
               layer=layer, alpha=alpha)
    with pytest.raises(ValueError):
        validate(case)


@pytest.mark.parametrize('tokens', [0, 23, 767, 769, 768.])
def test_shortened_length_finish_is_rejected(case, tokens):
    case[0][0].update(finish_reason='length', output_tokens=tokens)
    with pytest.raises(ValueError):
        validate(case)


def test_full_length_and_empty_eos_outputs_are_valid(case):
    case[0][0].update(finish_reason='length', output_tokens=768)
    case[0][1].update(finish_reason='stop', output_tokens=0, budget_terminated=False)
    validate(case)


def test_rendered_hash_is_consistent_across_interventions(case):
    row = next(r for r in case[0] if r['intervention'] == 'ablate_d')
    row['rendered_prompt_sha256'] = 'b'*64
    with pytest.raises(ValueError, match='rendered_prompt_sha256'):
        validate(case)


def test_rendered_hash_is_consistent_across_different_ids_with_same_messages(case):
    rows = [r for r in case[0] if r['split'] == 'test' and r['format'] == 'boxed'
            and r['template_idx'] == 0 and r['intervention'] == 'baseline']
    assert len(rows) == 2 and rows[0]['messages'] == rows[1]['messages']
    case[0][:] = rows
    case[3]['output_rows'] = 2
    rows[1]['rendered_prompt_sha256'] = 'b'*64
    with pytest.raises(ValueError, match='rendered_prompt_sha256'):
        validate(case)


@pytest.mark.parametrize('field,value', [
    ('status', 'budget_stopped'), ('status', 'running'),
    ('run_id', 'other-run'), ('model', 'unregistered'),
    ('revision', 'other-revision'), ('budget_terminated', True),
    ('output_rows', 85),
])
def test_metrics_must_describe_completed_matching_run(case, field, value):
    case[3][field] = value
    with pytest.raises(ValueError):
        validate(case)


@pytest.mark.parametrize('field,value', [
    ('dataset', 'other'), ('config', 'other'), ('split', 'train'),
    ('revision', 'other'), ('sample_seed', 1), ('phase', 'phase1'),
    ('source_sha256', {'prompts/templates.json': 'b'*64}),
    ('sample', []),
])
def test_metrics_manifest_identity_and_hashes_match(case, field, value):
    case[3]['manifest'][field] = value
    with pytest.raises(ValueError):
        validate(case)


def test_optional_hashes_can_be_absent_but_cannot_be_dropped_on_one_side(case):
    del case[3]['manifest']['source_sha256']
    with pytest.raises(ValueError):
        validate(case)
    del case[2]['source_sha256']
    validate(case)


@pytest.mark.parametrize('field', ['layer', 'alpha', 'test_templates'])
def test_metrics_selection_cannot_conflict(case, field):
    case[3]['selection'][field] += 1
    with pytest.raises(ValueError):
        validate(case)


def test_unknown_model_is_rejected_even_if_metrics_agree(case):
    for row in case[0]:
        row['model'] = 'unregistered'
    case[3]['model'] = 'unregistered'
    with pytest.raises(ValueError):
        validate(case)


def test_unpinned_revision_is_rejected_even_if_metrics_agree(case):
    for row in case[0]:
        row['model_revision'] = 'unpinned-revision'
    case[3]['revision'] = 'unpinned-revision'
    with pytest.raises(ValueError):
        validate(case)


@pytest.mark.parametrize('field,value', [('split', 'test'), ('selection_rank', 2)])
def test_development_source_rank_and_split_are_checked(case, field, value):
    case[0][0][field] = value
    with pytest.raises(ValueError):
        validate(case)


@pytest.mark.parametrize('field,value', [('layer', True), ('layer', -1),
                                       ('alpha', float('inf')), ('alpha', True)])
def test_invalid_selection_parameters_raise_value_error(case, field, value):
    case[1][field] = value
    with pytest.raises(ValueError):
        validate(case)


def test_invalid_manifest_sample_raises_value_error(case):
    case[2]['sample'] = case[2]['sample'][:-1]
    case[3]['manifest'] = deepcopy(case[2])
    with pytest.raises(ValueError):
        validate(case)


@pytest.mark.parametrize('field', ['batch_seed', 'layer', 'alpha', 'phase1_id',
                                   'messages', 'rendered_prompt_sha256'])
def test_missing_row_fields_raise_value_error(case, field):
    del case[0][0][field]
    with pytest.raises(ValueError):
        validate(case)


def test_empty_run_is_rejected(case):
    case[0].clear()
    case[3]['output_rows'] = 0
    with pytest.raises(ValueError):
        validate(case)
