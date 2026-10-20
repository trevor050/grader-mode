import json
from pathlib import Path
import numpy as np
import pytest
from src.phase2 import (transform_residual, compute_directions, degeneration,
                        build_requests, test_cells as cells, middle_layers, choose_plan)


def test_ablation_removes_only_direction_component_every_position():
    x = np.random.default_rng(4).normal(size=(3, 7, 16))
    d = np.arange(1., 17.); d /= np.linalg.norm(d)
    y = transform_residual(x, d, 'ablate')
    assert np.allclose(y @ d, 0, atol=1e-12)
    assert np.allclose(x-y, (x @ d)[..., None]*d)


def test_addition_shifts_projection_by_alpha_every_position():
    x = np.random.default_rng(4).normal(size=(3, 7, 16))
    d = np.arange(1., 17.); d /= np.linalg.norm(d)
    y = transform_residual(x, d, 'add', 5.)
    assert np.allclose(y @ d - x @ d, 5.)


def test_direction_is_boxed_minus_none_unit_and_projection_norm_scale():
    none = np.array([[[1.,0.],[2.,0.]], [[3.,0.],[4.,0.]]])
    boxed = none + np.array([[0.,2.],[0.,-3.]])
    d, stats = compute_directions(none, boxed)
    assert np.allclose(d, [[0.,1.],[0.,-1.]])
    assert [s['boxed_projection_abs_mean'] for s in stats] == [2.,3.]
    assert [s['difference_norm'] for s in stats] == [2.,3.]
    with pytest.raises(ValueError):
        compute_directions(none, none)


def test_degeneration_empty_repetition_numbers():
    assert degeneration('', '12 and 8')['empty']
    assert degeneration('a b c d '*10, '12 and 8')['repetition']
    assert not degeneration('12 and 8, try adding these.', '12 and 8')['degenerate']
    assert degeneration('12 12 12', '12 and 8')['low_number_count']
    assert degeneration('12 8', '12 and 8')['repetition_rate'] == 0


def test_requests_reuse_phase1_ids_seeds_and_no_cue_templates():
    problems = [{'problem_idx':i,'question':f'There are {i+2} and 3 apples.', 'gold':99} for i in range(200)]
    rows = build_requests(problems)
    assert len(rows) == 4000  # 200 x 5 formats x 4 templates
    first = rows[0]
    assert first['selection_rank'] == 1 and first['split'] == 'dev'
    assert all(r['split']=='test' for r in rows if r['selection_rank']>100)
    from src.build_prompts import request_seed
    assert first['seed'] == request_seed(first['phase1_id'])
    assert len(cells()) == 20
    assert middle_layers(32) == list(range(6,26))


def test_budget_reduces_screen_before_templates_and_never_randoms():
    plan = choose_plan(seconds_per_request=.85, spent_seconds=100, layers=20, reserve_seconds=120)
    assert plan['test_templates'] == 4 and plan['screen_problems'] < 25
    assert plan['projected_usd'] <= 6
    reduced = choose_plan(seconds_per_request=1.35, spent_seconds=100, layers=20, reserve_seconds=120)
    assert reduced['test_templates'] == 2 and reduced['screen_problems'] == 1
    assert reduced['projected_usd'] <= 6
    with pytest.raises(ValueError, match='budget'):
        choose_plan(seconds_per_request=10, spent_seconds=100, layers=20, reserve_seconds=120)


def test_all_phase1_prompt_fields_exact_and_heldout_suffixes():
    from src.build_prompts import expand_requests, ROOT
    problems = [{'problem_idx':i,'question':f'{i+2} plus 3?', 'gold':99} for i in range(200)]
    templates = json.loads((ROOT/'prompts/templates.json').read_text())
    system = (ROOT/'prompts/system_tutor.txt').read_text()
    phase1 = {r['id']:r for r in expand_requests(problems,templates,system)}
    heldout = json.loads((ROOT/'prompts/templates_2.json').read_text())['heldout_formats']
    for r in build_requests(problems):
        if r['format'] in ('none','boxed'):
            old = phase1[r['phase1_id']]
            assert (r['messages'],r['seed'],r['gold']) == (old['messages'],old['seed'],old['gold'])
        else:
            baseid = f"p{r['problem_idx']:04d}:none:none:t{r['template_idx']}:tutor"
            assert r['messages'][-1]['content'] == phase1[baseid]['messages'][-1]['content'] + heldout[r['format']]


def test_phase2_gpu_timeout_single_container_and_no_warm_pool():
    import ast
    tree = ast.parse(Path('src/modal_app_2.py').read_text())
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='study')
    kw = {k.arg:k.value for k in node.decorator_list[0].keywords}
    assert ast.literal_eval(kw['gpu'])=='L40S'
    assert ast.literal_eval(kw['timeout'])<=10800
    for key in ('min_containers','buffer_containers','retries'):
        assert ast.literal_eval(kw[key])==0
    assert ast.literal_eval(kw['max_containers'])==1
    assert ast.literal_eval(kw['single_use_containers']) is True


def test_budget_accounts_batch_floor_and_startup_without_local_controller():
    from src.phase2 import remote_timeout
    plan=choose_plan(.85,100,22,reserve_seconds=120,batch_size=128,batch_floor=.25)
    assert plan['projected_usd']<=6
    assert remote_timeout(237)+237+120+180 <= 6/(1.95/3600)
    with pytest.raises(ValueError): remote_timeout(11000)


def test_runtime_override_keywords_supported_by_pinned_modal():
    import ast,inspect
    from src.modal_app_2 import study
    tree=ast.parse(Path('src/modal_app_2.py').read_text())
    supported=set(inspect.signature(study.with_options).parameters)
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='with_options':
            assert {k.arg for k in node.keywords} <= supported
