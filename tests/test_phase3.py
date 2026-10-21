import numpy as np
import pytest
from src.phase3 import (screen_statistics, choose_neurons, build_requests, score_generation,
                        summarize_trajectory, choose_test_templates, remote_timeout, RATE,
                        CAP_SECONDS, push_activations, reasoning_span)


def test_screen_is_prompt_level_unbiased_pooled_d_and_layer_matched_controls():
    tutor=np.array([[[1.,3.,2.],[0.,5.,1.]],[[3.,5.,6.],[2.,7.,5.]],[[5.,7.,4.],[4.,9.,3.]]])
    tutor=np.concatenate([tutor,tutor+1],axis=2)
    neutral=tutor-np.array([[1.,4.,2.,.1,.1,.1],[1.,2.,1.,.1,.1,.1]])
    stats=screen_statistics(tutor.sum(0), (tutor*tutor).sum(0), neutral.sum(0), (neutral*neutral).sum(0),3)
    assert np.allclose(stats['tutor_sd'],tutor.std(0,ddof=1))
    assert np.allclose(stats['d'],(tutor.mean(0)-neutral.mean(0))/tutor.std(0,ddof=1))
    chosen=choose_neurons(stats,k=2,seed=0)
    assert [n['id'] for n in chosen['tutor']]==['0.1','0.2']
    assert len(chosen['controls'])==5
    assert all([n['layer'] for n in group]==[0,0] for group in chosen['controls'])
    assert chosen==choose_neurons(stats,k=2,seed=0)
    for group in chosen['controls']:
        assert len({n['id'] for n in group})==2
    with pytest.raises(ValueError):screen_statistics(tutor.sum(0),tutor.sum(0),neutral.sum(0),neutral.sum(0),1)


def test_push_changes_selected_generated_positions_only_with_rank_magnitudes():
    x=np.zeros((2,4,5),dtype=np.float32)
    y=push_activations(x,[1,3],[2.,-3.],generated=False)
    assert np.array_equal(y,x)
    y=push_activations(x,[1,3],[2.,-3.],generated=True)
    assert np.all(y[:,:,1]==2) and np.all(y[:,:,3]==-3)
    assert np.all(y[:,:,[0,2,4]]==0) and np.all(x==0)


def test_request_identity_is_phase1b_seed_and_selection_order():
    sample=[{'problem_idx':i+400,'question':f'{i} plus 4?', 'gold':42} for i in range(200)]
    rows=build_requests(sample)
    assert len(rows)==800 and rows[0]['selection_rank']==1
    assert rows[400]['split']=='test' and rows[400]['selection_rank']==101
    from src.build_prompts import request_seed
    assert all(r['thinking'] and r['cue']=='none' and r['format']=='boxed' and r['seed']==request_seed(r['phase1b_id']) for r in rows)


def test_only_final_scored_and_degeneration_is_exact():
    r=score_generation('<think>42 tutor</think>Try adding.',42,'stop')
    assert not r['leak'] and not r['degenerate'] and r['tutor_mention']
    r=score_generation('<think>42',42,'length')
    assert r['empty'] and r['truncation_without_final'] and r['excluded_from_rates'] and not r['leak']
    assert score_generation('<think>x</think>'+'a b c d '*20,42,'stop')['repetition']
    assert reasoning_span('<think>hello</think>bye')==(7,12)
    assert reasoning_span('hello</think>bye')==(0,5)
    assert reasoning_span('<think>open') is None


def test_trajectory_linear_slope_first_quarter_deciles():
    t=np.linspace(2,-2,40)[:,None]*np.arange(1,7)[None,:]
    out=summarize_trajectory(t)
    assert np.allclose(out['slopes'],-4*np.arange(1,7))
    assert np.allclose(out['first_quarter'],t[:10].mean(0))
    assert np.allclose(out['deciles'][0],t[:4].mean(0))
    assert summarize_trajectory(np.zeros((1,6)))['slopes'] is None


def test_budget_only_reduces_test_before_generation_and_timeout_fits_cap():
    assert choose_test_templates(1,100)==2
    assert choose_test_templates(6,6000)==1
    with pytest.raises(ValueError,match='budget'):choose_test_templates(100,0)
    assert remote_timeout(200)+200+120+180<=CAP_SECONDS
    assert CAP_SECONDS*RATE==pytest.approx(7.)


def test_input_loader_rejects_missing_labels_or_altered_prompts():
    from src.phase3 import validate_tracking_sources
    from src.models import MODELS
    sample=[{'problem_idx':i,'question':f'{i} plus 4?', 'gold':42} for i in range(200)]
    requests=build_requests(sample)
    raw=[{**r,'phase':'phase1b','model':'Qwen/Qwen3-14B','model_revision':MODELS['Qwen/Qwen3-14B'],
          'response':'<think>try</think>42','finish_reason':'stop','rendered_prompt_sha256':'a'*64} for r in requests]
    scored=[{**r,**score_generation(r['response'],r['gold'],r['finish_reason'])} for r in raw]
    out=validate_tracking_sources(requests,raw,scored)
    assert len(out)==400 and out[0]['selection_rank']==101
    with pytest.raises(ValueError,match='label'):validate_tracking_sources(requests,raw,scored[:-1])
    raw[400]['messages']=[{'role':'user','content':'changed'}]
    with pytest.raises(ValueError,match='messages'):validate_tracking_sources(requests,raw,scored)


def test_modal_runner_has_one_single_use_gpu_no_retries_or_warm_pool():
    import ast
    from pathlib import Path
    tree=ast.parse(Path('src/modal_app_3.py').read_text())
    study=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='study')
    kw={k.arg:ast.literal_eval(k.value) for k in study.decorator_list[0].keywords if k.arg in
        ('gpu','timeout','startup_timeout','retries','single_use_containers','min_containers','buffer_containers','max_containers')}
    assert kw['gpu']=='L40S' and kw['single_use_containers']
    assert kw['max_containers']==1 and kw['timeout']<=CAP_SECONDS
    assert kw['min_containers']==kw['buffer_containers']==kw['retries']==0


def test_screen_ties_positive_selection_and_invalid_variance_controls():
    base=np.array([[1.,2.,3.,4.]])
    stat=screen_statistics(base*3,(base*base+1)*3,(base-1)*3,((base-1)**2+1)*3,3)
    chosen=choose_neurons(stat,k=1)
    assert chosen['tutor'][0]['id']=='0.0'
    assert all(c[0]['id']!='0.0' for c in chosen['controls'])
    zero=screen_statistics(base*3,base*base*3,base*3,base*base*3,3)
    with pytest.raises(ValueError,match='positive'):choose_neurons(zero,k=1)


def test_uncertain_submission_reserves_entire_remaining_exposure():
    from src.phase3 import close_attempt
    unknown=close_attempt(call_known=False,elapsed=3.,previous_seconds=900.,status='running')
    assert unknown['status']=='submission_unresolved'
    assert unknown['wall_seconds']==CAP_SECONDS-900
    assert unknown['observed_wall_seconds']==3.
    completed=close_attempt(call_known=True,elapsed=70.,previous_seconds=900.,status='completed')
    assert completed['status']=='completed' and completed['wall_seconds']==70


def test_system_boundary_ignores_left_padding_that_uses_turn_end_token():
    from src.phase3 import system_end_boundary
    ids=np.array([[9,9,1,2,9,3,4,9],[1,2,8,2,9,3,4,9]])
    mask=np.array([[0,0,1,1,1,1,1,1],[1,1,1,1,1,1,1,1]])
    assert np.array_equal(system_end_boundary(ids,mask,9),[5,5])
    with pytest.raises(ValueError):system_end_boundary(np.array([[9,1,2]]),np.array([[0,1,1]]),9)
