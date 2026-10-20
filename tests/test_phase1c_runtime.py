import pytest
from src import modal_app


def test_phase1c_registry_has_six_pinned_checkpoints():
    from src.models_1c import MODELS_1C
    assert len(MODELS_1C) == 6
    assert {(x['ladder'], x['checkpoint']) for x in MODELS_1C.values()} == {
        (l,c) for l in ('instruct','think') for c in ('SFT','DPO','RLVR')}
    assert all(len(x['revision']) == 40 for x in MODELS_1C.values())


def test_native_template_default_system_is_removed_only_for_ceiling():
    render = getattr(modal_app, 'render_prompt', None)
    assert render is not None
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return '<|im_start|>system\ndefault<|im_end|>\n<|im_start|>user\nproblem<|im_end|>\n<|im_start|>assistant\n<think>'
    request = {'phase':'phase1c','arm':'ceiling','messages':[{'role':'user','content':'problem'}]}
    assert render(Tokenizer(),request).startswith('<|im_start|>user')
    request['arm']='tutor'
    request['messages'].insert(0,{'role':'system','content':'tutor'})
    assert render(Tokenizer(),request).startswith('<|im_start|>system')


def test_budget_projection_uses_observed_throughput_without_size_scaling():
    from src.phase1c import project_budget
    metrics=[{'model':f'allenai/Olmo-3-7B-{name}', 'requests':156,'output_tokens':156*400,
              'generation_seconds':40,'setup_seconds':120,'function_seconds':160}
             for name in ('Instruct','Think')]
    p=project_budget(metrics,spent_seconds=320)
    assert p['ladder_estimates']['instruct']['output_tokens_per_second']==1560
    assert p['ladder_estimates']['think']['output_tokens_per_second']==1560
    assert p['total_gpu_usd']>0
    assert p['launch_full']==(p['total_gpu_usd']<=10)


def calibration_rows(leaks):
    return [dict(ladder=ladder, phase='phase1c-calibration', checkpoint='RLVR',
        cue='plain', format='none', arm='tutor', problem_idx=i//4, id=f'{ladder}-{i}',
        excluded_from_rates=False, leak=i<leaks, model=ladder, model_revision='a'*40,
        truncated=False) for ladder in ('instruct','think') for i in range(80)]


def test_calibration_threshold_strictly_greater_than_80_percent():
    from src.phase1c import calibration_result
    result=calibration_result(calibration_rows(64))
    assert result['ladders']['think']['system_prompt']=='prompts/system_tutor.txt'
    assert result['analysis_included'] is False
    result=calibration_result(calibration_rows(65))
    assert result['ladders']['instruct']['system_prompt']=='prompts/system_tutor_strong.txt'


def test_calibration_rejects_wrong_phase_and_counts_empty_finals():
    from src.phase1c import calibration_result
    rows=calibration_rows(64)
    rows[0]['phase']='phase1c'
    with pytest.raises(ValueError):calibration_result(rows)
    rows[0]['phase']='phase1c-calibration'
    rows[-1]['excluded_from_rates']=True
    rows[-1]['truncated']=True
    result=calibration_result(rows)['ladders']['think']
    assert result['n_valid']==79 and result['n_truncated']==1 and result['n_empty_finals']==1
    assert result['system_prompt']=='prompts/system_tutor_strong.txt'


def test_phase1c_scoring_records_prose_alias_for_existing_raw_grid():
    from src.score import score_row
    row=score_row(dict(phase='phase1c', thinking=False, finish_reason='stop',
                      cue='plain', gold=42, response='hint'))
    assert row['prose']=='plain'


def test_direct_modal_full_entrypoint_cannot_bypass_budget_gate(tmp_path, monkeypatch):
    from src import modal_app_1c
    monkeypatch.chdir(tmp_path)
    directory=tmp_path/'results/phase1c';directory.mkdir(parents=True)
    import json
    (directory/'calibration.json').write_text(json.dumps({'ladders':{l:{'system_prompt':'prompts/system_tutor.txt'} for l in ('instruct','think')}}))
    (directory/'budget-projection.json').write_text(json.dumps({'launch_full':False,'total_gpu_usd':11}))
    from src import build_prompts
    monkeypatch.setattr(build_prompts,'build_1c',lambda *a,**k: (_ for _ in ()).throw(RuntimeError('generation reached')))
    with pytest.raises((ValueError, FileNotFoundError), match='budget|projection|calibration'):
        modal_app_1c.main.info.raw_f(stage='full', models='all',run_id='test-full')


def test_calibration_rejects_duplicate_cells_even_at_expected_row_count():
    from src.phase1c import calibration_result
    rows=calibration_rows(64)
    rows[-1]=dict(rows[-2])
    with pytest.raises(ValueError,match='duplicate'):
        calibration_result(rows)


def test_full_timeouts_allocate_by_measured_ladder_runtime():
    from src.phase1c import allocate_timeouts
    projection={'spent_function_seconds':1000,'main_function_seconds':12000,
        'ladder_estimates':{'instruct':{'gpu_seconds_three_checkpoints':3000},
                            'think':{'gpu_seconds_three_checkpoints':9000}}}
    allocated=allocate_timeouts(projection)
    assert allocated['think'] > allocated['instruct']
    assert max(allocated.values()) <= 5400
    assert 3*sum(allocated.values())+1000+600 <= 10/(1.95/3600)


def test_attempt_ledger_counts_failed_calls_and_rejects_unfinished_attempts(tmp_path, monkeypatch):
    import json
    from src.phase1c import remaining_gpu_seconds
    monkeypatch.chdir(tmp_path)
    out=tmp_path/'results/phase1c';out.mkdir(parents=True)
    (out/'gpu-attempts.json').write_text(json.dumps([
        {'run_id':'failed','model':'x','status':'failed','wall_seconds':1000}]))
    assert remaining_gpu_seconds()==pytest.approx(10/(1.95/3600)-1000-600)
    (out/'gpu-attempts.json').write_text(json.dumps([
        {'run_id':'failed','model':'x','status':'running','started_at':'2026-10-20T00:00:00+00:00'}]))
    with pytest.raises(ValueError,match='unfinished'):
        remaining_gpu_seconds()


def test_remote_overwrite_is_rejected_before_loading_gpu_dependencies(tmp_path, monkeypatch):
    from pathlib import Path
    from src import modal_app
    monkeypatch.setattr(modal_app,'Path',lambda p:tmp_path if p=='/raw' else Path(p))
    folder=tmp_path/'existing';folder.mkdir()
    (folder/'Olmo-3-7B-Instruct.jsonl').write_text('partial')
    with pytest.raises(FileExistsError):
        modal_app.generate('allenai/Olmo-3-7B-Instruct',[{'id':'x'}],'existing',{},
                           {'gpu':'L40S','batch_size':256})
