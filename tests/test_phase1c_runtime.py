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


def test_budget_stop_summary_never_treats_smoke_or_calibration_as_main_data(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.phase1c import write_budget_stop
    import json
    projection={'total_gpu_usd':11,'launch_full':False,'budget_usd':10}
    write_budget_stop(tmp_path,projection,{'ladders':{l:{'system_prompt':'prompts/system_tutor.txt'} for l in ('instruct','think')}})
    result=json.loads((tmp_path/'summary.json').read_text())
    assert result['main_responses']==0 and result['calibration_data_included'] is False
    assert result['status']=='budget_stop_after_calibration'
    assert (tmp_path/'summary_gold_ge_10.md').exists()


def test_gpu_launch_lock_refuses_a_second_launcher(tmp_path,monkeypatch):
    from src.phase1c import launch_lock
    monkeypatch.chdir(tmp_path)
    with launch_lock():
        with pytest.raises(ValueError,match='launcher'):
            with launch_lock():pass


def test_direct_full_refreshes_approval_after_failed_attempt_spending(tmp_path,monkeypatch):
    import json
    from src import phase1c, modal_app_1c, build_prompts
    monkeypatch.chdir(tmp_path)
    out=tmp_path/'results/phase1c';out.mkdir(parents=True)
    (out/'budget-projection.json').write_text(json.dumps({'launch_full':True,'total_gpu_usd':5.74}))
    metrics=[dict(model=f'allenai/Olmo-3-7B-{l}',requests=156,output_tokens=62400,
                  generation_seconds=40,setup_seconds=120) for l in ('Instruct','Think')]
    monkeypatch.setattr(phase1c,'metrics_for',lambda _:metrics)
    monkeypatch.setattr(phase1c,'accounted_seconds',lambda:10000)
    monkeypatch.setattr(build_prompts,'build_1c',lambda *a,**k: (_ for _ in ()).throw(RuntimeError('spawn reached')))
    with pytest.raises(ValueError,match='Current budget'):
        modal_app_1c.main.info.raw_f(models='all',stage='full',run_id='test-full')
    assert json.loads((out/'budget-projection.json').read_text())['launch_full'] is False


def test_projection_labels_ledger_time_as_accounted_not_pure_function_time():
    from src.phase1c import project_budget
    metrics=[dict(model=f'allenai/Olmo-3-7B-{l}',requests=156,output_tokens=62400,
                  generation_seconds=40,setup_seconds=120) for l in ('Instruct','Think')]
    result=project_budget(metrics,spent_seconds=1000)
    assert result['spent_accounted_seconds']==1000
    assert 'spent_function_seconds' not in result


def test_instruct_projection_uses_batched_reference_and_new_run_cap():
    from src.phase1c import project_budget
    metric=dict(model='allenai/Olmo-3-7B-Instruct',requests=156,
                output_tokens=64524,generation_seconds=41.099,setup_seconds=231.17)
    result=project_budget([metric],spent_seconds=1904,ladders='instruct')
    spec=result['ladder_estimates']['instruct']
    assert spec['output_tokens_per_second']==pytest.approx(3932.697939156243)
    assert spec['mean_output_tokens']==pytest.approx(413.6153846)
    assert set(result['ladder_estimates'])=={'instruct'}
    assert result['budget_usd']==5 and result['spent_accounted_seconds']==0
    assert 2 < result['total_gpu_usd'] < 3 and result['launch_full']
    assert result['historical_accounted_seconds']==1904


def test_live_projection_replaces_reference_with_measured_chunk_runtime():
    from src.phase1c import measured_projection
    progress=dict(completed=2048,requests=10400,output_tokens=850000,
                  generation_seconds=220,setup_seconds=230)
    result=measured_projection(progress,checkpoints=3)
    assert result['total_gpu_seconds']==pytest.approx(3*(230+220*10400/2048)+180)
    assert result['total_gpu_usd']<5
    progress['generation_seconds']=650
    assert measured_projection(progress,checkpoints=3)['total_gpu_usd']>5


def test_ladder_selection_and_future_think_context():
    from src.phase1c import select_ladders
    from src.modal_app import generation_limits
    assert select_ladders('both')==('instruct','think')
    assert select_ladders('instruct')==('instruct',)
    with pytest.raises(ValueError): select_ladders('typo')
    instruct=generation_limits('allenai/Olmo-3-7B-Instruct',{},False)
    think=generation_limits('allenai/Olmo-3-7B-Think',{'max_tokens_think':8192},True)
    assert instruct==(768,8192)
    assert think[0]==8192 and think[1]>=8192+4096


@pytest.mark.parametrize('over_budget', [False, True])
def test_instruct_full_spawns_all_three_before_collect_and_cancels_on_projection(tmp_path, monkeypatch, over_budget):
    import json
    from contextlib import contextmanager
    from src import phase1c, modal_app_1c, build_prompts
    monkeypatch.chdir(tmp_path)
    out=tmp_path/'results/phase1c';out.mkdir(parents=True)
    (out/'calibration.json').write_text(json.dumps({'ladders':{
        'instruct':{'system_prompt':'prompts/system_tutor_strong.txt'}}}))
    (out/'budget-projection.json').write_text(json.dumps({'launch_full':True,'total_gpu_usd':2.3}))
    monkeypatch.setattr(phase1c,'metrics_for',lambda _:[])
    monkeypatch.setattr(phase1c,'accounted_seconds',lambda:0)
    monkeypatch.setattr(phase1c,'project_budget',lambda *a:{'launch_full':True,'total_gpu_usd':2.3})
    prompts=[]
    def build(path,ladder,checkpoint,**kwargs):
        prompts.append((ladder,checkpoint,kwargs['system_path']))
        return [{'id':'test'}],{}
    monkeypatch.setattr(build_prompts,'build_1c',build)
    spawned=[];cancelled=[]
    monkeypatch.setattr(modal_app_1c.time,'sleep',lambda _:None)
    class Call:
        pending=True
        def get(self,**kwargs):
            assert len(spawned)==3
            if self.pending:
                self.pending=False
                raise TimeoutError()
            return {}
        def cancel(self,**kwargs): cancelled.append(self)
    class Function:
        def with_options(self,**kwargs):
            assert kwargs=={'timeout':5400,'retries':0}
            return self
        def spawn(self,model,requests,run_id,manifest,config):
            spawned.append(model)
            assert config['max_tokens_think']==8192
            return Call()
    monkeypatch.setattr(modal_app_1c,'run_model',Function())
    class Progress:
        def items(self):
            return [(spawned[0],dict(completed=2048,requests=10400,output_tokens=850000,
                generation_seconds=650 if over_budget else 220,setup_seconds=230))]
    @contextmanager
    def ephemeral(): yield Progress()
    monkeypatch.setattr(modal_app_1c.modal.Dict,'ephemeral',ephemeral)
    def run():
        modal_app_1c._run_stage(models='all',stage='full',run_id='test-full',
                               ladders='instruct',max_tokens_think=8192)
    if over_budget:
        with pytest.raises(ValueError,match='cancelling'): run()
        assert len(cancelled)==3
    else:
        run()
        assert not cancelled
    assert len(spawned)==3 and all('Think' not in m for m in spawned)
    assert prompts==[('instruct',c,'prompts/system_tutor_strong.txt') for c in ('SFT','DPO','RLVR')]
    ledger=json.loads((out/'gpu-attempts.json').read_text())
    assert {a['status'] for a in ledger}==({'cancelled'} if over_budget else {'completed'})


def test_future_think_selection_projects_only_think_without_running_it():
    from src.phase1c import project_budget
    metric=dict(model='allenai/Olmo-3-7B-Think',requests=156,
                output_tokens=156*2000,generation_seconds=180,setup_seconds=200,
                sampling={'max_tokens':3072})
    result=project_budget([metric],ladders='think',max_tokens_think=8192)
    assert set(result['ladder_estimates'])=={'think'}
    assert result['ladder_estimates']['think']['mean_output_tokens']==pytest.approx(2000*8192/3072)
    assert result['max_tokens_think']==8192


def test_budget_stop_reports_selected_run_cap(tmp_path,monkeypatch):
    import json
    from src.phase1c import write_budget_stop
    monkeypatch.chdir(tmp_path)
    write_budget_stop(tmp_path,{'budget_usd':5,'total_gpu_usd':6},
                      {'ladders':{'instruct':{'system_prompt':'prompts/system_tutor_strong.txt'}}})
    result=json.loads((tmp_path/'summary.json').read_text())
    assert result['budget_usd']==5
    assert 'approved limit: $5.00' in (tmp_path/'summary.md').read_text()
