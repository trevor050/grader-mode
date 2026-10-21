"""CPU-only contracts. No Modal hydration, image build, GPU or paid calls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
from src.models_5 import NEMOTRON, GLM
from src.vast_generate_5 import expand_requests


def module():
    from src import modal_second_family_5
    return modal_second_family_5


def grid(model=NEMOTRON):
    return expand_requests([dict(problem_idx=i, question=f'Question {i}?', gold=i+10)
                            for i in range(200)], model)


def measurement(seconds=100, count=256):
    m = module()
    return dict(requests=count, output_tokens=count*1000, generation_seconds=seconds-1,
                wall_seconds=seconds, setup_seconds=0, batch_size=256, chunk_size=256,
                per_cell={c:dict(count=count//8, output_tokens=count//8*1000)
                          for c in m.cell_ids()})


def test_module_exists():
    assert importlib.util.find_spec('src.modal_second_family_5') is not None


def test_dry_is_default_and_inert(monkeypatch, capsys):
    m = module()
    monkeypatch.setattr(m, 'execute', lambda *a: pytest.fail('dry launched'))
    m.main([])
    plan = json.loads(capsys.readouterr().out)
    assert plan['cap_usd'] == 2.60 and plan['reserve_usd'] == .30
    assert plan['gpu'] == 'RTX-PRO-6000'
    assert list(plan['stages'].values()) == [8, 256, 800, 1600, 800]
    assert plan['dropped_cells'] == []
    assert m.RATE >= .000842 + 4*.0000131 + 32*.00000222
    with pytest.raises(SystemExit): m.main(['--execute'])
    with pytest.raises(SystemExit): m.main(['--dry-run','--execute','--run-id','x'])


def test_full_grid_and_pilot_never_mutate_or_shorten():
    m = module()
    rows = grid(); original = deepcopy(rows)
    stages = m.stage_rows(rows)
    assert [len(v) for v in stages.values()] == [800,1600,800]
    assert rows == original
    assert {r['id'] for v in stages.values() for r in v} == {r['id'] for r in rows}
    assert len(m.balanced_pilot(rows,8)) == 8
    assert len(m.balanced_pilot(rows,256)) == 256
    with pytest.raises(ValueError): m.stage_rows(rows[:-1])


def test_gate_defers_whole_remaining_stages_with_shortfall():
    m = module(); stages = m.stage_rows(grid())
    fast = m.project_stage(stages['h20_core'], measurement(), [],
                           charged_usd=.2, elapsed_seconds=100, seconds_left=1500)
    assert fast['decision'] == 'run' and fast['count'] == 800
    slow = m.project_stage(stages['h21_main'], measurement(1000), [],
                           charged_usd=.2, elapsed_seconds=100, seconds_left=1500)
    assert slow['decision'] == 'defer' and slow['shortfall_usd'] > 0
    assert slow['outcomes_consulted'] is False and slow['dropped_cells'] == []
    with pytest.raises(ValueError):
        m.project_stage(stages['h20_core'],measurement(count=8),[],charged_usd=0,
                        elapsed_seconds=0,seconds_left=2000)


def test_observed_stage_wall_can_only_make_later_gate_more_conservative():
    m = module(); rows = m.stage_rows(grid())['h21_main']; base = measurement()
    observed = dict(requests=800,output_tokens=1600000,wall_seconds=1600,
                    generation_seconds=1500,per_cell={'prefill/empty':dict(count=400,output_tokens=800000),
                    'prefill/reminder':dict(count=400,output_tokens=800000)})
    a = m.project_stage(rows,base,[],charged_usd=0,elapsed_seconds=0,seconds_left=2200)
    b = m.project_stage(rows,base,[observed],charged_usd=0,elapsed_seconds=0,seconds_left=2200)
    assert b['projected_seconds'] > a['projected_seconds']
    poisoned = {**base,'final_reply':'bad','violation':True,'accuracy':0}
    assert m.project_stage(rows,poisoned,[],charged_usd=0,elapsed_seconds=0,seconds_left=2200) == a


def test_authorization_all_attempts_setup_teardown_and_restart(tmp_path):
    m = module(); p=tmp_path/'authorization.json'; ledger=m.Authorization(p)
    a=ledger.begin('one',now=100)
    assert a['work_deadline_unix'] < a['hard_deadline_unix']
    with pytest.raises(m.BudgetError): ledger.begin('concurrent',now=101)
    ledger.bind('one','ap-one')
    ledger.finish('one',True,now=200)
    assert ledger.snapshot()['attempts'][0]['charged_usd'] == pytest.approx(.1)
    b=m.Authorization(p).begin('two',now=300)
    assert b['hard_deadline_unix']-300 < a['hard_deadline_unix']-100
    m.Authorization(p).finish('two',False,now=310)
    with pytest.raises(m.BudgetError): m.Authorization(p).begin('three',now=400)
    assert ledger.snapshot()['cap_usd'] == 2.6
    assert not (tmp_path/'aggregate-ledger.json').exists()


def test_budget_expiry_keeps_reserve_and_refuses_reuse(tmp_path):
    m=module(); ledger=m.Authorization(tmp_path/'a.json'); a=ledger.begin('one',now=100)
    ledger.finish('one',True,now=a['hard_deadline_unix'])
    assert sum(t['charged_usd'] for t in ledger.snapshot()['attempts']) <= 2.3
    with pytest.raises(m.BudgetError): ledger.begin('two',now=5000)
    with pytest.raises(m.BudgetError): ledger.begin('one',now=5000)


@pytest.mark.parametrize('phase,text,expected',[
    ('infrastructure','RuntimeError: model unavailable','infrastructure_failure'),
    ('runtime_smoke','ConnectionError: download timed out','infrastructure_failure'),
    ('model_load','CUDA driver error: device unavailable','infrastructure_failure'),
    ('model_load','ValueError: unsupported quantization modelopt','model_failure'),
    ('runtime_smoke','RuntimeError: smoke failed native generation protocol','model_failure'),
    ('h20_core','RuntimeError: out of memory','execution_failure'),
    ('runtime_smoke','TimeoutError: absolute work deadline','budget_stop'),
])
def test_failure_semantics(phase,text,expected):
    assert module().failure_kind(phase,text) == expected


def test_fallback_only_after_two_real_model_failures():
    m=module(); calls=[]; proofs=[]
    def attempt(model,stage,fix,proof):
        calls.append((model,stage,fix,proof))
        if model==NEMOTRON:
            return dict(status='model_failure',phase='runtime_smoke',stage=stage,
                        started_unix=len(calls)*10,ended_unix=len(calls)*10+1,
                        attempt_id=stage,evidence=stage+'.json',violation_scored=False,
                        runtime_fix=dict(resident_sequences=128,enforce_eager=False))
        return dict(status='completed')
    assert m.run_families(attempt,proofs.append,clock=lambda:30)['status']=='completed'
    assert [c[:2] for c in calls] == [(NEMOTRON,'startup'),(NEMOTRON,'fix'),(GLM,'fallback')]
    assert len(proofs)==1 and proofs[0]['gpu']=='RTX-PRO-6000'
    calls.clear(); proofs.clear()
    result=m.run_families(lambda *a: calls.append(a) or dict(status='infrastructure_failure'),proofs.append)
    assert result['status']=='infrastructure_failure' and len(calls)==1 and proofs==[]


def test_main_publication_excludes_pilots_and_incomplete_stages(tmp_path):
    m=module(); rows=grid(); stages=m.stage_rows(rows); slug=NEMOTRON.split('/')[-1]
    for name,selected in [('smoke',rows[:8]),('throughput',rows[:256]),*stages.items()]:
        d=tmp_path/name;d.mkdir()
        with (d/(slug+'.jsonl')).open('w') as f:
            for r in selected: f.write(json.dumps({**r,'run_id':'shared-main','stage':
                'pilot' if name in ('smoke','throughput') else 'main','pilot':name in ('smoke','throughput')})+'\n')
    result=m.publish_main(tmp_path,rows,{'sample':[]},['h20_core'],['h21_main','h20_controls'])
    main=[json.loads(s) for s in Path(result['jsonl']).read_text().splitlines()]
    assert len(main)==800 and all(r['stage']=='main' and not r['pilot'] for r in main)
    manifest=json.loads(Path(result['manifest']).read_text())
    assert manifest['requests']==3200 and manifest['dropped_cells']==[]
    assert manifest['deferred_stages']==['h21_main','h20_controls']
    assert manifest['inference_eligible'] is False


def payload_for(tmp_path, m, rows=None):
    rows=grid() if rows is None else rows
    return dict(directory=str(tmp_path),requests=rows,manifest=dict(sample=[],implementation_sha256={}),
                model=NEMOTRON,attempt_stage='startup',attempt_id='unit-startup',
                source_hashes={},work_deadline_unix=10**12,resident_sequences=256,
                enforce_eager=False,prior_charged_usd=0,started_unix=1)


def test_dict_worker_rejects_shortened_permit_and_preserves_smoke(tmp_path,monkeypatch):
    m=module();p=payload_for(tmp_path,m);channel={'command:throughput':dict(
        action='run',attempt_id=p['attempt_id'],count=255,work_deadline_unix=p['work_deadline_unix'],
        stage_deadline_unix=p['work_deadline_unix'])};generated=[];commits=[]
    def fake_generate(model,selected,run_id,manifest,config):
        generated.append(len(selected));directory=Path(config['output_dir']);directory.mkdir()
        return dict(requests=len(selected),output_tokens=len(selected)*100,generation_seconds=.01,
                    setup_seconds=0,per_cell={model.split('/')[-1]+'/'+m.cell_id(r):dict(count=1,output_tokens=100)
                                             for r in selected})
    monkeypatch.setattr(m,'validate_runtime_smoke',lambda *a:dict(protocol_checked=True))
    state=m.worker_session(p,channel,checkpoint=lambda:commits.append('result'),
                           cache_commit=lambda:commits.append('cache'),generate_fn=fake_generate)
    assert state['status']=='budget_stop'
    assert generated==[8] and 'done:throughput' not in channel
    assert (tmp_path/'smoke.done.json').exists() and 'cache' in commits
    assert json.loads((tmp_path/'main/manifest.json').read_text())['completed_rows']==0


def test_cached_real_generator_full_sequence_and_analyzer_schema(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from src import vast_generate_5 as generator
    from test_phase5_formats import Tokenizer
    from test_phase5_requests import RenderTokenizer
    m=module();p=payload_for(tmp_path,m);rows=p['requests'];models=[];batches=[];commits=[]
    tok=Tokenizer(NEMOTRON);tok.apply_chat_template=RenderTokenizer(NEMOTRON).apply_chat_template
    class LLM:
        def __init__(self,**options):
            models.append(options)
            self.llm_engine=SimpleNamespace(model_config=SimpleNamespace(quantization='modelopt_mixed'))
        def generate(self,prompts,params,**kwargs):
            batches.append(len(prompts))
            result=[]
            for prompt,param in zip(prompts,params):
                assert param.max_tokens==4096 and param.top_k==-1 and param.temperature==.7
                rendered=tok.decode(prompt['prompt_token_ids'])
                response=('' if '<think></think>' in rendered else 'work</think>')+'Try a step.<|im_end|>'
                result.append(SimpleNamespace(prompt_token_ids=prompt['prompt_token_ids'],outputs=[
                    SimpleNamespace(token_ids=tok.encode(response),finish_reason='stop',stop_reason=11)]))
            return result
    vllm=SimpleNamespace(LLM=LLM,SamplingParams=lambda **kw:SimpleNamespace(**kw))
    monkeypatch.setitem(sys.modules,'vllm',vllm)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(
        get_device_name=lambda i:'NVIDIA RTX PRO 6000 Blackwell Server Edition',
        get_device_capability=lambda i:(12,0))))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *a,**k:tok)))
    monkeypatch.setattr(generator,'runtime_versions',lambda:dict(generator.PACKAGE_VERSIONS))
    monkeypatch.setattr(generator,'verify_tokenizer',lambda *a:None)
    channel={}
    for name,count in [('throughput',256),('h20_core',800),('h21_main',1600),('h20_controls',800)]:
        channel['command:'+name]=dict(action='run',count=count,attempt_id=p['attempt_id'],
            work_deadline_unix=p['work_deadline_unix'],stage_deadline_unix=p['work_deadline_unix'])
    state=m.worker_session(p,channel,checkpoint=lambda:commits.append('results'),
                           cache_commit=lambda:commits.append('cache'))
    assert state['status']=='completed', (tmp_path/'runtime-trace.txt').read_text() if (tmp_path/'runtime-trace.txt').exists() else state
    assert len(models)==1 and vllm.LLM is LLM
    assert models[0]['quantization'] is None and models[0]['max_num_seqs']==256
    assert models[0]['enforce_eager'] is False
    assert batches==[8,256]+[256,256,256,32]+[256]*6+[64]+[256,256,256,32]
    raw=[json.loads(s) for s in (tmp_path/'main/generations.jsonl').read_text().splitlines()]
    assert len(raw)==3200 and {r['id'] for r in raw}=={r['id'] for r in rows}
    assert {r['run_id'] for r in raw}=={'unit-startup-main'}
    assert all(r['stage']=='main' and not r['pilot'] for r in raw)
    assert all(r['model_revision']==models[0]['revision'] for r in raw)
    assert all(r['gpu']=='RTX-PRO-6000' for r in raw)
    assert len(commits)>15 and 'cache' in commits
    # Feed actual generated schemas to the existing grid validator, without
    # scoring any outcomes. Replace only the external dataset with our fixture.
    from src import analyze_5
    from src.score_5 import normalize_row
    from src.models_5 import REGISTRATION_COMMIT
    identities={r['problem_idx']:(r['gold'],r['question']) for r in rows}
    monkeypatch.setattr(analyze_5,'_original_sample',lambda:identities)
    manifest=json.loads((tmp_path/'main/manifest.json').read_text())
    manifest.update(registration_commit=REGISTRATION_COMMIT,
                    sample=[dict(problem_idx=i,gold=g,question=q) for i,(g,q) in identities.items()])
    report=analyze_5.validate_grid([normalize_row(r) for r in raw],
                                 problem_manifest=manifest,model_scope=NEMOTRON)
    assert report['n_problems']==200 and report['dropped_cells']==[]


def test_control_defers_suffix_without_launching_any_part(monkeypatch):
    m=module();p=payload_for(Path('/unused'),m);records=[];channel={};waits=[]
    monkeypatch.setattr(m.time,'time',lambda:100)
    p.update(started_unix=1,work_deadline_unix=5000)
    def done(channel,call,name,deadline):
        waits.append(name)
        return measurement(1000) if name=='throughput' else measurement(1,count=8)
    monkeypatch.setattr(m,'wait_done',done)
    m.control_stages(p,channel,None,records.append)
    assert waits==['smoke','throughput'] and channel['stop'] is True
    assert records[-1]['deferred_stages']==list(m.STAGES)
    assert records[-1]['shortfall_usd']>0
    assert 'command:h20_core' not in channel


def test_overrun_never_silently_caps_real_exposure(tmp_path):
    m=module();ledger=m.Authorization(tmp_path/'a.json');a=ledger.begin('one',now=100)
    result=ledger.finish('one',True,now=a['hard_deadline_unix']+30)
    assert result['charged_usd'] > a['reserved_usd']
    assert result['status']=='unresolved' and result['overrun_usd']>0
    with pytest.raises(m.BudgetError):ledger.begin('two',now=5000)


def test_supervisor_kills_process_group_and_commits_on_deadline(tmp_path,monkeypatch):
    from types import SimpleNamespace
    m=module();p=payload_for(tmp_path,m);p['work_deadline_unix']=100
    channel={};killed=[];commits=[];waits=[]
    monkeypatch.setattr(m,'results',SimpleNamespace(commit=lambda:commits.append(True)))
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(m.time,'time',lambda:100)
    monkeypatch.setattr(m,'Path',lambda value:tmp_path if value=='/phase5' else Path(value))
    class Process:
        pid=123
        def poll(self):return None
        def wait(self,timeout):waits.append(timeout);return -9
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**k:Process())
    monkeypatch.setattr(m.os,'killpg',lambda pid,sig:killed.append((pid,sig)))
    class Timer:
        def __init__(self,*a):pass
        def start(self):pass
        def cancel(self):pass
    monkeypatch.setattr(m.threading,'Timer',Timer)
    class Channel(dict):object_id='di-fake'
    c=Channel()
    state=m.second_family_gpu._raw_f_(p,c)
    assert state['status']=='budget_stop' and killed==[(123,m.signal.SIGKILL)]
    assert waits==[8] and len(commits)>=2
    assert json.loads((tmp_path/p['attempt_id']/'remote-execution.json').read_text())['status']=='budget_stop'


def test_watchdog_requires_stopped_app_and_explicit_empty_container_verification(tmp_path,monkeypatch):
    m=module();path=tmp_path/'a.json';ledger=m.Authorization(path)
    a=ledger.begin('unit',now=100);ledger.bind('unit','ap-owned')
    calls=[]
    monkeypatch.setattr(m,'Authorization',lambda:ledger)
    monkeypatch.setattr(m.time,'time',lambda:a['work_deadline_unix'])
    monkeypatch.setattr(m,'stop_and_verify',lambda app_id:calls.append(app_id) or (True,dict(
        app_stop_confirmed=True,containers_after=[])))
    m.watchdog('unit')
    assert calls==['ap-owned'] and ledger.snapshot()['attempts'][0]['teardown_verified']


def test_watchdog_unbound_unknown_app_never_claims_verified(tmp_path,monkeypatch):
    m=module();ledger=m.Authorization(tmp_path/'a.json');a=ledger.begin('unit',now=100)
    monkeypatch.setattr(m,'Authorization',lambda:ledger)
    monkeypatch.setattr(m.time,'time',lambda:a['hard_deadline_unix'])
    monkeypatch.setattr(m,'modal_cli',lambda *a,**k:'[]')
    m.watchdog('unit')
    value=ledger.snapshot()['attempts'][0]
    assert value['status']=='unresolved' and not value['teardown_verified']
    assert value['charged_usd']==value['reserved_usd']


@pytest.mark.parametrize('verified',[True,False])
def test_execute_always_cancel_stop_verify_then_pull_even_if_cancel_fails(tmp_path,monkeypatch,verified):
    from contextlib import nullcontext
    from types import SimpleNamespace
    m=module();ledger=m.Authorization(tmp_path/'authorization.json');events=[]
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(m,'Authorization',lambda:ledger)
    monkeypatch.setattr(m,'source_hashes',lambda:{'fixed':'hash'})
    from src import vast_generate_5 as generator
    monkeypatch.setattr(generator,'build',lambda model,n:(grid({'nemotron':NEMOTRON,'glm':GLM}[model]),{'sample':[]}))
    monkeypatch.setattr(m,'app',SimpleNamespace(app_id='ap-owned',run=lambda **kw:nullcontext()))
    monkeypatch.setattr(m.modal,'enable_output',lambda:nullcontext())
    monkeypatch.setattr(m.modal.Dict,'ephemeral',lambda:nullcontext({}))
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**k:SimpleNamespace(pid=123))
    class Call:
        object_id='fc-owned'
        def get(self,timeout):return dict(status='deferred',completed_stages=[],deferred_stages=list(m.STAGES))
        def cancel(self,**kw):
            assert kw==dict(terminate_containers=True)
            events.append('cancel');raise RuntimeError('simulated cancel failure')
    monkeypatch.setattr(m,'second_family_gpu',SimpleNamespace(spawn=lambda *a:Call()))
    monkeypatch.setattr(m,'control_stages',lambda *a:None)
    def stop(app_id):
        assert app_id=='ap-owned';events.append('stop')
        return verified,dict(app_stop_confirmed=verified,containers_after=[] if verified else ['live'])
    monkeypatch.setattr(m,'stop_and_verify',stop)
    def cli(*args,**kwargs):
        if args[0]=='volume':events.append('pull');return ''
        return '[]'
    monkeypatch.setattr(m,'modal_cli',cli)
    if verified:
        result=m.execute('unit')
        assert result['zero_containers_verified'] and result['status']=='deferred'
        assert ledger.snapshot()['attempts'][0]['status']=='settled'
    else:
        with pytest.raises(m.BudgetError,match='unresolved'):m.execute('unit')
        assert ledger.snapshot()['attempts'][0]['status']=='unresolved'
    assert events.index('cancel')<events.index('stop')<events.index('pull')
    assert not (tmp_path/'results/phase5/aggregate-ledger.json').exists()


@pytest.mark.parametrize('sdk_retains_id',[True,False])
def test_image_setup_failure_recovers_owned_app_before_settling(tmp_path,monkeypatch,sdk_retains_id):
    from contextlib import nullcontext
    from types import SimpleNamespace
    m=module();ledger=m.Authorization(tmp_path/'authorization.json');events=[]
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(m,'Authorization',lambda:ledger)
    monkeypatch.setattr(m,'source_hashes',lambda:{'fixed':'hash'})
    from src import vast_generate_5 as generator
    monkeypatch.setattr(generator,'build',lambda model,n:(grid(model),{'sample':[]}))
    app=SimpleNamespace(app_id=None)
    class BrokenSetup:
        def __enter__(self):
            events.append('setup-entered')
            app.app_id='ap-new' if sdk_retains_id else None
            raise RuntimeError('image setup failed')
        def __exit__(self,*args):pass
    app.run=lambda **kw:BrokenSetup()
    monkeypatch.setattr(m,'app',app)
    monkeypatch.setattr(m.modal,'enable_output',lambda:nullcontext())
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**k:SimpleNamespace(pid=123))
    def cli(*args,**kwargs):
        if args[:2]==('app','list'):
            if not events:events.append('initial-list');return '[]'
            return json.dumps([dict(app_id='ap-new',description=m.APP_NAME,state='initializing')])
        if args[0]=='volume':events.append('pull')
        return '[]'
    monkeypatch.setattr(m,'modal_cli',cli)
    monkeypatch.setattr(m,'stop_and_verify',lambda app_id:events.append(('stop',app_id)) or
        (True,dict(app_stop_confirmed=True,containers_after=[])))
    result=m.execute('setup-error')
    ticket=ledger.snapshot()['attempts'][0]
    assert ticket['app_id']=='ap-new' and ticket['teardown_verified']
    assert ticket['charged_usd']<ticket['reserved_usd']
    assert result['status']=='infrastructure_failure'
    assert 'setup-entered' in events and 'image setup failed' in result['controller_error']
    assert events.index(('stop','ap-new'))<events.index('pull')


@pytest.mark.parametrize('discover_new',[True,False])
def test_failed_create_never_releases_budget_using_stale_sdk_app(tmp_path,monkeypatch,discover_new):
    from contextlib import nullcontext
    from types import SimpleNamespace
    m=module();ledger=m.Authorization(tmp_path/'authorization.json');stops=[];listed=[False]
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(m,'Authorization',lambda:ledger)
    monkeypatch.setattr(m,'source_hashes',lambda:{'fixed':'hash'})
    from src import vast_generate_5 as generator
    monkeypatch.setattr(generator,'build',lambda model,n:(grid(model),{'sample':[]}))
    class LostCreate:
        def __enter__(self):raise RuntimeError('AppCreate acknowledgement lost')
        def __exit__(self,*args):pass
    monkeypatch.setattr(m,'app',SimpleNamespace(app_id='ap-old',run=lambda **kw:LostCreate()))
    monkeypatch.setattr(m.modal,'enable_output',lambda:nullcontext())
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**k:SimpleNamespace(pid=123))
    def cli(*args,**kwargs):
        if args[:2]==('app','list'):
            rows=[dict(app_id='ap-old',description=m.APP_NAME,state='stopped')]
            if listed[0] and discover_new:
                rows.append(dict(app_id='ap-new',description=m.APP_NAME,state='initializing'))
            listed[0]=True
            return json.dumps(rows)
        return '[]'
    monkeypatch.setattr(m,'modal_cli',cli)
    monkeypatch.setattr(m,'stop_and_verify',lambda app_id:stops.append(app_id) or
        (True,dict(app_stop_confirmed=True,containers_after=[])))
    if discover_new:
        m.execute('lost-create')
        assert stops==['ap-new']
    else:
        with pytest.raises(m.BudgetError,match='unresolved'):m.execute('lost-create')
        a=ledger.snapshot()['attempts'][0]
        assert a['charged_usd']==a['reserved_usd'] and not a['teardown_verified']
        assert stops==[]


def test_model_failure_limit_survives_controller_restart():
    m=module();calls=[];proofs=[]
    prior=dict(status='model_failure',phase='model_load',stage='startup',attempt_id='earlier-startup',
               started_unix=10,ended_unix=11,evidence='earlier.json',violation_scored=False,
               runtime_fix=dict(resident_sequences=128,enforce_eager=False))
    def attempt(model,stage,fix,proof):
        calls.append((model,stage,fix))
        return dict(status='infrastructure_failure')
    m.run_families(attempt,proofs.append,prior_failures=[prior])
    assert calls==[(NEMOTRON,'fix',prior['runtime_fix'])] and proofs==[]
    second={**prior,'stage':'fix','attempt_id':'earlier-fix','started_unix':20,'ended_unix':21}
    calls.clear()
    m.run_families(attempt,proofs.append,clock=lambda:30,prior_failures=[prior,second])
    assert calls==[(GLM,'fallback',None)] and len(proofs)==1
    assert proofs[0]['decision_unix']==30 and proofs[0]['gpu']=='RTX-PRO-6000'
    with pytest.raises(m.BudgetError):
        m.run_families(attempt,proofs.append,prior_failures=[prior,second,second])


def test_corrupt_authorization_cannot_raise_cap_or_reset_charge(tmp_path):
    m=module();ledger=m.Authorization(tmp_path/'a.json');ledger.begin('one',now=100)
    value=json.loads(ledger.path.read_text());value['cap_usd']=3.1
    ledger.path.write_text(json.dumps(value))
    with pytest.raises(m.BudgetError,match='refuse reset'):ledger.begin('two',now=200)


def test_resume_proof_still_requires_chronological_real_model_failures():
    m=module();failure=dict(status='model_failure',phase='infrastructure',stage='startup',
                           attempt_id='old',started_unix=10,ended_unix=11,evidence='x',violation_scored=False)
    with pytest.raises(ValueError):
        m.run_families(lambda *a:pytest.fail('launched'),lambda p:None,prior_failures=[failure])


def test_later_gate_includes_commit_publication_and_dict_wall_time(monkeypatch):
    m=module();p=payload_for(Path('/unused'),m);p.update(started_unix=1,work_deadline_unix=5000)
    channel={};records=[];ticks=iter([0,100,100,1700])
    monkeypatch.setattr(m.time,'time',lambda:100)
    monkeypatch.setattr(m.time,'monotonic',lambda:next(ticks))
    stage_timing=dict(requests=800,output_tokens=800000,generation_seconds=90,wall_seconds=100,
        per_cell={k:dict(count=400,output_tokens=400000) for k in ('prefill/empty','prefill/reminder')})
    def done(channel,call,name,deadline):
        return {'smoke':measurement(1,count=8),'throughput':measurement(100),'h20_core':stage_timing}[name]
    monkeypatch.setattr(m,'wait_done',done)
    m.control_stages(p,channel,None,records.append)
    assert [r['decision'] for r in records]==['run','defer']
    assert records[-1]['stage']=='h21_main' and records[-1]['shortfall_usd']>0
    assert channel['stop'] is True and 'command:h21_main' not in channel
