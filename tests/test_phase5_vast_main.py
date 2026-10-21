"""Offline spending/chronology tests, never contact a compute provider."""
import copy
import json
from pathlib import Path

import pytest
from src import vast_generate_5 as generator
from src.models_5 import NEMOTRON, GLM, PACKAGE_VERSIONS


def api():
    from src import phase5_vast_main
    return phase5_vast_main


def grid(model=NEMOTRON):
    return generator.expand_requests([dict(problem_idx=i, question=f'Problem {i}?', gold=i)
                                     for i in range(200)], model)


def measurement():
    m = api()
    return dict(requests=256, output_tokens=256000, generation_seconds=128,
                wall_seconds=130, batch_size=256, chunk_size=256,
                per_cell={k:dict(count=32, output_tokens=32000) for k in m.cell_ids()})


def write_raw_stage(directory, selected, run_id, stage, config):
    """Replayable synthetic continuations with the real generator's provenance."""
    from tests.test_phase5_score import Tokenizer, continuation
    from src.phase5_split import decoded_segments, split_response
    from src.models_5 import SAMPLING, REGISTRATION_COMMIT, RENDER_DATE, model_spec
    directory.mkdir(parents=True, exist_ok=True)
    rows=[]
    for request in selected:
        model=request['model']; tok=Tokenizer(model); spec=model_spec(model)
        response=continuation(model,thinking=request['thinking'])
        ids=tok.encode(response)
        row={**request, 'run_id':run_id,'stage':stage,'pilot':stage=='pilot',
             'registration_commit':REGISTRATION_COMMIT,'package_versions':PACKAGE_VERSIONS,
             'render_date':RENDER_DATE,'model_revision':spec['revision'],
             'tokenizer_revision':spec['tokenizer_revision'],'dtype':spec['dtype'],
             'quantization':spec['quantization'],'reasoning_effort':spec['reasoning_effort'],
             'sampling':{**SAMPLING,'skip_special_tokens':False,'stop_token_ids':spec['stop_token_ids']},
             'response':response,'raw_text':response,'output_token_ids':ids,'raw_token_ids':ids,
             'output_tokens':len(ids),'finish_reason':'stop','special_token_ids':tok.all_special_ids,
             'decoded_token_segments':decoded_segments(ids,tok,model),
             'initial_channel':'thought' if request['thinking'] else 'final',
             'engine':dict(batch_size=config.get('resident_sequences',256),chunk_size=256,
                           enforce_eager=config['enforce_eager'])}
        row.update(split_response(row)); rows.append(row)
    (directory/(selected[0]['model'].split('/')[-1]+'.jsonl')).write_text(
        ''.join(json.dumps(r)+'\n' for r in rows))
    return rows


def test_import_and_default_plan_are_inert(monkeypatch, capsys):
    m = api()
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **k: pytest.fail('no subprocess in prep'))
    assert m.main([]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['paid_actions_performed'] is False
    assert plan['stages'] == [dict(name='h20_core', count=800), dict(name='h21_main', count=1600),
                              dict(name='h20_controls', count=800)]
    assert plan['package_versions'] == PACKAGE_VERSIONS


def test_grid_exact_stages_and_balanced_pilots():
    m = api(); rows = grid()
    stages = m.stage_rows(rows)
    assert [(k,len(v)) for k,v in stages.items()] == [('h20_core',800),('h21_main',1600),('h20_controls',800)]
    assert len({r['id'] for v in stages.values() for r in v}) == 3200
    assert {m.cell_id(r) for r in m.balanced_pilot(rows, 8)} == set(m.cell_ids())
    pilot = m.balanced_pilot(rows, 256)
    assert len(pilot) == len({r['id'] for r in pilot}) == 256
    assert all(sum(m.cell_id(r)==key for r in pilot)==32 for key in m.cell_ids())
    with pytest.raises(ValueError, match='complete'):
        m.stage_rows(rows[:-1])


def test_measured_projection_preserves_reserve_and_complete_stage():
    m=api(); rows=m.stage_rows(grid())['h20_core']
    result=m.stage_gate(rows, measurement(), credit_usd=.50, ledger_remaining_usd=5,
                        hourly_usd=.65, seconds_left=5000, transfer_remaining_usd=.03)
    assert result['decision']=='run'
    assert result['count']==800
    assert result['reserve_usd']==.30
    assert result['projected_usd']==pytest.approx(1.2*800*1000/(256000/130)*.65/3600+.03)
    denied=m.stage_gate(rows,measurement(),credit_usd=.31,ledger_remaining_usd=5,
                        hourly_usd=.65,seconds_left=5000,transfer_remaining_usd=.03)
    assert denied['decision']=='stop_whole_stage'
    assert denied['count']==800
    assert denied['outcomes_consulted'] is False


@pytest.mark.parametrize('mutation', ['tiny','missing_cell','nan','cap','empty','bad_total'])
def test_measurement_gate_fails_closed(mutation):
    m=api(); metrics=measurement()
    if mutation=='tiny': metrics['requests']=16
    if mutation=='missing_cell': metrics['per_cell'].pop(next(iter(metrics['per_cell'])))
    if mutation=='nan': metrics['generation_seconds']=float('nan')
    if mutation=='cap': next(iter(metrics['per_cell'].values()))['output_tokens']=32*4097
    if mutation=='empty': next(iter(metrics['per_cell'].values()))['count']=0
    if mutation=='bad_total': metrics['output_tokens']=123
    with pytest.raises(ValueError): m.measurement_summary(metrics)


def test_priority_stop_never_runs_half_stage_or_lower_priority():
    m=api(); ran=[]; saved=[]
    def gate(name, rows):
        return dict(decision='run' if name=='h20_core' else 'stop_whole_stage', count=len(rows))
    result=m.run_stages(grid(),gate,lambda name,rows:ran.append((name,len(rows))),saved.append)
    assert ran==[('h20_core',800)]
    assert result['completed_stages']==['h20_core']
    assert result['dropped_stages']==['h21_main','h20_controls']
    assert [r['stage'] for r in saved]==['h20_core','h21_main']


def test_two_runtime_failures_persist_decision_before_any_glm_generation():
    m=api(); events=[]; proofs=[]; now=iter(range(1,100))
    def candidate(model,stage,proof):
        events.append((model,stage))
        if model==NEMOTRON:
            raise m.RuntimeSmokeFailure(dict(attempt_id=stage,stage=stage,purpose='runtime_smoke',
                 status='failed',violation_scored=False,started_unix=next(now),ended_unix=next(now),
                 evidence='saved/runtime-error.json'))
        assert len(proofs)==1 and proofs[0]['decision_unix']>proofs[0]['attempts'][-1]['ended_unix']
        assert 'first_generation_unix' not in proofs[0]
        proof['first_generation_unix']=next(now)
        return dict(model=model,proof=proof)
    result=m.run_families(candidate,lambda proof:proofs.append(copy.deepcopy(proof)),lambda:next(now))
    assert events==[(NEMOTRON,'startup'),(NEMOTRON,'fix'),(GLM,'fallback')]
    from src.analyze_5 import validate_fallback_provenance
    assert validate_fallback_provenance(result['proof'])=='runtime_startup_failure'


def test_successful_fix_never_loads_glm():
    m=api(); events=[]
    def candidate(model,stage,proof):
        events.append((model,stage))
        if stage=='startup':
            raise m.RuntimeSmokeFailure(dict(attempt_id='a',stage=stage,purpose='runtime_smoke',status='failed',
                violation_scored=False,started_unix=1,ended_unix=2,evidence='runtime-error.json'))
        return {'model':model}
    assert m.run_families(candidate,lambda p:pytest.fail('no fallback proof'))['model']==NEMOTRON
    assert events==[(NEMOTRON,'startup'),(NEMOTRON,'fix')]


@pytest.mark.parametrize('error',[RuntimeError('provider failure'),ValueError('invalid package pins')])
def test_nonruntime_failure_never_triggers_family_switch(error):
    m=api(); calls=[]
    def candidate(*args): calls.append(args); raise error
    with pytest.raises(type(error)):
        m.run_families(candidate,lambda p:pytest.fail('no fallback'))
    assert len(calls)==1


def test_offer_accounting_includes_80gb_storage_and_all_candidate_downloads():
    m=api()
    offer=dict(id=1,num_gpus=1,gpu_name='RTX 5090',gpu_ram=32607,reliability=.995,
        inet_down=704.8,dph_base=.6,storage_cost=.2,inet_down_cost=.0026041667,
        inet_up_cost=.00390625,rentable=True,verified=True)
    costs=m.offer_costs(offer)
    assert costs['hourly_usd']==pytest.approx(.6+80*.2/720)
    assert costs['network_usd']==pytest.approx(100*.0026041667+8*.00390625)
    assert m.public_account(dict(credit=2.8,total_spend=1,api_key='secret',email='x'))==dict(credit=2.8,total_spend=1)
    allocation=m.deadline_allocation(2.824799,9,costs)
    assert allocation['reserved_usd']<=2.824799-.30
    assert allocation['startup_seconds']>=600
    assert allocation['runtime_seconds']>0
    assert allocation['teardown_seconds']>=120


def test_reserve_and_detached_watchdog_precede_create_and_bind(tmp_path):
    m=api(); events=[]
    class Ledger:
        def reserve(self,*a,**k): events.append('reserve'); return dict(resource_label='phase5-ticket')
        def bind_resource(self,*a): events.append('bind')
    class Provider:
        def create(self,offer,label): events.append(('create',label)); return 123
    result=m.reserve_create(Ledger(),Provider(),'ticket',dict(id=1),
                            dict(hourly_usd=1,runtime_seconds=1,startup_seconds=1,teardown_seconds=1),
                            lambda *a:events.append('watchdog'))
    assert result==123
    assert events==['reserve','watchdog',('create','phase5-ticket'),'bind']


def test_pull_failure_still_destroys_and_confirms_global_zero():
    m=api(); events=[]
    class Provider:
        def pull(self): events.append('pull'); raise OSError('transfer failed')
        def destroy_owned(self): events.append('destroy')
        def instances(self): events.append('zero'); return []
    result=m.cleanup(Provider())
    assert result['zero_instances'] is True
    assert result['pulled'] is False
    assert events==['pull','destroy','zero']


def test_cached_engine_constructed_once_and_refuses_option_drift():
    m=api(); calls=[]
    factory=m.CachedEngine(lambda **kw:calls.append(kw) or object())
    first=factory(model=NEMOTRON,max_num_seqs=256,enforce_eager=True)
    assert factory(model=NEMOTRON,max_num_seqs=256,enforce_eager=True) is first
    assert len(calls)==1
    with pytest.raises(ValueError,match='engine'):
        factory(model=GLM,max_num_seqs=256,enforce_eager=True)


def test_network_guard_stops_at_transfer_or_clock_limit_without_global_kill():
    m=api()
    assert m.remote_stop_reason((0,0),(1,1),deadline=100,now=99) is None
    assert m.remote_stop_reason((0,0),(99*10**9,1),deadline=100,now=99)=='download_budget'
    assert m.remote_stop_reason((0,0),(1,7*10**9),deadline=100,now=99)=='upload_budget'
    assert m.remote_stop_reason((0,0),(1,1),deadline=100,now=100)=='absolute_deadline'


@pytest.mark.parametrize('resident',[128,256])
def test_persistent_worker_reuses_engine_across_smoke_throughput_and_three_stages(tmp_path,monkeypatch,resident):
    import sys
    import types
    m=api(); root=tmp_path/'remote'; output=root/'output'; output.mkdir(parents=True)
    monkeypatch.setattr(m,'REMOTE_ROOT',str(root))
    rows=grid(); (root/'requests.json').write_text(json.dumps({NEMOTRON:dict(requests=rows,manifest={})}))
    from src.models_5 import REGISTRATION_COMMIT
    config=dict(model=NEMOTRON,attempt_id='test',authorization_id='NEW-task',
                registration_commit=REGISTRATION_COMMIT,deadline_unix=m.time.time()+100,
                enforce_eager=True,resident_sequences=resident)
    (root/'config.json').write_text(json.dumps(config))
    for name,count in zip(('throughput',*m.STAGES),(256,800,1600,800)):
        (output/(name+'.permit.json')).write_text(json.dumps(dict(action='run',count=count,
              attempt_id='test',deadline_unix=config['deadline_unix'],measurement_deadline_unix=config['deadline_unix'])))
    constructs=[]; calls=[]; run_ids=[]
    vllm=types.SimpleNamespace(LLM=lambda **kw:constructs.append(kw) or object())
    monkeypatch.setitem(sys.modules,'vllm',vllm)
    monkeypatch.setattr(generator,'runtime_versions',lambda:PACKAGE_VERSIONS.copy())
    def generate(model,selected,run_id,manifest,config):
        vllm.LLM(model=model,max_num_seqs=config['batch_size'],enforce_eager=config['enforce_eager'])
        calls.append((manifest['controller_stage'],len(selected),manifest['stage']))
        run_ids.append(run_id)
        write_raw_stage(Path(config['output_dir']), selected, run_id, manifest['stage'], config)
        counts=Counter(m.cell_id(r) for r in selected)
        return dict(requests=len(selected),output_tokens=len(selected)*1000,
                    generation_seconds=.000001,setup_seconds=0,
                    per_cell={model.split('/')[-1]+'/'+k:dict(count=n,output_tokens=n*1000) for k,n in counts.items()})
    from collections import Counter
    monkeypatch.setattr(generator,'generate',generate)
    m.worker(root/'config.json')
    assert calls==[('smoke',8,'pilot'),('throughput',256,'pilot'),('h20_core',800,'main'),
                   ('h21_main',1600,'main'),('h20_controls',800,'main')]
    assert len(constructs)==1
    assert constructs[0]['max_num_seqs']==resident
    assert json.loads((output/'throughput.done.json').read_text())['batch_size']==resident
    assert run_ids == ['test-smoke','test-throughput','test-main','test-main','test-main']
    assert (output/'finished.json').exists()


def test_smoke_projection_excludes_paid_setup_from_generation_throughput():
    m=api()
    result=m.timing_measurement(dict(requests=8,output_tokens=8000,generation_seconds=10,
            setup_seconds=600,per_cell={}),wall_seconds=612,name='smoke')
    assert result['wall_seconds']==12
    assert result['setup_seconds']==600
    # A reused main engine's per-call overhead stays in its measurement.
    assert m.timing_measurement(dict(requests=256,output_tokens=256000,generation_seconds=10,
            setup_seconds=2,per_cell={}),wall_seconds=12,name='throughput')['wall_seconds']==12


def test_gemma_order_guard_accepts_completed_metadata_without_outcomes(tmp_path):
    m=api(); from src.models_5 import GEMMA
    path=tmp_path/'analysis.json'
    path.write_text(json.dumps(dict(phase='phase5',grid=dict(model_scope=GEMMA,n_problems=200,
         n_responses=2400,mode='cell_drop_stage_1',problem_manifest_checked=True,original_sample_checked=True),
         parser_agreement=dict(status='complete',n_required=40,n_labeled=40,
             committed_number=dict(status='complete'),splitter=dict(status='complete',n_labeled=40)))))
    assert m.require_gemma_analysis(path)['outcomes_consulted'] is False
    doc=json.loads(path.read_text()); doc['parser_agreement']['splitter']['n_labeled']=39
    path.write_text(json.dumps(doc))
    with pytest.raises(Exception,match='blind parser'):m.require_gemma_analysis(path)
    doc['grid']['mode']='pilot';path.write_text(json.dumps(doc))
    with pytest.raises(Exception,match='Gemma analysis'):m.require_gemma_analysis(path)


@pytest.fixture
def fake_supervisor(tmp_path, monkeypatch):
    """Real ledger + real supervisor + fake network/worker boundaries."""
    from types import SimpleNamespace
    m=api(); from src.phase5_control import BudgetLedger
    from src.models_5 import REGISTRATION_COMMIT
    ledger=BudgetLedger.initialize(tmp_path/'ledger.json')
    ledger.upgrade_registration(registration_commit=REGISTRATION_COMMIT)
    directory=tmp_path/'raw';directory.mkdir()
    bundle=directory/'bundle.tar.gz';bundle.write_bytes(b'fake bundle')
    bundle.with_suffix('.requests.json').write_text(json.dumps({model:dict(requests=grid(model),manifest={})
                                                               for model in (NEMOTRON,GLM)}))
    events=[]; providers=[]
    class FakeProvider:
        runtime_failures=0
        pull_fails=False
        lower_credit_after_core=False
        def __init__(self, local):
            self.local=Path(local);self.label=None;self.instance=None;self.host='mock';self.port=22
            self.permits=[];self.active=False;self.operation_deadline=None;providers.append(self)
        def account(self):
            credit=.31 if self.lower_credit_after_core and 'h20_core' in self.permits else 2.824799
            return dict(credit=credit,total_spend=0)
        def instances(self):
            return [dict(id=self.instance,label=self.label,dph_total=.62222222)] if self.active else []
        def owned(self,rows):return rows
        def scan(self,offer_id):
            events.append('fresh_scan')
            return dict(id=offer_id,num_gpus=1,gpu_name='RTX 5090',gpu_ram=32607,
                 reliability=.995,verified=True,rentable=True,dph_base=.6,storage_cost=.2,
                 inet_down_cost=.0026041667,inet_up_cost=.00390625)
        def create(self,offer,label):
            assert ledger.snapshot()['attempts'][-1]['status']=='reserved'
            events.append('create');self.active=True;self.instance=len(providers);self.label=label
            return self.instance
        def wait_ssh(self,deadline):pass
        def ssh(self,*a,**k):return ''
        def put(self,*a,**k):pass
        def send(self,name,value):
            if name.endswith('.permit.json'):self.permits.append(name.removesuffix('.permit.json'))
        def read(self,name):
            model=GLM if self.local.name=='fallback' else NEMOTRON
            if name=='error.json':
                if self.local.name in ('startup','fix') and self.runtime_failures:
                    type(self).runtime_failures-=1
                    return dict(phase='runtime_smoke',started_unix=m.time.time()-.0001,
                                ended_unix=m.time.time(),error_type='RuntimeError')
                return None
            if name=='smoke.done.json':
                resident=json.loads((self.local/'worker-config.json').read_text())['resident_sequences']
                return dict(requests=8,output_tokens=8000,generation_seconds=4,wall_seconds=5,
                     batch_size=resident,chunk_size=256,first_generation_unix=m.time.time(),
                     protocol=dict(requests=8,protocol_checked=True,violation_scored=False),
                     per_cell={k:dict(count=1,output_tokens=1000) for k in m.cell_ids()})
            if name=='throughput.done.json':
                return measurement() | dict(batch_size=json.loads((self.local/'worker-config.json').read_text())['resident_sequences'])
            for stage in m.STAGES:
                if name==stage+'.done.json' and stage in self.permits:
                    selected=m.stage_rows(grid(model))[stage]
                    from collections import Counter
                    counts=Counter(m.cell_id(r) for r in selected)
                    return dict(requests=len(selected),output_tokens=1000*len(selected),
                         wall_seconds=len(selected)/2,per_cell={k:dict(count=n,output_tokens=n*1000) for k,n in counts.items()})
            return None
        def pull(self):
            events.append('pull')
            if self.pull_fails:raise OSError('mock transfer failure')
            model=GLM if self.local.name=='fallback' else NEMOTRON
            if self.local.name in ('startup','fix') and not self.permits:
                (self.local/'runtime-trace.txt').write_text('torch.OutOfMemoryError: CUDA out of memory')
            bundled=json.loads(bundle.with_suffix('.requests.json').read_text())
            for stage in self.permits:
                if stage in m.STAGES:
                    path=self.local/stage;path.mkdir(exist_ok=True)
                    selected=m.stage_rows(bundled[model]['requests'])[stage]
                    config=json.loads((self.local/'worker-config.json').read_text())
                    write_raw_stage(path,selected,config['attempt_id']+'-main','main',config)
        def destroy_owned(self):events.append('destroy');self.active=False
    monkeypatch.setattr(m,'Provider',FakeProvider)
    monkeypatch.setattr(m,'arm_watchdog',lambda *a:events.append('watchdog'))
    args=SimpleNamespace(run_id='new-execution',offer_id=51405136,max_hourly_usd=.75,authorization_id='NEW-task')
    supervisor=m.Supervisor(args,ledger,directory,bundle,dict(credit=2.824799))
    return supervisor,ledger,FakeProvider,events,providers


def test_supervisor_complete_run_reserves_pulls_destroys_and_settles(fake_supervisor):
    m=api();supervisor,ledger,provider,events,providers=fake_supervisor
    result=m.run_families(supervisor.candidate,supervisor.persist_proof)
    assert result['completed_stages']==list(m.STAGES)
    assert events[:3]==['fresh_scan','watchdog','create']
    assert events[-1]=='destroy'
    ticket=ledger.snapshot()['attempts'][0]
    assert ticket['status']=='completed' and ticket['teardown_verified'] is True
    assert ticket['charged_microusd']>290000  # full conservative bandwidth cost retained
    assert not providers[0].active


def test_supervisor_failed_pull_destroys_and_never_switches_family(fake_supervisor):
    m=api();supervisor,ledger,provider,events,providers=fake_supervisor
    provider.pull_fails=True
    with pytest.raises(RuntimeError,match='artifact recovery failed'):
        m.run_families(supervisor.candidate,supervisor.persist_proof)
    assert len(providers)==1 and 'destroy' in events
    ticket=ledger.snapshot()['attempts'][0]
    assert ticket['status']=='failed' and ticket['teardown_verified'] is True


def test_supervisor_two_smoke_failures_then_glm_have_real_prospective_proof(fake_supervisor):
    m=api();supervisor,ledger,provider,events,providers=fake_supervisor
    provider.runtime_failures=2
    result=m.run_families(supervisor.candidate,supervisor.persist_proof)
    assert result['model']==GLM
    assert len(providers)==3 and events.count('destroy')==3
    proof=json.loads((supervisor.directory/'fallback-provenance.json').read_text())
    from src.analyze_5 import validate_fallback_provenance
    assert validate_fallback_provenance(proof)=='runtime_startup_failure'
    assert [a['status'] for a in ledger.snapshot()['attempts']]==['failed','failed','completed']
    assert all(a['teardown_verified'] for a in ledger.snapshot()['attempts'])


def test_supervisor_live_credit_drop_stops_whole_next_stage(fake_supervisor):
    m=api();supervisor,ledger,provider,events,providers=fake_supervisor
    provider.lower_credit_after_core=True
    result=m.run_families(supervisor.candidate,supervisor.persist_proof)
    assert result['completed_stages']==['h20_core']
    assert result['dropped_stages']==['h21_main','h20_controls']
    assert providers[0].permits==['throughput','h20_core']
    assert not providers[0].active


def test_lost_create_response_is_pulled_destroyed_and_never_retried_blindly(fake_supervisor,monkeypatch):
    m=api();supervisor,ledger,provider,events,providers=fake_supervisor
    original=provider.create
    def uncertain(self,*args):
        original(self,*args)
        raise RuntimeError('lost create response')
    monkeypatch.setattr(provider,'create',uncertain)
    with pytest.raises(RuntimeError,match='lost create'):
        m.run_families(supervisor.candidate,supervisor.persist_proof)
    assert len(providers)==1 and not providers[0].active
    assert events[-2:]==['pull','destroy']
    assert ledger.snapshot()['attempts'][0]['teardown_verified'] is True


def test_main_export_contains_only_completed_stages_and_preserves_raw_bytes(fake_supervisor):
    m=api();supervisor,ledger,provider,events,providers=fake_supervisor
    provider.lower_credit_after_core=True
    result=m.run_families(supervisor.candidate,supervisor.persist_proof)
    export=Path(result['main_output'])
    rows=[json.loads(line) for line in export.read_text().splitlines()]
    assert len(rows)==800
    assert all(r['arm']=='prefill' and r['prefill'] in ('empty','reminder') for r in rows)
    source=supervisor.directory/'startup/h20_core'/(NEMOTRON.split('/')[-1]+'.jsonl')
    assert export.read_bytes()==source.read_bytes()
    assert Path(result['manifest']).exists()


def test_create_disables_cli_transport_retries(monkeypatch,tmp_path):
    m=api();backend=m.Provider(tmp_path);calls=[]
    monkeypatch.setattr(backend,'run',lambda argv,**kwargs:calls.append(argv) or '{"success":true,"new_contract":123}')
    assert backend.create(dict(id=1),'phase5-one')==123
    assert '--retry' in calls[0] and calls[0][calls[0].index('--retry')+1]=='1'


def test_pull_enforces_cumulative_egress_envelope_before_scp(monkeypatch,tmp_path):
    m=api();backend=m.Provider(tmp_path,host='mock',port=22);calls=[]
    monkeypatch.setattr(backend,'ssh',lambda *a,**k:str(4*10**9))
    monkeypatch.setattr(backend,'run',lambda argv,**kwargs:calls.append(argv) or '')
    backend.pull()
    with pytest.raises(Exception,match='transfer'):
        backend.pull()
    assert len(calls)==1  # refused before the second paid transfer


def test_uncertain_create_with_no_instance_yet_keeps_ticket_and_watchdog_unresolved(fake_supervisor,monkeypatch):
    m=api();supervisor,ledger,provider,events,providers=fake_supervisor
    def uncertain(self,*args):raise RuntimeError('request timed out before contract ID')
    monkeypatch.setattr(provider,'create',uncertain)
    with pytest.raises(Exception,match='zero Vast instances not confirmed'):
        m.run_families(supervisor.candidate,supervisor.persist_proof)
    ticket=ledger.snapshot()['attempts'][0]
    assert ticket['status']=='recovered_unresolved'
    assert ticket['teardown_verified'] is False
    assert ticket['charged_microusd']>=ticket['reserved_microusd']
    assert any(e.get('create_uncertain') is True for e in ticket['events'])


def test_full_export_passes_strict_family_analyzer(fake_supervisor,monkeypatch):
    from src import analyze_5
    from src.analyze_5 import analyze
    from src.models_5 import REGISTRATION_COMMIT
    m=api(); supervisor,ledger,provider,events,providers=fake_supervisor
    # Synthetic identities stand in for the local pinned dataset cache only;
    # full grid, manifest, every row pin, and native saved split stay strict.
    sample=[dict(problem_idx=p,gold=p,question=f'Problem {p}?') for p in range(200)]
    monkeypatch.setattr(analyze_5,'_original_sample',lambda:{p['problem_idx']:(p['gold'],p['question']) for p in sample})
    bundled=json.loads(supervisor.bundle.with_suffix('.requests.json').read_text())
    bundled[NEMOTRON]=dict(requests=generator.expand_requests(sample,NEMOTRON),
                          manifest=dict(registration_commit=REGISTRATION_COMMIT,sample=sample))
    supervisor.bundle.with_suffix('.requests.json').write_text(json.dumps(bundled))
    result=m.run_families(supervisor.candidate,supervisor.persist_proof)
    exported=Path(result['main_output'])
    rows=[json.loads(line) for line in exported.read_text().splitlines()]
    assert len(rows)==3200 and {r['run_id'] for r in rows}=={'new-execution-startup-main'}
    sources=[supervisor.directory/'startup'/name/(NEMOTRON.split('/')[-1]+'.jsonl') for name in m.STAGES]
    assert exported.read_bytes()==b''.join(p.read_bytes() for p in sources)
    report=analyze(rows,model_scope=NEMOTRON,problem_manifest=json.loads(Path(result['manifest']).read_text()))
    assert report['grid']['mode']=='full' and report['grid']['original_sample_checked'] is True
    assert report['grid']['problem_manifest_checked'] is True
    assert not report['inference']['eligible']  # Synthetic replies, no human labels.
    rows[-1]['run_id']='accidentally-stage-specific'
    with pytest.raises(ValueError,match='run_id'):
        analyze(rows,model_scope=NEMOTRON)


def test_slow_smoke_cannot_veto_bounded_full_chunk_measurement(fake_supervisor,monkeypatch):
    m=api(); supervisor,ledger,provider,events,providers=fake_supervisor
    original=provider.read
    def read(self,name):
        value=original(self,name)
        if name=='smoke.done.json':
            value.update(wall_seconds=1000000,generation_seconds=1000000)
        return value
    monkeypatch.setattr(provider,'read',read)
    result=m.run_families(supervisor.candidate,supervisor.persist_proof)
    assert result['completed_stages']==list(m.STAGES)
    receipt=json.loads((supervisor.directory/'startup/throughput-receipt.json').read_text())
    assert receipt['allowance']['allowance_seconds']==900
    assert receipt['allowance']['maximum_usd']<.16
    assert receipt['smoke_speed_used'] is False


@pytest.mark.parametrize('credit,seconds',[(.31,1000),(2.8,899)])
def test_measurement_allowance_refuses_insufficient_money_or_time(credit,seconds):
    m=api()
    result=m.throughput_allowance(credit_usd=credit,ledger_remaining_usd=9,
          hourly_usd=.65,seconds_left=seconds,now=100)
    assert result['decision']=='stop_measurement'
    assert result['deadline_unix']==1000 and result['smoke_speed_used'] is False


@pytest.mark.parametrize('error,resident,eager',[
    ('torch.OutOfMemoryError: CUDA out of memory',128,False),
    ('No available memory for the cache blocks',128,False),
    ('RuntimeError: CUDA graph capture failed',256,True)])
def test_targeted_fix_changes_only_matching_engine_setting(error,resident,eager):
    policy=api().runtime_fix_policy(error)
    assert policy['resident_sequences']==resident and policy['enforce_eager'] is eager
    assert api().runtime_fix_policy('unsupported Mamba kernel') is None
    measured=measurement() | dict(batch_size=resident)
    assert api().measurement_summary(measured)['measured_requests']==256


def test_unsupported_first_error_stops_before_speculative_fix_rental(fake_supervisor,monkeypatch):
    m=api(); supervisor,ledger,provider,events,providers=fake_supervisor
    provider.runtime_failures=1
    original=provider.pull
    def pull(self):
        original(self)
        (self.local/'runtime-trace.txt').write_text('Unsupported Mamba kernel on this architecture')
    monkeypatch.setattr(provider,'pull',pull)
    with pytest.raises(Exception,match='no supported targeted fix'):
        m.run_families(supervisor.candidate,supervisor.persist_proof)
    assert events.count('create')==1 and not providers[0].active


@pytest.mark.parametrize('case',['valid','cap','stale_split','protocol','short_cap','missing'])
def test_smoke_replays_protocol_without_scoring_and_accepts_actual_cap(tmp_path,case):
    from src.phase5_split import split_response, decoded_segments
    from tests.test_phase5_score import Tokenizer
    m=api(); selected=m.balanced_pilot(grid(),8)
    rows=write_raw_stage(tmp_path,selected,'pilot','pilot',dict(enforce_eager=False))
    if case in ('cap','short_cap'):
        r=rows[0];tok=Tokenizer(NEMOTRON)
        ids=tok.encode('r'*(4096 if case=='cap' else 100))
        r.update(output_token_ids=ids,raw_token_ids=ids,response=tok.decode(ids),raw_text=tok.decode(ids),
                 decoded_token_segments=decoded_segments(ids,tok,NEMOTRON),output_tokens=len(ids),finish_reason='length')
        r.update(split_response(r))
    if case=='stale_split': rows[0]['final_reply']='tampered'
    if case=='protocol':
        r=rows[0];tok=Tokenizer(NEMOTRON);ids=tok.encode('<think>nested')
        r.update(output_token_ids=ids,raw_token_ids=ids,response=tok.decode(ids),raw_text=tok.decode(ids),
                 decoded_token_segments=decoded_segments(ids,tok,NEMOTRON),output_tokens=len(ids))
        r.update(split_response(r))
    if case=='missing':rows.pop()
    (tmp_path/(NEMOTRON.split('/')[-1]+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    if case in ('valid','cap'):
        assert m.validate_runtime_smoke(tmp_path,selected)['violation_scored'] is False
    else:
        with pytest.raises(RuntimeError):m.validate_runtime_smoke(tmp_path,selected)


def test_bundle_remote_prepare_needs_only_stdlib_and_all_worker_imports_are_present(tmp_path):
    import subprocess,tarfile,sys
    m=api(); bundle=tmp_path/'bundle.tar.gz'
    m.make_bundle(bundle,{})
    extracted=tmp_path/'extracted';extracted.mkdir()
    with tarfile.open(bundle) as archive:archive.extractall(extracted,filter='data')
    result=subprocess.run([sys.executable,'-S','-m','src.phase5_vast_main','--prepare'],
                          cwd=extracted,capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['paid_actions_performed'] is False
    # Worker imports, before GPU dependencies, must work in the minimal bundle.
    subprocess.run([sys.executable,'-S','-c','from src import vast_generate_5, phase5_split'],
                   cwd=extracted,check=True,capture_output=True)


def test_bootstrap_shell_syntax_and_explicit_python3(fake_supervisor):
    import subprocess
    m=api(); supervisor,*_=fake_supervisor
    m.run_families(supervisor.candidate,supervisor.persist_proof)
    path=supervisor.directory/'startup/bootstrap.sh'
    subprocess.run(['bash','-n',str(path)],check=True)
    text=path.read_text()
    assert 'uv pip install --system --no-deps' in text and 'exec python3 -m src.phase5_vast_main' in text
    assert all(k+'=='+v in text for k,v in PACKAGE_VERSIONS.items() if k != 'vllm')
    assert "m.version('vllm') == '0.32.0'" in text


def test_remote_guard_enforces_measurement_deadline_during_blocking_generate(tmp_path,monkeypatch):
    m=api(); root=tmp_path/'remote';out=root/'output';out.mkdir(parents=True)
    monkeypatch.setattr(m,'REMOTE_ROOT',str(root))
    config=root/'config.json';config.write_text(json.dumps(dict(deadline_unix=m.time.time()+1000)))
    (out/'throughput.permit.json').write_text(json.dumps(dict(measurement_deadline_unix=m.time.time()-1)))
    class Child:
        pid=12345;returncode=None
        def poll(self):return self.returncode
        def wait(self,**kwargs):self.returncode=-15;return -15
    killed=[]
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**k:Child())
    monkeypatch.setattr(m.os,'killpg',lambda pid,sig:killed.append((pid,sig)))
    monkeypatch.setattr(m,'network_counters',lambda:(0,0))
    m.guarded_launch(config)
    assert killed==[(12345,m.signal.SIGTERM)]
    assert json.loads((out/'error.json').read_text())['error_type']=='throughput_measurement_deadline'


def test_memory_fix_is_saved_before_rental_and_reported_as_128_resident(fake_supervisor):
    m=api(); supervisor,ledger,provider,events,providers=fake_supervisor
    provider.runtime_failures=1
    result=m.run_families(supervisor.candidate,supervisor.persist_proof)
    assert len(providers)==2 and result['model']==NEMOTRON
    launch=json.loads((supervisor.directory/'fix/launch-accounting.json').read_text())
    config=json.loads((supervisor.directory/'fix/worker-config.json').read_text())
    receipt=json.loads((supervisor.directory/'fix/throughput-receipt.json').read_text())
    assert launch['runtime_policy']==config['runtime_policy']==dict(
        kind='memory_concurrency',resident_sequences=128,enforce_eager=False)
    assert receipt['measured']['batch_size']==128
    assert receipt['measured']['chunk_size']==256 and receipt['measured']['requests']==256


def test_explicit_refusal_has_no_transfer_charge_and_no_uncertain_quarantine(fake_supervisor,monkeypatch):
    m=api();supervisor,ledger,provider,events,providers=fake_supervisor
    def refused(self,offer,label):raise m.ProviderCreateRejected('definitive refusal')
    monkeypatch.setattr(provider,'create',refused)
    with pytest.raises(m.ProviderCreateRejected):m.run_families(supervisor.candidate,supervisor.persist_proof)
    ticket=ledger.snapshot()['attempts'][0]
    assert ticket['teardown_verified'] is True
    assert ticket['resource_id'] is None
    assert any(e.get('create_rejected') for e in ticket['events'])
    assert not any(e.get('create_uncertain') for e in ticket['events'])
    assert ticket['charged_microusd'] < 20000  # elapsed reservation floor only; no model transfer
