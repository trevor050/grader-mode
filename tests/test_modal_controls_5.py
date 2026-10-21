"""Controls-only continuation keeps the hard dollar cap and old charges."""
import json
from pathlib import Path
from copy import deepcopy
import pytest
from src import modal_second_family_5 as base
from tests.test_modal_second_family_5 import grid, payload_for


def test_separate_authorization_refuses_old_identity_and_covers_setup(tmp_path):
    path=tmp_path/'controls.json'
    ledger=base.Authorization(path,cap=1.4,authorization_id='controls-test')
    ticket=ledger.begin('controls',now=100)
    assert ticket['reserved_usd']==pytest.approx(1.1)
    assert ticket['hard_deadline_unix']==1200
    assert ticket['work_deadline_unix']==1080
    ledger.finish('controls',True,now=250)
    assert ledger.snapshot()['attempts'][0]['charged_usd']==pytest.approx(.15)
    with pytest.raises(base.BudgetError):base.Authorization(path).snapshot()


def test_runtime_relaxation_does_not_relax_money_or_reserve():
    from src import modal_controls_5 as c
    prior=dict(stage='h20_controls',count=800,projected_seconds=567.435,
               projected_usd=.567435,runtime_margin=1.25,outcomes_consulted=False)
    # The old deadline gate failed at 517.56 s, while dollars fit.
    g=c.controls_gate(prior,elapsed_seconds=0,prior_charged_usd=0,seconds_left=517.56)
    assert g['decision']=='run' and g['runtime_projected_seconds']<517.56
    assert g['projected_usd']==.567435 and g['reserve_usd']==.3
    assert c.controls_gate(prior,elapsed_seconds=600,prior_charged_usd=0,seconds_left=1000)['decision']=='defer'
    assert c.controls_gate(prior,elapsed_seconds=0,prior_charged_usd=0,seconds_left=400)['decision']=='defer'
    assert g['outcomes_consulted'] is False


def test_free_credit_gate_and_fixed_cap():
    from src import modal_controls_5 as c
    bill=dict(billed_cost='0',metered_cost_breakdown={'ephemeral_apps':'28.2'})
    assert c.free_credit(bill)==pytest.approx(1.8)
    with pytest.raises(base.BudgetError):c.free_credit({**bill,'billed_cost':'.01'})
    with pytest.raises(base.BudgetError):c.free_credit({**bill,'metered_cost_breakdown':{'ephemeral_apps':'28.7'}})
    assert c.CAP==1.4 and c.RESERVE==.3


def test_controls_worker_never_repeats_old_stages_or_throughput(tmp_path,monkeypatch):
    monkeypatch.setattr(base.time,'sleep',lambda *_: (_ for _ in ()).throw(RuntimeError('unexpected wait for another stage')))
    rows=grid();p=payload_for(tmp_path,base,rows)
    p['controls_only']=True;p['authorization_id']='controls-test'
    channel={'command:h20_controls':dict(action='run',attempt_id=p['attempt_id'],count=800,
        work_deadline_unix=p['work_deadline_unix'],stage_deadline_unix=p['work_deadline_unix'])}
    calls=[]
    def fake(model,selected,run_id,manifest,config):
        name=Path(config['output_dir']).name;calls.append((name,len(selected)))
        out=Path(config['output_dir']);out.mkdir()
        with (out/(model.split('/')[-1]+'.jsonl')).open('w') as f:
            for r in selected:f.write(json.dumps({**r,'run_id':run_id,'stage':'pilot' if name=='smoke' else 'main','pilot':name=='smoke'})+'\n')
        per={}
        for r in selected:
            key=model.split('/')[-1]+'/'+base.cell_id(r)
            cell=per.setdefault(key,dict(count=0,output_tokens=0));cell['count']+=1;cell['output_tokens']+=100
        return dict(requests=len(selected),output_tokens=len(selected)*100,generation_seconds=.01,setup_seconds=0,per_cell=per)
    monkeypatch.setattr(base,'validate_runtime_smoke',lambda *a:dict(protocol_checked=True))
    state=base.worker_session(p,channel,checkpoint=lambda:None,cache_commit=lambda:None,generate_fn=fake)
    assert calls==[('smoke',8),('h20_controls',800)]
    assert state['status']=='completed' and state['completed_stages']==['h20_controls']
    assert state['deferred_stages']==[]
    assert not (tmp_path/'main').exists()


@pytest.mark.parametrize('verified',[True,False])
def test_execute_cancels_stops_then_pulls_and_retains_unresolved(tmp_path,monkeypatch,verified):
    from src import modal_controls_5 as c
    from src import vast_generate_5 as gen
    from contextlib import nullcontext
    from types import SimpleNamespace
    ledger=base.Authorization(tmp_path/'auth.json',cap=1.4,authorization_id='unit-controls')
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'authorization',lambda:ledger)
    prior=tmp_path/'prior.json';prior.write_text(json.dumps([dict(stage='h20_controls',count=800,
        projected_seconds=567.435,projected_usd=.567435,runtime_margin=1.25,outcomes_consulted=False)]))
    monkeypatch.setattr(c,'PRIOR_GATES',prior)
    monkeypatch.setattr(gen,'build',lambda *a:(grid(),{'implementation_sha256':{}}))
    events=[]
    monkeypatch.setattr(c,'app',SimpleNamespace(app_id='ap-own',run=lambda **k:nullcontext()))
    monkeypatch.setattr(c.modal,'enable_output',lambda:nullcontext())
    monkeypatch.setattr(c.modal.Dict,'ephemeral',lambda:nullcontext({}))
    monkeypatch.setattr(c.subprocess,'Popen',lambda *a,**kw:SimpleNamespace(pid=123))
    class Call:
        object_id='fc-own'
        def get(self,**kw):return dict(status='completed',completed_stages=['h20_controls'])
        def cancel(self,**kw):events.append('cancel');raise RuntimeError('cancel unavailable')
    monkeypatch.setattr(c,'controls_gpu',SimpleNamespace(spawn=lambda *a:Call()))
    monkeypatch.setattr(base,'wait_done',lambda *a:{'requests':8})
    def stop(app_id):
        assert app_id=='ap-own';events.append('stop');return verified,dict(app_stop_confirmed=verified)
    monkeypatch.setattr(c,'stop_and_verify',stop)
    def cli(*a,**k):
        if a[0]=='billing':return json.dumps(dict(billed_cost='0',metered_cost_breakdown={'ephemeral_apps':'28.2'}))
        if a[0]=='volume':events.append('pull');return ''
        return '[]'
    monkeypatch.setattr(c,'modal_cli',cli)
    if verified:
        assert c.execute('unit')['zero_phase5_app_verified']
        assert ledger.snapshot()['attempts'][0]['status']=='settled'
    else:
        with pytest.raises(base.BudgetError,match='unresolved'):c.execute('unit')
        assert ledger.snapshot()['attempts'][0]['charged_usd']>=1.1
    assert events.index('cancel')<events.index('stop')<events.index('pull')
