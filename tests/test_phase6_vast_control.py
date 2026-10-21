import pytest
from src.phase6_vast_control import choose_offer, offer_costs, stage_gate, measurement


def offer(**extra):
    return dict(id=1,machine_id=1,gpu_name='RTX PRO 6000 WS',gpu_ram=97887,
                num_gpus=1,reliability=.995,rentable=True,verification='verified',
                cuda_max_good=13.0,dph_base=1.1,storage_cost=.2,
                inet_down_cost=.001,inet_up_cost=.002,**extra)


def metrics():
    return dict(requests=256,generation_seconds=100,wall_seconds=120,output_tokens=256000,
                per_cell={'main/off/plain':dict(count=128,output_tokens=128000),
                          'main/on/explicit':dict(count=128,output_tokens=128000)})


def test_all_in_price_and_transfer_cap():
    cost=offer_costs(offer())
    assert cost['hourly_usd']==pytest.approx(1.1+100*.2/720)
    assert cost['network_envelope_usd']==pytest.approx(.136)
    bad=offer();bad['dph_base']=1.3
    with pytest.raises(ValueError):offer_costs(bad)
    bad=offer();bad['gpu_ram']=32000
    with pytest.raises(ValueError):offer_costs(bad)


def test_native_full_batch_gpu_and_failed_host_exclusion():
    a=offer();b=offer();b.update(id=2,machine_id=2,gpu_name='A100 SXM4',dph_base=.7)
    assert choose_offer([a,b],[])[0]['id']==1
    assert choose_offer([a,b],[1])[0]['id']==2
    with pytest.raises(RuntimeError):choose_offer([a,b],[1,2])


def test_stage_is_dropped_whole_with_dollar_shortfall():
    rows=[dict(arm='main',thinking=False,request_type='plain')]*1600
    yes=stage_gate(rows,metrics(),remaining=2,hourly=1,seconds_left=6000)
    no=stage_gate(rows,metrics(),remaining=.4,hourly=1,seconds_left=6000)
    assert yes['decision']=='run' and yes['rows']==1600
    assert no['decision']=='drop' and no['rows']==1600
    assert no['shortfall_usd']==pytest.approx(no['projected_usd']-.1)
    assert not no['outcomes_consulted']


def test_measurement_cannot_omit_cells_or_invent_token_counts():
    m=metrics();m['output_tokens']+=1
    with pytest.raises(ValueError):measurement(m)
    m=metrics();m['requests']=8
    with pytest.raises(ValueError):measurement(m)
    with pytest.raises(ValueError):stage_gate([dict(arm='prefill',prefill='reminder')],metrics(),remaining=5,hourly=1,seconds_left=6000)


def test_cli_timeout_retries_reads_but_never_create(monkeypatch,tmp_path):
    import subprocess
    from src import phase6_vast_control as c
    monkeypatch.setattr(c,'CLI_LOCK',tmp_path/'cli.lock')
    monkeypatch.setattr(c.time,'sleep',lambda _:None)
    calls=[]
    def call(argv,**kw):
        calls.append(argv)
        if len(calls)==1:raise subprocess.TimeoutExpired(argv,50)
        return subprocess.CompletedProcess(argv,0,'[]','')
    monkeypatch.setattr(c.subprocess,'run',call)
    assert c.cli('show','instances')==[] and len(calls)==2
    calls.clear()
    with pytest.raises(RuntimeError):c.cli('create','instance',123)
    assert len(calls)==1


def test_partial_chunk_retrieved_and_replaced_only_after_checksum(monkeypatch,tmp_path):
    import hashlib,json
    from src import phase6_vast_control as c
    conn=c.Connection(dict(ssh_host='example',ssh_port=22))
    good=b'{"id":"complete"}\n'
    digest=hashlib.sha256(good).hexdigest()
    monkeypatch.setattr(conn,'ssh',lambda _:json.dumps([dict(name='main/chunk-00000.jsonl',sha256=digest)]))
    dest=tmp_path/'main/chunk-00000.jsonl';dest.parent.mkdir();dest.write_bytes(b'partial')
    def run(argv,**kw):
        target=c.Path(argv[-1])
        target.write_bytes(good if target.suffix=='.download' else b'worker log')
    monkeypatch.setattr(c,'run',run)
    conn.pull(tmp_path)
    assert dest.read_bytes()==good
    assert not list(tmp_path.rglob('*.download'))


def test_watchdog_does_not_exit_after_destroy_if_owned_instance_remains(monkeypatch,tmp_path):
    import json
    from src import phase6_vast_control as c
    config=tmp_path/'watchdog.json'
    config.write_text(json.dumps(dict(controller_pid=999999,label='mine',deadline_unix=0,instance_id=4)))
    owned=[dict(id=4,label='mine')]
    replies=iter([owned,owned,owned,[]])
    monkeypatch.setattr(c,'instances',lambda:next(replies))
    monkeypatch.setattr(c.time,'sleep',lambda _:None)
    destroyed=[]
    monkeypatch.setattr(c,'cli',lambda *a,**kw:destroyed.append(a))
    c.watchdog(config)
    assert len(destroyed)==2
    assert json.loads((tmp_path/'watchdog-fired.json').read_text())['zero_verified']


def test_mutable_progress_can_advance_between_listing_and_download(monkeypatch,tmp_path):
    import hashlib,json
    from src import phase6_vast_control as c
    conn=c.Connection(dict(ssh_host='example',ssh_port=22))
    old=b'{"requests":256}';new=b'{"requests":512}'
    monkeypatch.setattr(conn,'ssh',lambda _:json.dumps([dict(name='progress.json',sha256=hashlib.sha256(old).hexdigest())]))
    def run(argv,**kw):c.Path(argv[-1]).write_bytes(new if argv[-1].endswith('.download') else b'log')
    monkeypatch.setattr(c,'run',run)
    conn.pull(tmp_path)
    assert json.loads((tmp_path/'progress.json').read_text())['requests']==512


def test_35_cannot_follow_an_admitted_but_failed_prefill(monkeypatch,tmp_path):
    import json
    from types import SimpleNamespace
    from src import phase6_vast_control as c
    evidence=tmp_path/'evidence';evidence.mkdir();ledger=evidence/'ledger.json'
    ledger.write_text(json.dumps(dict(attempts=[],completed_stages=['3.8-main'],
        stage_decisions={'3.8-main':{'decision':'run'},'3.8-prefill':{'decision':'run'}})))
    requests=tmp_path/'requests.json';requests.write_text(json.dumps({'models':{'3.5':{'requests':[]}}}))
    monkeypatch.setattr(c,'LEDGER',ledger);monkeypatch.setattr(c,'EVIDENCE',evidence)
    monkeypatch.setattr(c,'instances',lambda:[])
    monkeypatch.setattr(c,'cli',lambda *a,**kw:pytest.fail('no provider call allowed after incomplete prior stage'))
    with pytest.raises(RuntimeError,match='earlier stages'):
        c.execute(SimpleNamespace(model='3.5',requests=requests))


def test_smoke_off_closer_is_diagnostic_not_gate(tmp_path):
    import json
    from src.phase6_vast_control import smoke_protocol
    rows=[dict(id=str(i),thinking=False,output_token_ids=[42,248069,43],
               output_tokens=3,finish_reason='stop') for i in range(8)]
    (tmp_path/'chunk.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=smoke_protocol(tmp_path,dict(chunks=[dict(path='chunk.jsonl')]))
    assert result['off_replies_with_thought_boundaries']==8
    assert result['violation_scored'] is False
