import json
import subprocess
import pytest
from src import phase5_vast_main as m


def test_official_image_and_ssh_launch(tmp_path, monkeypatch):
    p=m.Provider(tmp_path); calls=[]
    monkeypatch.setattr(p,'cli',lambda *a:calls.append(a) or {'success':True,'new_contract':1})
    p.create({'id':123},'test')
    a=calls[0]
    assert m.IMAGE.startswith('vllm/vllm-openai@sha256:')
    assert '--ssh' in a and '--onstart-cmd' in a and '--args' not in a


def test_transient_read_error_backs_off_but_create_is_never_retried(tmp_path,monkeypatch):
    p=m.Provider(tmp_path); calls=[]; waits=[]
    def run(a):
        calls.append(a)
        if len(calls)==1:raise RuntimeError('rate limited')
        return '[]'
    monkeypatch.setattr(p,'run',run)
    monkeypatch.setattr(m.time,'sleep',waits.append)
    assert p.cli('show','instances')==[]
    assert waits==[5]
    calls.clear()
    with pytest.raises(RuntimeError):p.cli('create','instance',1)
    assert len(calls)==1


def test_bootstrap_preserves_image_cuda_stack_and_checks_packages():
    script=m.bootstrap_script()
    assert 'pip install' not in script or 'vllm==' not in script.split('pip install')[1]
    assert '--index-url https://pypi.org/simple' in script
    assert '--no-deps' in script
    assert 'runtime-environment.json' in script
    subprocess.run(['bash','-n'],input=script,text=True,check=True)


def test_cli_pacing_uses_shared_lock(tmp_path,monkeypatch):
    p=m.Provider(tmp_path); waits=[]
    monkeypatch.setattr(m,'CLI_RATE_PATH',tmp_path/'rate')
    monkeypatch.setattr(m.time,'time',lambda:100.)
    monkeypatch.setattr(m.time,'sleep',waits.append)
    p.pace_cli();p.pace_cli()
    assert waits==[5.]


def test_read_backoff_stops_at_operation_deadline(tmp_path,monkeypatch):
    p=m.Provider(tmp_path);p.operation_deadline=102.;calls=[];waits=[]
    monkeypatch.setattr(m.time,'time',lambda:100.)
    monkeypatch.setattr(m.time,'sleep',waits.append)
    monkeypatch.setattr(p,'run',lambda a:calls.append(a) or '')
    with pytest.raises(RuntimeError):p.cli('show','instances')
    assert len(calls)==1 and waits==[]


def test_zero_offer_id_selects_fresh_compatible_lowest_price(tmp_path,monkeypatch):
    p=m.Provider(tmp_path)
    base=dict(gpu_name='RTX 5090',num_gpus=1,gpu_ram=32607,reliability=.995,
              verification='verified',rentable=True,dph_base=.6,storage_cost=.2,
              inet_down_cost=0,inet_up_cost=0,cuda_max_good=13.0)
    monkeypatch.setattr(p,'cli',lambda *a:[base|{'id':1,'cuda_max_good':12.8},base|{'id':2},base|{'id':3,'dph_base':.7}])
    assert p.scan(0)['id']==2
    with pytest.raises(m.BudgetError):p.scan(999)


def test_failed_cli_diagnostics_are_retained_privately_not_in_exception(tmp_path,monkeypatch):
    p=m.Provider(tmp_path)
    monkeypatch.setattr(p,'pace_cli',lambda *a:None)
    def fail(*a,**kw):raise subprocess.CalledProcessError(1,a[0],output='',stderr='HTTP 400 invalid image reference')
    monkeypatch.setattr(m.subprocess,'run',fail)
    with pytest.raises(RuntimeError) as error:p.run(['vastai','create','instance','1'])
    assert 'invalid image' not in str(error.value)
    records=list(tmp_path.glob('provider-error-*.json'))
    assert len(records)==1
    assert json.loads(records[0].read_text())['stderr']=='HTTP 400 invalid image reference'
    assert records[0].stat().st_mode & 0o777 == 0o600


def test_exit_zero_cli_error_preserves_stderr_and_disables_internal_retries(tmp_path,monkeypatch):
    p=m.Provider(tmp_path);calls=[]
    monkeypatch.setattr(p,'pace_cli',lambda *a:None)
    def run(argv,**kw):
        calls.append(argv);return subprocess.CompletedProcess(argv,0,stdout='',stderr='HTTP 429 too frequent')
    monkeypatch.setattr(m.subprocess,'run',run)
    assert p.run(['vastai','show','instances','--raw'])==''
    assert calls[0][-2:]==['--retry','1']
    records=list(tmp_path.glob('provider-response-*.json'))
    assert json.loads(records[0].read_text())['stderr']=='HTTP 429 too frequent'


def test_pacing_cannot_launch_after_deadline(tmp_path,monkeypatch):
    p=m.Provider(tmp_path);p.operation_deadline=102.;calls=[]
    rate=tmp_path/'rate';(tmp_path/'rate-show-instances').write_text('100')
    monkeypatch.setattr(m,'CLI_RATE_PATH',rate)
    monkeypatch.setattr(m.time,'time',lambda:100.)
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:calls.append(a))
    with pytest.raises(RuntimeError):p.run(['vastai','show','instances'])
    assert calls==[]


def test_exact_offer_rejects_incompatible_cuda(tmp_path,monkeypatch):
    p=m.Provider(tmp_path)
    monkeypatch.setattr(p,'cli',lambda *a:[{'id':123,'cuda_max_good':12.8}])
    with pytest.raises(m.BudgetError):p.scan(123)


def test_definitive_create_rejection_is_distinct_from_uncertain_transport(tmp_path,monkeypatch):
    p=m.Provider(tmp_path)
    monkeypatch.setattr(p,'pace_cli',lambda *a:None)
    monkeypatch.setattr(m.subprocess,'run',lambda argv,**kw:subprocess.CompletedProcess(
        argv,0,stdout='',stderr=json.dumps({'error':True,'status_code':400,'msg':'invalid image'})))
    with pytest.raises(m.ProviderCreateRejected):p.cli('create','instance',123)


def test_distinct_api_operations_do_not_delay_fresh_create(tmp_path,monkeypatch):
    p=m.Provider(tmp_path);waits=[]
    monkeypatch.setattr(m,'CLI_RATE_PATH',tmp_path/'rate')
    monkeypatch.setattr(m.time,'time',lambda:100.)
    monkeypatch.setattr(m.time,'sleep',waits.append)
    p.pace_cli(('search','offers'));p.pace_cli(('create','instance'))
    assert waits==[]
    p.pace_cli(('search','offers'))
    assert waits==[5.]


def test_scans_explicitly_exclude_already_rented_offers(tmp_path,monkeypatch):
    p=m.Provider(tmp_path);calls=[]
    monkeypatch.setattr(p,'cli',lambda *a:calls.append(a) or [])
    with pytest.raises(m.BudgetError):p.scan(0)
    assert 'rented=false' in calls[0][2]


def test_detached_launcher_releases_parent_pipes_while_child_runs(tmp_path):
    import os,signal,time
    root=tmp_path/'worker';(root/'src').mkdir(parents=True);(root/'output').mkdir()
    (root/'src/__init__.py').write_text('')
    (root/'src/phase5_vast_main.py').write_text('import time\ntime.sleep(30)\n')
    started=time.monotonic()
    result=subprocess.run(['bash','-c',m.detached_launch_command(root)],capture_output=True,text=True,timeout=3,check=True)
    pid=int(result.stdout.strip())
    try:
        os.kill(pid,0)
        assert time.monotonic()-started < 3
    finally:os.killpg(pid,signal.SIGTERM)


def test_unverified_option_still_rejects_deverified_and_bad_hardware():
    base=dict(gpu_name='RTX 5090',num_gpus=1,gpu_ram=32607,reliability=.995,
              verification='unverified',rentable=True,rented=False,dph_base=.45,storage_cost=.2,
              inet_down_cost=0,inet_up_cost=0)
    with pytest.raises(m.BudgetError):m.offer_costs(base)
    assert m.offer_costs(base,allow_unverified=True)['hourly_usd']<.5
    for change in ({'verification':'deverified'},{'reliability':.98},{'gpu_ram':24000},{'rented':True}):
        with pytest.raises(m.BudgetError):m.offer_costs(base|change,allow_unverified=True)


def test_hardware_probe_requires_one_real_5090():
    good=dict(count=1,name='NVIDIA GeForce RTX 5090',capability=[12,0],memory_mib=32607)
    m.validate_hardware(good)
    for change in ({'count':2},{'name':'RTX 4090'},{'capability':[8,9]},{'memory_mib':24000}):
        with pytest.raises(m.BudgetError):m.validate_hardware(good|change)


def test_auto_scan_skips_previously_failed_machine(tmp_path,monkeypatch):
    p=m.Provider(tmp_path);p.excluded_machine_ids={58482}
    base=dict(gpu_name='RTX 5090',num_gpus=1,gpu_ram=32607,reliability=.995,
              verification='verified',rentable=True,rented=False,dph_base=.5,storage_cost=.2,
              inet_down_cost=0,inet_up_cost=0,cuda_max_good=13.)
    monkeypatch.setattr(p,'cli',lambda *a:[base|{'id':1,'machine_id':58482},base|{'id':2,'machine_id':77}])
    assert p.scan(0)['id']==2


def test_wait_ssh_stops_on_fatal_cdi_error_before_model_deployment(tmp_path,monkeypatch):
    p=m.Provider(tmp_path,label='owned',instance=123)
    monkeypatch.setattr(p,'instances',lambda:[dict(id=123,label='owned',actual_status='created',status_msg='unresolvable CDI devices')])
    with pytest.raises(RuntimeError):p.wait_ssh(m.time.time()+2)
    assert p.host is None
    assert (tmp_path/'provider-startup-error.json').exists()
