"""One owned Vast instance, separate $1.50 ledger and independent deadline guard."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time
from explore.rlprompt2.run import HERE, ROOT, save
from src.phase6_vast_control import cli, account, instances, Connection, IMAGE

OUT=HERE/'results'
LABEL='grader-rlprompt2-20261021'
PROTECTED={57125990,57136341}
CAP=1.50
RESERVE=.15

def owned():
    return [r for r in instances() if r.get('label')==LABEL and r['id'] not in PROTECTED]

def destroy():
    for r in owned():
        assert r['id'] not in PROTECTED and r['label']==LABEL
        cli('destroy','instance',r['id'],'--yes',mutation=True)
    left=owned()
    if left: raise RuntimeError('owned rental destruction unresolved')

def guard():
    state=OUT/'guard.json'
    while True:
        s=json.loads(state.read_text())
        if s.get('finished'): return
        if time.time()>=s['deadline']:
            destroy()
            save(OUT/'guard-fired.json',dict(unix=time.time(),owned_remaining=owned()))
            return
        time.sleep(min(5,max(.1,s['deadline']-time.time())))

def pull(conn, key):
    # Pull only current experiment files; no operation on the lead's host.
    local=OUT/key
    local.mkdir(exist_ok=True)
    command='tar czf - -C /rlprompt2/explore/rlprompt2/results '+shlex.quote(key)
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','StrictHostKeyChecking=accept-new',
        '-p',str(conn.port),'root@'+conn.host,command],capture_output=True,timeout=55,check=True)
    import io
    with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:gz') as tar:
        for m in tar.getmembers():
            if m.name.startswith('/') or '..' in Path(m.name).parts or m.issym() or m.islnk():
                raise ValueError('unsafe archive member')
        tar.extractall(OUT,filter='data')

def execute(keys=('3.8','3.5','gemma'), accounting_file='execution.json', prior_spend=0):
    ledger_path=OUT/accounting_file
    limit=CAP-prior_spend
    if ledger_path.exists(): raise RuntimeError('refuse ledger reset')
    initial=account()
    assert float(initial['credit'])>=limit
    prior=instances()
    assert not owned()
    offers=cli('search','offers','gpu_ram>=48 num_gpus=1 rentable=true dph<=0.85',
               '--storage',140,'--limit',150,'--order','dph_total')
    choices=[]
    for r in offers:
        if (r['gpu_name'] not in ('RTX PRO 5000','RTX 6000Ada','L40S') or r['inet_down_cost']>.0001
            or r['reliability']<.98 or r['verification']!='verified' or r['cuda_max_good']<13.0
            or r['disk_space']<140 or r['cpu_arch']!='amd64'): continue
        hourly=float(r['dph_base'])+140*float(r['storage_cost'])/720
        network=160*float(r['inet_down_cost'])+4*float(r['inet_up_cost'])
        if hourly>0.85 or network>(.55 if keys==('32',) else .02): continue
        choices.append((hourly+network,r,hourly,network))
    if not choices: raise RuntimeError('no low-transfer native-FP8 offer; no spend')
    _,offer,hourly,network=min(choices,key=lambda x:(x[1]['gpu_name']!='RTX PRO 5000',x[0]))
    # Conservative full schedule: 30 min initial image/setup, 15 min/model load,
    # 10 min/model generation, 2 min retrieval/teardown. Gate uses timing only.
    projected=hourly*(1800+len(keys)*900+len(keys)*(1200 if keys==('32',) else 600)+120)/3600+network
    if projected+RESERVE>limit: raise RuntimeError('required Qwen plus optional Gemma initial projection exceeds cap')
    ledger=dict(cap_usd=CAP,initial_account=initial,initial_instances=prior,
        label=LABEL,offer={k:offer[k] for k in ('id','machine_id','gpu_name','gpu_ram','dph_base','storage_cost','inet_down_cost','inet_up_cost')},
        hourly_usd=hourly,network_envelope_usd=network,projected_usd=projected,
        network_envelope=dict(download_gb=160,upload_gb=4),conservative_usd=prior_spend,prior_spend_usd=prior_spend,selected_models=list(keys),
        decisions=['One owned instance. Qwen 360 replies required; Gemma 160 replies conditional on outcome-blind timing gates.'], models={})
    if keys==('32',):
        ledger['decisions']=['Qwen3-32B reintroduced after measured primary costs: official FP8 checkpoint on a 48 GB rental; Phase 4 BF16 weights are not loaded.']
    save(ledger_path,ledger)
    start=time.time()
    deadline=start+(limit-RESERVE-network)*3600/hourly
    save(OUT/'guard.json',dict(deadline=deadline,finished=False))
    log=(OUT/'guard.log').open('ab')
    gp=subprocess.Popen([sys.executable,'-m','explore.rlprompt2.control','guard'],cwd=ROOT,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    ledger.update(started_unix=start,deadline_unix=deadline,guard_pid=gp.pid)
    save(ledger_path,ledger)
    conn=None; key=None
    try:
        reply=cli('create','instance',offer['id'],'--image',IMAGE,'--disk',140,'--label',LABEL,
                  '--ssh','--direct','--cancel-unavail','--env','-e HF_HOME=/hf -e VLLM_USE_FLASHINFER_SAMPLER=0',mutation=True)
        save(OUT/'create.json',{k:reply[k] for k in ('success','new_contract') if k in reply})
        assert reply.get('success') is True and reply.get('new_contract') not in PROTECTED
        ledger['instance_id']=reply['new_contract']
        save(ledger_path,ledger)
        while time.time()<min(start+1800,deadline-120):
            state=owned()
            if state and state[0].get('ssh_port'):
                possible=Connection(state[0])
                try:
                    receipt=json.loads(possible.ssh("python3 -c 'import json,torch,vllm;print(json.dumps(dict(vllm=vllm.__version__,torch=torch.__version__,gpu=torch.cuda.get_device_name(0))))'",timeout=45))
                    assert receipt['vllm']=='0.32.0'
                    save(OUT/'runtime.json',receipt)
                    conn=possible
                    break
                except (subprocess.SubprocessError,json.JSONDecodeError): pass
            print(json.dumps(dict(stage='waiting_image_ssh',seconds=round(time.time()-start))),flush=True)
            time.sleep(10)
        if conn is None: raise RuntimeError('setup exceeded allowance')
        bundle=OUT/'bundle.tar.gz'
        with tarfile.open(bundle,'w:gz') as tar:
            for name in ('src','prompts'):
                for p in (ROOT/name).rglob('*'):
                    if p.is_file() and '__pycache__' not in p.parts: tar.add(p,arcname=str(p.relative_to(ROOT)))
            p=ROOT/'explore/rlprompt/run.py';tar.add(p,arcname=str(p.relative_to(ROOT)))
            for name in ('run.py','templates.json','results/inputs.json'):
                p=HERE/name;tar.add(p,arcname=str(p.relative_to(ROOT)))
        assert conn is not None and ledger['instance_id'] not in PROTECTED
        conn.ssh('mkdir -p /rlprompt2')
        conn.put(bundle,'/rlprompt2/bundle.tar.gz')
        conn.ssh('cd /rlprompt2 && tar xzf bundle.tar.gz')
        for key in keys:
            elapsed=time.time()-start
            # Remaining model allowance adapts to elapsed cost, never outcomes.
            remaining_models=len(keys)-keys.index(key)
            remaining_seconds=remaining_models*1500+120
            projection=(elapsed+remaining_seconds)*hourly/3600+network
            if projection+RESERVE>limit:
                ledger['decisions'].append(f'{key} not started: updated timing projection ${projection:.4f} plus reserve exceeds cap.')
                if key!='gemma': raise RuntimeError('required model cannot fit safely')
                break
            ledger['models'][key]=dict(started_unix=time.time(),projection_usd=projection)
            save(ledger_path,ledger)
            seconds=max(1,int(deadline-time.time()-90))
            command=['timeout','-s','KILL',str(seconds),'python3','-m','explore.rlprompt2.run','worker',key]
            launch="import subprocess,os;f=open('/rlprompt2/"+key+".log','ab');p=subprocess.Popen("+repr(command)+",cwd='/rlprompt2',env=dict(os.environ,HF_HOME='/hf',VLLM_USE_FLASHINFER_SAMPLER='0'),stdout=f,stderr=f,stdin=subprocess.DEVNULL,start_new_session=True);print(p.pid)"
            pid=int(conn.ssh('python3 -c '+shlex.quote(launch)))
            ledger['models'][key]['remote_pid']=pid
            save(ledger_path,ledger)
            gated=False
            while time.time()<deadline-90:
                status=json.loads(conn.ssh('python3 -c '+shlex.quote(
                    "import json,pathlib;d=pathlib.Path('/rlprompt2/explore/rlprompt2/results/"+key+"');print(json.dumps(dict(exists=d.exists(),done=(d/'done.json').exists(),first=(d/'chunk-0.done.json').exists())))")))
                if status['exists']: pull(conn,key)
                if status['first'] and not gated:
                    first=json.loads((OUT/key/'chunk-0.done.json').read_text())
                    elapsed=time.time()-start
                    # Small-batch timing extrapolation with 50% margin.
                    rest=1.5*first['seconds']*((360 if key=='3.8' else 160)-first['n'])/first['n']
                    later=(len(keys)-1-keys.index(key))*1500
                    projection=(elapsed+rest+later+120)*hourly/3600+network
                    decision='run' if projection+RESERVE<=limit else 'drop'
                    gate=dict(decision=decision,projection_usd=projection,first_batch=first,margin=1.5,outcomes_consulted=False)
                    ledger['models'][key]['gate']=gate
                    save(ledger_path,ledger)
                    conn.ssh('python3 -c '+shlex.quote("import json,pathlib;pathlib.Path('/rlprompt2/explore/rlprompt2/results/"+key+"/continue.json').write_text("+repr(json.dumps(gate))+")"))
                    gated=True
                    if decision=='drop': raise RuntimeError('timing gate does not fit cap')
                if status['done']:
                    ledger['models'][key]['completed']=True
                    save(ledger_path,ledger)
                    # Worker exits after serializing output; wait for process/GPU release.
                    conn.ssh('while kill -0 '+str(pid)+' 2>/dev/null; do s=$(ps -o stat= -p '+str(pid)+'); case "$s" in Z*) break;; esac; sleep 1; done',timeout=30)
                    break
                alive=conn.ssh('kill -0 '+str(pid)+' 2>/dev/null && ps -o stat= -p '+str(pid)+' || true').strip()
                if not alive or alive.startswith('Z'):
                    raise RuntimeError('worker exited early; inspect remote log')
                print(json.dumps(dict(model=key,stage='generating' if gated else 'loading_or_first40',elapsed_seconds=round(time.time()-start),cost_envelope=round((time.time()-start)*hourly/3600+network,4))),flush=True)
                time.sleep(12)
            else: raise RuntimeError('absolute budget deadline')
            conn.ssh('tail -100 /rlprompt2/'+key+'.log > /rlprompt2/explore/rlprompt2/results/'+key+'/worker-tail.log')
            pull(conn,key)
        ledger['status']='completed_selected_models'
    except BaseException as e:
        ledger.update(status='failed',error=str(e) if not isinstance(e,subprocess.SubprocessError) else type(e).__name__)
        raise
    finally:
        if conn and key:
            try:
                conn.ssh('mkdir -p /rlprompt2/explore/rlprompt2/results/'+key+'; tail -100 /rlprompt2/'+key+'.log > /rlprompt2/explore/rlprompt2/results/'+key+'/worker-tail.log')
                pull(conn,key)
            except Exception: ledger['last_pull_failed']=True
        destroy()
        end=time.time()
        ledger.update(ended_unix=end,elapsed_seconds=end-start,
                      rental_usd=(end-start)*hourly/3600+network,
                      conservative_usd=prior_spend+(end-start)*hourly/3600+network,
                      final_account=account(),final_instances=instances(),owned_instances_gone=True)
        save(ledger_path,ledger)
        save(OUT/'guard.json',dict(deadline=deadline,finished=True))
        print(json.dumps(dict(status=ledger['status'],conservative_usd=ledger['conservative_usd'],owned_instances_gone=True)),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='guard': guard()
    else: execute(('3.8','gemma'))
