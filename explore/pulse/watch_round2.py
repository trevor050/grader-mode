"""Independent 45-second watchdog; no credentials in output or on the instance."""
import argparse,json,pathlib,subprocess,time
p=argparse.ArgumentParser()
p.add_argument('instance',type=int);p.add_argument('--rate',type=float,required=True)
p.add_argument('--credit',type=float,required=True);p.add_argument('--start',type=float,required=True)
p.add_argument('--reserve',type=float,default=.30);p.add_argument('--out',default='results/raw/pulse2')
a=p.parse_args();out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)

def cli(*args):
    r=subprocess.run(['vastai',*args,'--raw'],capture_output=True,text=True,timeout=12)
    if r.returncode:raise RuntimeError('Vast read failed')
    return json.loads(r.stdout)

stop_reason=None
while True:
    tick=time.time();t=tick
    estimate=a.credit-(t-a.start)/3600*a.rate-a.reserve
    row={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(t)),'elapsed_s':t-a.start,'conservative_remaining':estimate,'instance':a.instance}
    try:
        user=cli('show','user');inst=next((r for r in cli('show','instances') if r['id']==a.instance),None)
        row.update(credit=user['credit'],state=inst.get('actual_status') if inst else 'absent')
        if inst is None:
            with (out/'budget.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True);break
        connection=out/'connection.json'
        if connection.exists():
            c=json.loads(connection.read_text())
            r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=4','-o','StrictHostKeyChecking=accept-new',
                '-o','UserKnownHostsFile='+str(out/'known_hosts'),'-p',str(c['port']),'root@'+c['host'],
                "find /workspace/pulse -maxdepth 2 -type f \\( -name '*.log' -o -name 'heartbeat' -o -name '*.jsonl' \\) -printf '%T@\\n' 2>/dev/null | sort -nr | head -1"],
                capture_output=True,text=True,timeout=8)
            if r.returncode==0 and r.stdout.strip():row['progress_age_s']=t-float(r.stdout.strip())
        if min(float(user['credit']),estimate)<.40:stop_reason='budget'
        if row.get('progress_age_s',t-a.start)>900:stop_reason='stall_15min'
    except Exception as exc:
        row['read_error']=type(exc).__name__
        if estimate<.50:stop_reason='budget_api_unavailable'
        if t-a.start>900 and not (out/'connection.json').exists():stop_reason='setup_stall'
    if stop_reason:
        row['stop_reason']=stop_reason
        try:
            r=subprocess.run(['vastai','destroy','instance',str(a.instance),'--yes'],capture_output=True,text=True,timeout=12)
            row['destroy_returncode']=r.returncode
        except Exception as exc:row['destroy_error']=type(exc).__name__
        # Keep retrying teardown until show instances verifies absence.
    with (out/'budget.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
    time.sleep(max(0,45-(time.time()-tick)))
