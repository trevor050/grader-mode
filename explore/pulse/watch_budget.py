"""Persistent credit/time guard, using only sanitized account snapshots."""
import argparse,json,pathlib,subprocess,time
p=argparse.ArgumentParser();p.add_argument('instance',type=int);p.add_argument('--rate',type=float,required=True);p.add_argument('--credit',type=float,required=True);p.add_argument('--start',type=float,required=True);a=p.parse_args()
out=pathlib.Path('results/raw/pulse');out.mkdir(parents=True,exist_ok=True)
def cli(*args):
    r=subprocess.run(['vastai',*args,'--raw'],capture_output=True,text=True,timeout=40)
    if r.returncode: raise RuntimeError('Vast CLI failed: '+str(args))
    return json.loads(r.stdout)
while True:
    try:
        user=cli('show','user'); insts=cli('show','instances');inst=next((x for x in insts if x['id']==a.instance),None)
        t=time.time(); estimated=a.credit-(t-a.start)/3600*a.rate-.45
        row={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(t)),'elapsed_s':round(t-a.start,1),'credit':user['credit'],'conservative_remaining':estimated,'instance':a.instance,'state':inst.get('actual_status') if inst else 'absent'}
        connection=out/'connection.json'
        if connection.exists() and inst:
            c=json.loads(connection.read_text())
            r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile=results/raw/pulse/known_hosts','-p',str(c['port']),'root@'+c['host'],"find /workspace/pulse -maxdepth 2 -type f \\( -name '*.log' -o -name 'heartbeat' -o -name '*.jsonl' \\) -printf '%T@\\n' 2>/dev/null | sort -nr | head -1"],capture_output=True,text=True,timeout=25)
            if r.returncode==0 and r.stdout.strip():
                row['progress_age_s']=t-float(r.stdout.strip())
        stop=min(float(user['credit']),estimated)<.40 or row.get('progress_age_s',t-a.start)>900
        if stop and inst:
            row['stop_reason']='budget' if min(float(user['credit']),estimated)<.40 else 'stall_15min'
            d=subprocess.run(['vastai','destroy','instance',str(a.instance),'--yes'],capture_output=True,text=True,timeout=45)
            row['destroy_returncode']=d.returncode
        with (out/'budget.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if not inst or stop:break
    except Exception as e:
        # Conservatively tear down if the billing API cannot be read near the hard wall.
        print(type(e).__name__,str(e),flush=True)
        if a.credit-(time.time()-a.start)/3600*a.rate-.45<.50:
            subprocess.run(['vastai','destroy','instance',str(a.instance),'--yes'],timeout=45)
            break
    time.sleep(45)
