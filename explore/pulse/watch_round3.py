"""Independent 45-second budget and progress watchdog. No secret-bearing output."""
import argparse,json,pathlib,subprocess,time
p=argparse.ArgumentParser();p.add_argument('instance',type=int);p.add_argument('--rate',type=float,required=True)
p.add_argument('--credit',type=float,required=True);p.add_argument('--start',type=float,required=True)
p.add_argument('--reserve',type=float,default=.3);p.add_argument('--out',default='results/raw/pulse3');a=p.parse_args()
out=pathlib.Path(a.out);stop=None;last_progress=a.start

def cli(*args):
 r=subprocess.run(['vastai',*args,'--raw'],capture_output=True,text=True,timeout=12)
 if r.returncode:raise RuntimeError('Vast read failed')
 return json.loads(r.stdout)

while True:
 tick=time.time();estimate=a.credit-(tick-a.start)*a.rate/3600-a.reserve
 row=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),instance=a.instance,elapsed_s=tick-a.start,conservative_remaining=estimate)
 try:
  user=cli('show','user');inst=next((r for r in cli('show','instances') if r['id']==a.instance),None)
  row.update(credit=float(user['credit']),state=inst.get('actual_status') if inst else 'absent')
  if not inst:
   with (out/'budget.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps(row),flush=True);break
  if (out/'connection.json').exists():
   c=json.loads((out/'connection.json').read_text())
   r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=4','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+str(out/'known_hosts'),'-p',str(c['port']),'root@'+c['host'],
    "find /workspace/pulse -maxdepth 2 -type f \\( -name '*.log' -o -name heartbeat -o -name '*.jsonl' \\) -printf '%T@\\n' 2>/dev/null | sort -nr | head -1"],capture_output=True,text=True,timeout=8)
   if r.returncode==0 and r.stdout.strip():last_progress=max(last_progress,float(r.stdout.strip()))
  if min(row['credit'],estimate)<.40:stop='budget_floor'
 except Exception as exc:
  row['read_error']=type(exc).__name__
  if estimate<.50:stop='budget_read_failure'
 row['progress_age_s']=tick-last_progress
 if tick-last_progress>900:stop='stall_15min'
 if stop:
  row['stop_reason']=stop
  try:row['destroy_returncode']=subprocess.run(['vastai','destroy','instance',str(a.instance),'--yes'],capture_output=True,timeout=12).returncode
  except Exception as exc:row['destroy_error']=type(exc).__name__
 with (out/'budget.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True);time.sleep(max(0,45-(time.time()-tick)))
