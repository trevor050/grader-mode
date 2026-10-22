"""Local Round 2 transport, credit snapshots and sanitized artifact retrieval."""
import argparse, hashlib, json, pathlib, shlex, subprocess, time
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'results/raw/pulse2'

def cli(*args):
    p=subprocess.run(['vastai',*args,'--raw'],capture_output=True,text=True,errors="replace",timeout=20)
    if p.returncode:raise RuntimeError('Vast CLI failed: '+str(args))
    return json.loads(p.stdout)

def connection():return json.loads((OUT/'connection.json').read_text())

def ssh(command,timeout=30,check=True):
    c=connection()
    r=subprocess.run(['ssh','-o','ConnectTimeout=6','-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new',
        '-o','UserKnownHostsFile='+str(OUT/'known_hosts'),'-p',str(c['port']),'root@'+c['host'],command],
        capture_output=True,text=True,errors="replace",timeout=timeout)
    if check and r.returncode:raise RuntimeError('SSH failed: '+r.stderr[-500:])
    return r

def scp(sources,dest,pull=False):
    c=connection(); host='root@'+c['host']+':'
    src=[host+s for s in sources] if pull else sources
    target=dest if pull else host+dest
    r=subprocess.run(['scp','-q','-o','ConnectTimeout=8','-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new',
        '-o','UserKnownHostsFile='+str(OUT/'known_hosts'),'-P',str(c['port']),*src,target],capture_output=True,text=True,errors="replace",timeout=90)
    if r.returncode:raise RuntimeError('SCP failed: '+r.stderr[-500:])

def credit(stage):
    clock=json.loads((OUT/'launch_clock.json').read_text());user=cli('show','user');t=time.time()
    estimate=clock['initial_credit']-(t-clock['start'])*clock['rate']/3600-clock['reserve']
    row={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(t)),'stage':stage,'credit':float(user['credit']),
         'conservative_remaining':estimate,'usable_remaining':min(float(user['credit']),estimate),'elapsed_s':t-clock['start']}
    with (OUT/'stage-credit.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    return row

def fetch():
    names=json.loads(ssh("python3 -c \"import pathlib,json;print(json.dumps([p.name for p in pathlib.Path('/workspace/pulse/out').iterdir() if p.is_file()]))\"").stdout)
    if names:scp(['/workspace/pulse/out/'+n for n in names],str(OUT/'out'),pull=True)
    # Verify all copied raw outputs against remote digests before claiming collection.
    remote=json.loads(ssh("python3 -c \"import pathlib,json,hashlib;print(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in pathlib.Path('/workspace/pulse/out').iterdir() if p.is_file()}))\"").stdout)
    # events.jsonl can grow between the copy and hash; check stable experiment files strictly.
    for name,digest in remote.items():
        if name=='events.jsonl':continue
        assert hashlib.sha256((OUT/'out'/name).read_bytes()).hexdigest()==digest,name
    (OUT/'raw-hashes.json').write_text(json.dumps(remote,indent=2)+'\n')
    return names

def send(command):
    row=credit('before_'+command['arm'])
    if row['usable_remaining']<.45:raise RuntimeError('Budget too close to hard stop')
    payload=json.dumps(command)
    ssh('printf %s '+shlex.quote(payload)+' > /workspace/pulse/command.tmp && mv /workspace/pulse/command.tmp /workspace/pulse/command.json')
    with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(command)+'\n')
    return row

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['status','fetch','send','credit']);p.add_argument('payload',nargs='?');a=p.parse_args()
    if a.action=='credit':print(json.dumps(credit(a.payload or 'manual')))
    elif a.action=='fetch':print(json.dumps({'fetched':fetch(),'credit':credit('after_fetch')}))
    elif a.action=='send':print(json.dumps(send(json.loads(a.payload))))
    else:
        r=ssh("test -d /workspace/pulse && { tail -c 1800 /workspace/pulse/run.log 2>/dev/null; tail -c 900 /workspace/pulse/setup.log 2>/dev/null; ls /workspace/pulse/out 2>/dev/null; }",check=False)
        print(r.stdout);print(json.dumps(credit('status')))
