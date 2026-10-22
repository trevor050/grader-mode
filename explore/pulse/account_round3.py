"""Publish only allowlisted execution evidence; never persist account/instance secrets."""
import json,pathlib,time,hashlib,subprocess,re
ROOT=pathlib.Path(__file__).resolve().parents[2];RAW=ROOT/'results/raw/pulse3';DEST=ROOT/'explore/pulse'

def rows(p):return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
def load(p):return json.loads(p.read_text()) if p.exists() else {}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

clock=load(RAW/'launch_clock.json');td=load(RAW/'teardown.json');budget=rows(RAW/'budget.jsonl');events=rows(RAW/'out/events.jsonl')
u=subprocess.run(['vastai','show','user','--raw'],capture_output=True,text=True,timeout=20);u.check_returncode();credit=float(json.loads(u.stdout)['credit'])
i=subprocess.run(['vastai','show','instances','--raw'],capture_output=True,text=True,timeout=20);i.check_returncode();instances=[dict(id=r['id'],state=r.get('actual_status')) for r in json.loads(i.stdout)]
assert not instances,'Do not publish complete teardown while resources remain'
execution=dict(instance=clock['instance'],offer=clock['offer'],image=clock['image'],disk_gb=clock['disk_gb'],
    initial_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(clock['start'])),initial_credit=clock['initial_credit'],
    final_credit=credit,final_credit_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    posted_credit_delta=clock['initial_credit']-credit,billing_status='Provisional posted snapshot; not settled invoice',
    guard_hourly_rate=clock['rate'],network_reserve=clock['reserve'],hard_floor=.40,watchdog_interval_s=45,stall_limit_s=900,
    rental=load(RAW/'rental.json'),**{k:v for k,v in td.items() if k not in ('instances_after','instance')},instances_after=instances,
    stage_credit=rows(RAW/'stage-credit.jsonl'),projections=rows(RAW/'projections.jsonl'),
    commands=rows(RAW/'commands.jsonl'),events=[{k:v for k,v in e.items() if k!='scores'} for e in events],
    guard_stop_events=[r for r in budget if 'stop_reason' in r],
    conservative_time_cost_with_reserve=td.get('wall_s',0)*clock['rate']/3600+clock['reserve'],
    minimum_reported_credit_while_active=min((r['credit'] for r in budget if r.get('state')!='absent' and 'credit' in r),default=None),
    minimum_conservative_remaining_while_active=min((r['conservative_remaining'] for r in budget if r.get('state')!='absent'),default=None),
    guard_update=load(RAW/'guard-update.json'),controller_error=load(RAW/'controller-error.json'),
    remote_errors=[load(p) for p in (RAW/'out').glob('*.error.json')],
    deployed_hashes=load(RAW/'deployed-hashes.json'),raw_hashes=load(RAW/'raw-hashes.json'),
    log_hashes={n:sha(RAW/n) for n in ['setup.log','run.log','generation_config.json'] if (RAW/n).exists()},
    protocol_commit='01e53c6',implementation_commit='950ec47',
    transport_notes=['Initial SSH background-launch acknowledgement timed out; setup was verified progressing, not relaunched.'],
    deviations=[],verification=dict(cpu_contract_tests=10,full_grid_validation='pending',secret_scan='pending'))
# Chronology is essential: no steered job can precede the threshold event.
frozen=next((e['utc'] for e in events if e['event']=='threshold_frozen'),None)
steered=[e for e in events if e['event']=='closed_loop_start' and not e.get('unsteered',False)]
assert not steered or (frozen and all(e['utc']>=frozen for e in steered))
execution['threshold_frozen_utc']=frozen
for p in (RAW/'out').iterdir():
 if p.is_file():
  contents=p.read_bytes()
  assert not re.search(rb'(?:sk-[A-Za-z0-9_-]{16,}|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY|hf_[A-Za-z0-9]{20,})',contents),p.name
execution['verification']['secret_scan']='passed for output artifacts'
(DEST/'execution3.json').write_text(json.dumps(execution,indent=2)+'\n')
print(json.dumps({k:execution[k] for k in ['instance','initial_credit','final_credit','instances_after','reason','threshold_frozen_utc']}))
