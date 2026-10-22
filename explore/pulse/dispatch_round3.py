"""Sequential priority dispatcher; pull and check credit after every complete job."""
import json,pathlib,time,shlex,subprocess
from control_round3 import OUT,ssh,fetch,send,credit,cli,scp

def wait_marker(arm=None):
 marker='ready' if arm is None else arm+'.done.json'
 while True:
  safe=credit('wait_'+str(arm))
  if safe['usable_remaining']<.40:raise RuntimeError('Hard budget floor')
  i=json.loads((OUT/'launch_clock.json').read_text())['instance']
  if not any(r['id']==i for r in cli('show','instances')):raise RuntimeError('Instance absent')
  code="import pathlib,json;p=pathlib.Path('/workspace/pulse/out');print(json.dumps({'done':(p/"+repr(marker)+").exists(),'errors':[x.name for x in p.glob('*.error.json')]}))"
  d=json.loads(ssh('python3 -c '+shlex.quote(code)).stdout)
  if d['errors']:raise RuntimeError('Job failed: '+str(d['errors']))
  if d['done']:return
  time.sleep(20)

def job(command):
 p=OUT/'out'/(command['arm']+'.done.json')
 if p.exists():return json.loads(p.read_text())
 send(command);wait_marker(command['arm']);fetch();c=credit('after_'+command['arm'])
 d=json.loads(p.read_text());print(json.dumps({'arm':command['arm'],'result':{k:v for k,v in d.items() if k!='roc'},'credit':c}),flush=True)
 return d

def affordable(arm,seconds):
 c=credit('projection_'+arm);cost=(seconds+600)*1.1/3600
 r=dict(arm=arm,projected_seconds=seconds,teardown_margin_s=600,projected_usd=cost,usable_remaining=c['usable_remaining'],allowed=cost<c['usable_remaining']-.40)
 with (OUT/'projections.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
 print(json.dumps({'projection':r}),flush=True);return r['allowed']

def finish(reason):
 errors=[]
 try:
  fetch();scp(['/workspace/pulse/'+n for n in ['setup.log','run.log','generation_config.json']],str(OUT),pull=True)
 except Exception as exc:errors.append(dict(type=type(exc).__name__,message=str(exc)))
 clock=json.loads((OUT/'launch_clock.json').read_text());i=clock['instance'];before=credit('before_destroy')
 destroy_time=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
 rc=subprocess.run(['vastai','destroy','instance',str(i),'--yes'],capture_output=True,timeout=30).returncode
 for _ in range(12):
  remaining=[dict(id=r['id'],state=r.get('actual_status')) for r in cli('show','instances')]
  if not any(r['id']==i for r in remaining):break
  time.sleep(3)
 else:raise RuntimeError('Instance still present')
 d=dict(instance=i,destroyed_utc=destroy_time,destroy_returncode=rc,verified_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        instances_after=remaining,before=before,after=credit('after_destroy'),wall_s=time.time()-clock['start'],reason=reason,retrieval_errors=errors)
 (OUT/'teardown.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d),flush=True)


def run():
 wait_marker();fetch()
 cal=job(dict(arm='calibration',kind='calibration'))
 # Freeze and retrieve threshold locally before submitting the first steered command.
 p=OUT/'out/monitor.json';assert p.exists();(OUT/'threshold-frozen-locally.json').write_text(p.read_text())
 if not affordable('closed_loop_target',1800):return 'target_projection_budget_stop'
 target=job(dict(arm='closed_loop_target',kind='closed_loop',spec='q38-T3E3.json'))
 # Conservative later-arm estimate scales observed target duration; never shrink grids.
 estimate=max(300,target['seconds']*1.5)
 if not affordable('closed_loop_random',estimate):return 'random_projection_budget_stop'
 job(dict(arm='closed_loop_random',kind='closed_loop',spec='q38-rand_T3E3.json'))
 for prefix,cond in [('student','student_tutor'),('plain','plain_nosys')]:
  arm=prefix+'_closed_loop_target'
  if not affordable(arm,estimate):return arm+'_projection_budget_stop'
  result=job(dict(arm=arm,kind='closed_loop',cond=cond,spec='q38-T3E3.json'))
  if result['push_on_chunks']==0 and affordable(prefix+'_chunked_baseline',result['seconds']*1.25):
   job(dict(arm=prefix+'_chunked_baseline',kind='closed_loop',cond=cond,unsteered=True,spec='q38-T3E3.json'))
 if affordable('closed_loop_target_reminder',max(600,target['seconds']*2)):
  job(dict(arm='closed_loop_target_reminder',kind='closed_loop',spec='q38-T3E3.json',prefill=True))
 else:return 'complete_core_reminder_budget_skip'
 return 'all_priorities_complete'

if __name__=='__main__':
 reason='controller_error'
 try:reason=run()
 except Exception as exc:
  (OUT/'controller-error.json').write_text(json.dumps(dict(type=type(exc).__name__,message=str(exc)))+'\n')
  print(json.dumps(dict(error=type(exc).__name__,message=str(exc))),flush=True)
 finally:finish(reason)
