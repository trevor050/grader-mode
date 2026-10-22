"""Execute only the registered list, with credit checks after every job."""
import json,pathlib,time
from control_round2 import OUT,ssh,fetch,send,credit,cli


def wait_marker(arm=None):
    marker='ready' if arm is None else arm+'.done.json'
    while True:
        safe=credit('wait_'+str(arm))
        if safe['usable_remaining']<.40:raise RuntimeError('Budget hard stop')
        inst=json.loads((OUT/'launch_clock.json').read_text())['instance']
        if not any(r['id']==inst for r in cli('show','instances')):raise RuntimeError('Instance absent')
        code="import pathlib,json;p=pathlib.Path('/workspace/pulse/out');print(json.dumps({'ready':(p/"+repr(marker)+").exists(),'errors':[x.name for x in p.glob('*.error.json')]}))"
        result=json.loads(ssh('python3 -c '+__import__('shlex').quote(code)).stdout)
        if result['errors']:raise RuntimeError('Remote job failure: '+str(result['errors']))
        if result['ready']:return
        time.sleep(25)


def job(command):
    start=time.time();before=send(command);time.sleep(2.5);wait_marker(command['arm']);fetch();after=credit('after_'+command['arm'])
    d=json.loads((OUT/'out'/(command['arm']+'.done.json')).read_text())
    print(json.dumps(d|{'wall_s':time.time()-start,'credit':after['credit'],'conservative_remaining':after['conservative_remaining']}),flush=True)
    if after['usable_remaining']<.40:raise RuntimeError('Budget hard stop after job')
    return d


def registered():
    wait_marker();fetch()
    gates=[json.loads(l) for l in (OUT/'out/gates.jsonl').read_text().splitlines()]
    print(json.dumps({'gates':{c:sum(r['gate'] for r in gates if r['cond']==c) for c in ['rl_tutor','plain_nosys','student_tutor']}}),flush=True)
    for arm,k,spec in [('target_k64',64,'q38-T3E3.json'),('random_k64',64,'q38-rand_T3E3.json'),
                       ('target_k16',16,'q38-T3E3.json'),('target_k256',256,'q38-T3E3.json'),
                       ('random_k16',16,'q38-rand_T3E3.json'),('random_k256',256,'q38-rand_T3E3.json')]:
        job({'arm':arm,'kind':'pulse','k':k,'spec':spec})
        if arm=='random_k64':
            events=[json.loads(l) for l in (OUT/'out/events.jsonl').read_text().splitlines()]
            durations=[r['seconds'] for r in events if r['event']=='pulse_complete']
            s=credit('first_pair_projection')
            # Upper bound: remaining four pulses at 1.5x slower than the slower observed arm,
            # plus 20 minutes for side effects and transfer/teardown. No shortening of jobs.
            seconds=max(durations)*4*1.5+1200
            cost=seconds*1.1/3600
            projection={'remaining_seconds':seconds,'remaining_usd':cost,'usable_remaining':s['usable_remaining'],'allowed':cost<s['usable_remaining']-.40}
            (OUT/'first-pair-projection.json').write_text(json.dumps(projection,indent=2)+'\n');print(json.dumps({'projection':projection}),flush=True)
            if not projection['allowed']:raise RuntimeError('Remaining-grid projection fails budget reserve')
    for prefix,cond in [('plain','plain_nosys'),('student','student_tutor')]:
        base=prefix+'_baseline';job({'arm':base,'kind':'baseline','cond':cond})
        for k in [16,64,256]:
            for kind,spec in [('target','q38-T3E3.json'),('random','q38-rand_T3E3.json')]:
                job({'arm':f'{prefix}_{kind}_k{k}','kind':'pulse','k':k,'spec':spec,'cond':cond,'baseline_file':base+'.jsonl'})
        job({'arm':prefix+'_gated_reminder','kind':'reminder','cond':cond,'baseline_file':base+'.jsonl'})
    (OUT/'registered-complete.json').write_text(json.dumps(credit('registered_complete'),indent=2)+'\n')
    print('REGISTERED LIST COMPLETE; amendment decision pending',flush=True)

if __name__=='__main__':
    try:registered()
    except Exception as exc:
        (OUT/'controller-error.json').write_text(json.dumps({'type':type(exc).__name__,'message':str(exc)}))
        try:fetch()
        except Exception:pass
        i=json.loads((OUT/'launch_clock.json').read_text())['instance']
        import subprocess
        subprocess.run(['vastai','destroy','instance',str(i),'--yes'],capture_output=True,text=True,timeout=30)
        raise
