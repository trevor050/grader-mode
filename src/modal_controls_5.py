"""Lead-authorized 800-row Nemotron continuation; dry by default.

New fixed $1.40 authorization includes setup and teardown, with the existing
$0.30 reserve. Only the runtime projection margin is relaxed (1.25 -> 1.0);
its dollar projection retains 1.25. Uses the successful compiler-complete
image, unchanged full-batch worker and historical outcome-blind timing.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from decimal import Decimal
import modal
from src import modal_second_family_5 as base
from src.models_5 import NEMOTRON
from src.phase5_control import ROOT, BudgetError, atomic_json, safe_id, positive
from src.modal_app_5 import modal_cli, stop_and_verify, RESULT_VOLUME

CAP=1.40
RESERVE=.30
RATE=base.RATE
AUTH_ID='phase5-nemotron-controls-140-8219501'
AUTH_PATH=ROOT/'results/phase5/modal-controls-authorization.json'
PRIOR_GATES=ROOT/'results/phase5/nemotron-execution-modal-gates.json'
APP_NAME='grader-mode-phase5-nemotron-controls'
app=modal.App(APP_NAME)


def authorization():
    return base.Authorization(AUTH_PATH,cap=CAP,authorization_id=AUTH_ID)


def free_credit(billing, *, minimum=CAP):
    if Decimal(billing['billed_cost'])!=0:
        raise BudgetError('Modal billed cost must remain zero')
    free=Decimal('30')-Decimal(billing['metered_cost_breakdown']['ephemeral_apps'])
    if free<Decimal(str(minimum)):
        raise BudgetError('whole remaining cap does not fit free compute credit')
    return float(free)


def controls_gate(prior, *, elapsed_seconds,prior_charged_usd,seconds_left):
    if (prior.get('stage')!='h20_controls' or prior.get('count')!=800
            or prior.get('outcomes_consulted') is not False or prior.get('runtime_margin')!=1.25):
        raise BudgetError('complete outcome-blind historical controls projection required')
    seconds=positive(prior['projected_seconds'],'measured projected time')
    cost=positive(prior['projected_usd'],'measured projected dollars')
    if abs(cost-seconds*RATE)>1e-8 or seconds<15:
        raise BudgetError('historical timing/cost bound mismatch')
    # Keep the original 25% dollar projection. Relax only runtime headroom.
    runtime=(seconds-15)/1.25+15
    available=max(0,CAP-RESERVE-positive(prior_charged_usd,'prior exposure',zero=True)
                  -positive(elapsed_seconds,'elapsed',zero=True)*RATE)
    left=positive(seconds_left,'remaining deadline',zero=True)
    return dict(stage='h20_controls',count=800,decision='run' if cost<=available and runtime<=left else 'defer',
                projected_usd=cost,projected_seconds=seconds,runtime_projected_seconds=runtime,
                dollar_margin=1.25,runtime_margin=1.0,available_usd=available,seconds_left=left,
                reserve_usd=RESERVE,shortfall_usd=max(0,cost-available,(runtime-left)*RATE),
                outcomes_consulted=False,dropped_cells=[])


@app.function(image=base.image,gpu='RTX-PRO-6000',volumes={'/hf':base.hf_cache,'/phase5':base.results},
              cpu=(4,4),memory=(32768,32768),timeout=1100,startup_timeout=180,
              min_containers=0,buffer_containers=0,max_containers=1,retries=0,
              single_use_containers=True,scaledown_window=2)
def controls_gpu(payload:dict,channel:modal.Dict):
    return base.supervise_worker(payload,channel)


def watchdog(attempt_id):
    ledger=authorization()
    while True:
        a=next(t for t in ledger.snapshot()['attempts'] if t['attempt_id']==attempt_id)
        if a['teardown_verified']:return
        if time.time()<a['work_deadline_unix']:
            time.sleep(min(1,a['work_deadline_unix']-time.time()));continue
        ids=[a['app_id']] if a.get('app_id') else []
        if not ids:
            try:
                ids=[r['app_id'] for r in json.loads(modal_cli('app','list','--json',timeout=5))
                     if r.get('state') not in ('stopped','disabled') and r.get('description')==APP_NAME]
            except Exception:pass
        zero=bool(ids);evidence=[]
        for app_id in ids:
            ok,detail=stop_and_verify(app_id);zero=zero and ok;evidence.append(detail)
        ledger.finish(attempt_id,zero,evidence=evidence)
        if zero or time.time()>=a['hard_deadline_unix']:return
        time.sleep(1)


def execute(run_id):
    safe_id(run_id);root=ROOT/'results/raw'/run_id;root.mkdir(parents=True,exist_ok=False)
    billing=json.loads(modal_cli('billing','summary','--json'));headroom=free_credit(billing)
    atomic_json(root/'billing-before.json',billing)
    apps=json.loads(modal_cli('app','list','--json'));containers=json.loads(modal_cli('container','list','--json'))
    if containers or any(a['state'] not in ('stopped','disabled') for a in apps):
        raise BudgetError('active Modal resources present; no resource will be touched')
    prior=[g for g in json.loads(PRIOR_GATES.read_text()) if g['stage']=='h20_controls']
    if len(prior)!=1:raise BudgetError('exact measured controls gate required')
    prior=prior[0];controls_gate(prior,elapsed_seconds=0,prior_charged_usd=0,seconds_left=980)
    from src.vast_generate_5 import build
    rows,manifest=build('nemotron',200);base.stage_rows(rows)
    frozen=base.source_hashes()|{'src/modal_controls_5.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    ledger=authorization();attempt_id=run_id+'-controls';ticket=ledger.begin(attempt_id)
    out=root/attempt_id;out.mkdir();before_id=app.app_id;known={a['app_id'] for a in apps}
    known|={a['app_id'] for a in ledger.snapshot()['attempts'] if a.get('app_id')}
    payload=dict(ticket,attempt_stage='controls',authorization_id=AUTH_ID,model=NEMOTRON,
                 requests=rows,manifest=manifest,source_hashes=frozen,controls_only=True,
                 resident_sequences=256,enforce_eager=False,fallback_provenance=None)
    atomic_json(out/'launch.json',payload)
    with (out/'watchdog.log').open('ab') as log:
        guard=subprocess.Popen([sys.executable,'-m','src.modal_controls_5','--watchdog',attempt_id],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,close_fds=True)
    ledger.note(attempt_id,watchdog_pid=guard.pid,source_hashes=frozen,billing_before=billing,
                free_credit_before_usd=headroom,projection_source_sha256=hashlib.sha256(PRIOR_GATES.read_bytes()).hexdigest())
    call=None;app_id=None;zero=False;detail=None;result=None;error=None;gate=None
    def terminate(*_):raise KeyboardInterrupt('controls controller terminated')
    previous=signal.signal(signal.SIGTERM,terminate)
    try:
        with modal.enable_output(),app.run(name=APP_NAME):
            app_id=app.app_id;ledger.bind(attempt_id,app_id)
            if time.time()>=ticket['work_deadline_unix']-60:raise TimeoutError('setup consumed budget')
            with modal.Dict.ephemeral() as channel:
                call=controls_gpu.spawn(payload,channel);ledger.note(attempt_id,function_call_id=call.object_id)
                smoke=base.wait_done(channel,call,'smoke',ticket['work_deadline_unix'])
                if smoke is not None:
                    now=time.time();current=json.loads(modal_cli('billing','summary','--json'))
                    free_credit(current,minimum=max(0,CAP-ticket['prior_charged_usd']-(now-ticket['started_unix'])*RATE))
                    atomic_json(out/'billing-before-controls.json',current)
                    gate=controls_gate(prior,elapsed_seconds=time.time()-ticket['started_unix'],
                        prior_charged_usd=ticket['prior_charged_usd'],seconds_left=max(0,ticket['work_deadline_unix']-time.time()-20))
                    gate['decided_unix']=time.time();atomic_json(out/'control-gate.json',gate)
                    print('MODAL_CONTROLS_GATE '+json.dumps(gate),flush=True)
                    if gate['decision']=='run':
                        channel['command:h20_controls']=dict(action='run',attempt_id=attempt_id,count=800,
                            work_deadline_unix=ticket['work_deadline_unix'],stage_deadline_unix=ticket['work_deadline_unix'])
                    else:channel['stop']=True
                while time.time()<ticket['work_deadline_unix']-5:
                    try:result=call.get(timeout=0);break
                    except TimeoutError:time.sleep(.5)
                if result is None:raise TimeoutError('absolute controls deadline')
    except BaseException as exc:error=f'{type(exc).__name__}: {exc}'
    finally:
        if call is not None:
            try:call.cancel(terminate_containers=True)
            except Exception as exc:ledger.note(attempt_id,cancel_error=str(exc))
        if app_id is None:
            sdk_id=app.app_id
            app_id=sdk_id if sdk_id and sdk_id!=before_id and sdk_id not in known else None
            if app_id is None:
                try:
                    allocated=[a['app_id'] for a in json.loads(modal_cli('app','list','--json'))
                        if a['app_id'] not in known and a.get('description')==APP_NAME]
                    if len(allocated)==1:app_id=allocated[0]
                except Exception:pass
            if app_id:ledger.bind(attempt_id,app_id)
        if app_id:zero,detail=stop_and_verify(app_id)
        settled=ledger.finish(attempt_id,zero,evidence=detail)
        try:modal_cli('volume','get',RESULT_VOLUME,attempt_id,str(out),'--force',timeout=90);pulled=True
        except Exception as exc:pulled=False;error=error or f'artifact recovery failed: {exc}'
        summary=dict(result or {},controller_error=error,authorization_attempt=settled,gate=gate,
                     artifacts_pulled=pulled,teardown=detail,zero_phase5_app_verified=zero)
        atomic_json(root/'execution.json',summary);signal.signal(signal.SIGTERM,previous)
    if not zero or settled['status']!='settled':raise BudgetError('controls teardown unresolved; full reservation retained')
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true');mode.add_argument('--dry-run',action='store_true')
    mode.add_argument('--watchdog',help=argparse.SUPPRESS);parser.add_argument('--run-id')
    args=parser.parse_args(argv)
    if args.watchdog:watchdog(safe_id(args.watchdog));return
    if args.execute:
        if not args.run_id:parser.error('--execute requires fresh --run-id')
        print(json.dumps(execute(args.run_id),indent=2));return
    print(json.dumps(dict(execute=False,cap_usd=CAP,reserve_usd=RESERVE,rate_bound=RATE,
        authorization_id=AUTH_ID,authorization_path=str(AUTH_PATH),gpu='RTX-PRO-6000',
        smoke=8,controls=800,resident_sequences=256,chunk_size=256,enforce_eager=False,
        dollar_margin=1.25,runtime_margin=1.0,source_projection=str(PRIOR_GATES),paid_calls=False),indent=2))


if __name__=='__main__':main()
