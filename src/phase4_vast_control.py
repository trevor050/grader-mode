"""Monitor an already-authorized Phase 4 host, gate 32B, pull and destroy it.

No credentials are read directly or printed. The configured vastai CLI handles
account access. A separate watchdog bounds the lifetime even if SSH stalls.
"""
import argparse
import datetime
import json
from pathlib import Path
import statistics
import subprocess
import time


def projected_total(spent, tokens_per_second, off_tokens, on_tokens, completed_tokens, hourly, *, fallback=False):
    """Conservative remaining 32B allocation, plus delayed-charge reserve."""
    if tokens_per_second <= 0 or min(spent, off_tokens, on_tokens, completed_tokens, hourly) < 0:
        raise ValueError('nonnegative measurements and positive throughput required')
    n = 800 if fallback else 1200
    seconds = max(0, n * (off_tokens + on_tokens) - completed_tokens) / tokens_per_second
    return spent + 1.2 * seconds * hourly / 3600 + .30


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instance', type=int, required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--run-id', default='phase4-main')
    args = p.parse_args()
    local = Path('results/raw') / args.run_id
    ledger = Path('results/phase4/vast-execution.json')
    local.mkdir(parents=True, exist_ok=True)
    remote = f'/raw/{args.run_id}'

    def run(cmd, **kw):
        return subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=40, **kw).stdout

    def ssh(command):
        # Input is always closed, including inside polling loops.
        return run(['ssh','-o','ConnectTimeout=15','-o','BatchMode=yes','-p',str(args.port),
                    'root@'+args.host,command], input='')

    def pull(name, target=None):
        target = target or local/name
        run(['scp','-o','ConnectTimeout=15','-o','BatchMode=yes','-P',str(args.port),
             f'root@{args.host}:{remote}/{name}',str(target)])

    def account():
        u = json.loads(run(['vastai','show','user','--raw']))
        return {k:u[k] for k in ('credit','total_spend')}

    def record(key,value):
        e = json.loads(ledger.read_text());e[key] = value
        ledger.write_text(json.dumps(e,indent=2)+'\n')

    current = '14B'
    launched32 = False
    gate32 = False
    last = None
    errors = 0
    while True:
        try:
            instances = json.loads(run(['vastai','show','instances','--raw']))
            if args.instance not in {i['id'] for i in instances}:
                raise RuntimeError('Phase4 instance disappeared before completed output pull')
            status = json.loads(ssh(f"python3 -c \"import json,pathlib; d=pathlib.Path('{remote}'); print(json.dumps({{'progress':json.loads((d/'Qwen3-{current}.progress.json').read_text()) if (d/'Qwen3-{current}.progress.json').exists() else None,'done':(d/'Qwen3-{current}.metrics.json').exists(),'gate':(d/'Qwen3-{current}.gate.json').exists()}}))\""))
            progress = status['progress']
            if progress != last:
                print(json.dumps({'model':current, 'progress':progress, 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}),flush=True)
                last = progress
            if current == '32B' and progress and not status['gate'] and not gate32:
                pull('Qwen3-32B.jsonl', local/'pilot-32B.jsonl')
                rows = [json.loads(l) for l in (local/'pilot-32B.jsonl').read_text().splitlines()]
                if len(rows) != 256: raise ValueError('expected exactly 256 first-chunk rows')
                means = [statistics.mean(r['output_tokens'] for r in rows if r['thinking']==t) for t in (False,True)]
                rate = progress['output_tokens']/progress['generation_seconds']
                u = account();e = json.loads(ledger.read_text());spent = e['credit_before']-u['credit']
                full = projected_total(spent,rate,*means,progress['output_tokens'],e['instance']['dph_total'])
                reduced = projected_total(spent,rate,*means,progress['output_tokens'],e['instance']['dph_total'],fallback=True)
                action = 'continue' if min(full,reduced) <= 4.50 else 'stop'
                drop = ['boxed'] if full > 4.50 else []
                decision = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pilot=progress,
                    mean_output_tokens_by_thinking=dict(zip(('off','on'),means)),tokens_per_second=rate,
                    spent_at_gate_usd=spent,full_grid_projected_total_usd=full,fallback_projected_total_usd=reduced,
                    action=action,drop_request_types=drop,remaining_runtime_multiplier=1.2,teardown_reserve_usd=.30)
                record('projection_32b',decision)
                gate = local/'gate-32B.json';gate.write_text(json.dumps({'action':action,'drop_request_types':drop}))
                run(['scp','-o','ConnectTimeout=15','-P',str(args.port),str(gate),f'root@{args.host}:{remote}/Qwen3-32B.gate.json'])
                gate32 = True
                print('32B_BUDGET_DECISION '+json.dumps(decision),flush=True)
                if action != 'continue': raise RuntimeError('Neither registered grid fits remaining budget')
            if status['done']:
                pull(f'Qwen3-{current}.jsonl')
                pull(f'Qwen3-{current}.metrics.json')
                pull(f'Qwen3-{current}.progress.json')
                print(f'PULLED_COMPLETE {current}',flush=True)
                if current == '14B':
                    # Separate process and a fresh engine, after the old process
                    # exits and frees its allocations. Model files remain cached.
                    ssh("cd /root/grader-mode; nohup env HF_HOME=/hf HF_XET_HIGH_PERFORMANCE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONUNBUFFERED=1 python3 -m src.vast_generate_4 --requests requests.json --model 32B --run-id "+args.run_id+" > /root/phase4-32B.log 2>&1 < /dev/null & echo $!")
                    launched32 = True;current='32B';last=None
                    record('launched_32b_utc',datetime.datetime.now(datetime.timezone.utc).isoformat())
                else:
                    break
            errors = 0
        except subprocess.SubprocessError as exc:
            errors += 1
            print(f'CONTROL_TRANSIENT_ERROR {errors} {exc}',flush=True)
            if errors >= 5: raise
        time.sleep(30)
    # No postprocessing on the rented host: destroy as soon as both files pull.
    destroyed = run(['vastai','destroy','instance',str(args.instance)],input='y\n')
    remaining = json.loads(run(['vastai','show','instances','--raw']))
    if remaining: raise RuntimeError('Vast instances remain after teardown')
    u = account();e = json.loads(ledger.read_text())
    record('teardown',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),destroy_stdout=destroyed,
            instances_after=remaining,credit_after=u['credit'],total_spend_after=u['total_spend'],
            credit_delta_usd=e['credit_before']-u['credit']))
    print('VAST_DESTROYED '+json.dumps(u),flush=True)


if __name__ == '__main__':
    main()
