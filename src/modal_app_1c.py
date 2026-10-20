"""Separate ephemeral Phase 1c app; never touches Phase 1/1b apps."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import modal
from src.modal_app import image, hf_cache, raw_volume, generate, pull_file, render_prompt
from src.models_1c import MODELS_1C

app = modal.App('grader-mode-phase1c')


@app.function(image=image, volumes={'/hf': hf_cache}, timeout=600,
              single_use_containers=True, min_containers=0, buffer_containers=0, scaledown_window=2)
def preflight():
    import hashlib
    from transformers import AutoConfig, AutoTokenizer
    from vllm.model_executor.models import ModelRegistry
    from src.modal_app import package_versions
    result = {'versions': package_versions(), 'models': {}}
    for model, spec in MODELS_1C.items():
        config = AutoConfig.from_pretrained(model, revision=spec['revision'])
        assert config.architectures == ['Olmo3ForCausalLM']
        assert config.architectures[0] in ModelRegistry.get_supported_archs()
        tokenizer = AutoTokenizer.from_pretrained(model, revision=spec['revision'])
        template = tokenizer.get_chat_template()
        assert hashlib.sha256(template.encode()).hexdigest() == spec['chat_template_sha256']
        tutor = render_prompt(tokenizer, {'phase':'phase1c','arm':'tutor','messages':[
            {'role':'system','content':'TUTOR_SENTINEL'}, {'role':'user','content':'PROBLEM_SENTINEL'}]})
        ceiling = render_prompt(tokenizer, {'phase':'phase1c','arm':'ceiling','messages':[
            {'role':'user','content':'PROBLEM_SENTINEL'}]})
        assert tutor.startswith('<|im_start|>system\nTUTOR_SENTINEL')
        assert '<|im_start|>system' not in ceiling
        assert ceiling.endswith('<think>') == (spec['ladder']=='think')
        result['models'][model] = {'revision':spec['revision'],'architecture':config.architectures,
            'template_sha256':spec['chat_template_sha256'],'tutor_render':tutor,'ceiling_render':ceiling}
    hf_cache.commit()
    return result


@app.function(image=image, gpu='L40S', volumes={'/hf': hf_cache, '/raw': raw_volume},
              timeout=5400, startup_timeout=1200, single_use_containers=True,
              min_containers=0, buffer_containers=0, scaledown_window=2, max_containers=3, cpu=4, memory=32768)
def run_model(model: str, requests: list, run_id: str, manifest: dict, config: dict):
    return generate(model, requests, run_id, manifest, {'gpu':'L40S','batch_size':256, **config})


def _run_stage(models: str = 'rlvr', stage: str = 'smoke', run_id: str = '',
         calibration_path: str = 'results/phase1c/calibration.json', timeout_seconds: int = 5400,
         ladders: str = 'both', max_tokens_think: int = 3072):
    if stage == 'preflight':
        result = preflight.remote()
        target = Path('results/phase1c/runtime-preflight.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
        return
    from src.build_prompts import build_1c
    if stage not in ('smoke','calibration','full'):
        raise ValueError('Unknown stage')
    from src.phase1c import select_ladders
    allowed = select_ladders(ladders)
    if max_tokens_think < 1:
        raise ValueError('max_tokens_think must be positive')
    selected = [m for m,s in MODELS_1C.items() if s['ladder'] in allowed and (models=='all' or (models=='rlvr' and s['checkpoint']=='RLVR') or m in models.split(','))]
    if not selected or not run_id or not run_id.replace('-','').replace('_','').isalnum():
        raise ValueError('Choose registered models and a safe nonempty fresh run ID')
    local = Path('results/raw') / run_id
    if local.exists():
        raise FileExistsError(local)
    if stage in ('smoke','calibration') and any(MODELS_1C[m]['checkpoint'] != 'RLVR' for m in selected):
        raise ValueError('Smoke/calibration must use RLVR checkpoints only')
    from src.phase1c import project_budget, metrics_for, accounted_seconds, read_attempts, write_json, measured_projection, RATE
    if stage == 'full':
        previous = json.loads(Path('results/phase1c/budget-projection.json').read_text())
        cap = 5 if ladders == 'instruct' else 10
        if not previous.get('launch_full') or previous['total_gpu_usd'] > cap:
            raise ValueError('budget projection blocks full generation')
        projection=project_budget(metrics_for('phase1c-smoke'),accounted_seconds(),ladders,max_tokens_think)
        write_json('results/phase1c/budget-projection.json',projection)
        if not projection['launch_full']:
            raise ValueError('Current budget projection blocks full generation')
        if set(selected) != {m for m,s in MODELS_1C.items() if s['ladder'] in allowed}:
            raise ValueError('Full run requires all three checkpoints per selected ladder')
    else:
        cap = 10
        accounted_seconds()  # Reject unresolved prior attempts.
    if timeout_seconds != 5400:
        raise ValueError('Phase 1c GPU timeouts must remain 5400 seconds')
    decisions = json.loads(Path(calibration_path).read_text()) if stage=='full' else None
    attempts=read_attempts()
    active=[]
    current=[]
    def save():
        write_json('results/phase1c/gpu-attempts.json',attempts)
    def finish(item, status):
        item['attempt'].update(status=status, wall_seconds=time.monotonic()-item['tick'],
                              ended_at=datetime.now(timezone.utc).isoformat())
        save()
    # The ephemeral dictionary receives numerical progress only, never responses.
    with modal.Dict.ephemeral() as progress:
        try:
            for model in selected:
                spec=MODELS_1C[model]
                system=decisions['ladders'][spec['ladder']]['system_prompt'] if decisions else 'prompts/system_tutor.txt'
                requests,manifest=build_1c(local/model.split('/')[-1],spec['ladder'],spec['checkpoint'],
                    limit_problems=3 if stage=='smoke' else None, system_path=system, calibration=stage=='calibration')
                attempt={'stage':stage,'run_id':run_id,'model':model,'status':'running',
                    'started_at':datetime.now(timezone.utc).isoformat(),'timeout_seconds':5400}
                attempts.append(attempt)
                tick=time.monotonic()
                current.append({'attempt':attempt,'tick':tick})
                save()
                call=run_model.with_options(timeout=5400,retries=0).spawn(model,requests,run_id,manifest,
                    {'max_tokens_think':max_tokens_think,'progress':progress})
                active.append({'call':call,'attempt':attempt,'tick':tick})
            seen={}
            while active:
                for model, snapshot in progress.items():
                    if seen.get(model)==snapshot['completed']:
                        continue
                    seen[model]=snapshot['completed']
                    measured=measured_projection(snapshot,len(selected))
                    record={'model':model,**snapshot,**measured,'budget_usd':cap}
                    write_json(local/(model.split('/')[-1]+'.progress.json'),record)
                    print('BUDGET_PROGRESS '+json.dumps(record),flush=True)
                    if stage=='full' and measured['total_gpu_usd']>cap:
                        write_json('results/phase1c/budget-stop.json',record)
                        raise ValueError(f'Measured progress projects above ${cap}; cancelling all Phase 1c calls')
                elapsed=sum(item['attempt'].get('wall_seconds', time.monotonic()-item['tick']) for item in current)
                historical = (projection['spent_accounted_seconds']
                              if stage=='full' and ladders!='instruct' else 0)
                if (elapsed+historical+60*len(selected))*RATE >= cap:
                    raise ValueError('Elapsed GPU allocation upper estimate reached cap; cancelling Phase 1c')
                for item in list(active):
                    try:
                        files=item['call'].get(timeout=0)
                    except TimeoutError:
                        continue
                    for remote in files.values():
                        pull_file(remote,local/Path(remote).name)
                    finish(item,'completed')
                    active.remove(item)
                if active:
                    time.sleep(2)
        finally:
            for item in active:
                try:
                    item['call'].cancel(terminate_containers=True)
                finally:
                    finish(item,'cancelled')
            for item in current:
                if item['attempt']['status']=='running':
                    finish(item,'failed')


@app.local_entrypoint()
def main(models: str = 'rlvr', stage: str = 'smoke', run_id: str = '',
         calibration_path: str = 'results/phase1c/calibration.json', timeout_seconds: int = 5400,
         ladders: str = 'both', max_tokens_think: int = 3072):
    from src.phase1c import launch_lock
    with launch_lock():
        return _run_stage(models,stage,run_id,calibration_path,timeout_seconds,ladders,max_tokens_think)
