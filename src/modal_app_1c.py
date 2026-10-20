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
              single_use_containers=True, min_containers=0)
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
              min_containers=0, max_containers=2, cpu=4, memory=32768)
def run_model(model: str, requests: list, run_id: str, manifest: dict):
    return generate(model, requests, run_id, manifest, {'gpu':'L40S','batch_size':256})


def _run_stage(models: str = 'rlvr', stage: str = 'smoke', run_id: str = '',
         calibration_path: str = 'results/phase1c/calibration.json', timeout_seconds: int = 5400):
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
    selected = [m for m,s in MODELS_1C.items() if models=='all' or (models=='rlvr' and s['checkpoint']=='RLVR') or m in models.split(',')]
    if not selected or not run_id or not run_id.replace('-','').replace('_','').isalnum():
        raise ValueError('Choose registered models and a safe nonempty fresh run ID')
    local = Path('results/raw') / run_id
    if local.exists():
        raise FileExistsError(local)
    if stage in ('smoke','calibration') and any(MODELS_1C[m]['checkpoint'] != 'RLVR' for m in selected):
        raise ValueError('Smoke/calibration must use RLVR checkpoints only')
    if stage == 'full':
        from src.phase1c import project_budget, metrics_for, accounted_seconds, write_json
        projection = json.loads(Path('results/phase1c/budget-projection.json').read_text())
        if not projection.get('launch_full') or projection['total_gpu_usd'] > 10:
            raise ValueError('budget projection blocks full generation')
        projection=project_budget(metrics_for('phase1c-smoke'),accounted_seconds())
        write_json('results/phase1c/budget-projection.json',projection)
        if not projection['launch_full']:
            raise ValueError('Current budget projection blocks full generation')
        if set(selected) != set(MODELS_1C):
            raise ValueError('Full run requires all six checkpoints')
        if not 1 <= timeout_seconds <= 5400:
            raise ValueError('Invalid timeout')
    from src.phase1c import allocate_timeouts, remaining_gpu_seconds, read_attempts, write_json
    timeouts = allocate_timeouts(projection) if stage == 'full' else {}
    decisions = json.loads(Path(calibration_path).read_text()) if stage=='full' else None
    calls=[]
    try:
        for model in selected:
            spec=MODELS_1C[model]
            system=decisions['ladders'][spec['ladder']]['system_prompt'] if decisions else 'prompts/system_tutor.txt'
            requests,manifest=build_1c(local/model.split('/')[-1],spec['ladder'],spec['checkpoint'],
                limit_problems=3 if stage=='smoke' else None, system_path=system, calibration=stage=='calibration')
            remaining=remaining_gpu_seconds()
            if remaining < 120:
                raise ValueError('Insufficient remaining total GPU budget before spawn')
            remote_timeout=min(5400,timeout_seconds,timeouts.get(spec['ladder'],5400),int(remaining))
            attempts=read_attempts()
            attempt={'stage':stage,'run_id':run_id,'model':model,'status':'running',
                'started_at':datetime.now(timezone.utc).isoformat(),'timeout_seconds':remote_timeout}
            attempts.append(attempt)
            write_json('results/phase1c/gpu-attempts.json',attempts)
            tick=time.monotonic()
            try:
                call=run_model.with_options(timeout=remote_timeout,retries=0).spawn(model,requests,run_id,manifest)
                calls.append(call)
                # Include startup/queue time in a local deadline; remote timeout
                # alone excludes startup and cannot enforce a cumulative budget.
                files=call.get(timeout=min(remote_timeout+1200,remaining))
                for remote in files.values():
                    pull_file(remote,local/Path(remote).name)
                calls.remove(call)
                attempt['status']='completed'
            finally:
                if attempt['status']=='running':
                    attempt['status']='failed'
                    for active in calls:
                        active.cancel(terminate_containers=True)
                    calls.clear()
                attempt['wall_seconds']=time.monotonic()-tick
                attempt['ended_at']=datetime.now(timezone.utc).isoformat()
                write_json('results/phase1c/gpu-attempts.json',attempts)
    finally:
        for call in calls:
            call.cancel(terminate_containers=True)


@app.local_entrypoint()
def main(models: str = 'rlvr', stage: str = 'smoke', run_id: str = '',
         calibration_path: str = 'results/phase1c/calibration.json', timeout_seconds: int = 5400):
    from src.phase1c import launch_lock
    with launch_lock():
        return _run_stage(models,stage,run_id,calibration_path,timeout_seconds)
