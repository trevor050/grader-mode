"""Synthetic CPU integration for optional Phase 3 continuation; no live outcomes."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from test_phase3_resume import frozen  # Reuse the established synthetic source contract.
from src.phase3 import CELLS, MODEL, RATE, choose_neurons, score_generation
from src.phase3_resume import (FROZEN_PATHS, SAMPLING, TIMING_METHOD, WORK_RESERVE,
    allocation_limits, digest, file_hash, load_source_package, make_generation_row,
    make_test_batches, project_test_plan)

CONTINUATION = ('src/phase3_resume.py', 'src/modal_app_3_resume.py',
                'src/phase3_validation.py', 'src/analyze_3.py')


def api():
    assert importlib.util.find_spec('src.phase3_resume_analysis') is not None, 'standalone integration module is missing'
    return importlib.import_module('src.phase3_resume_analysis')


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))


def record(batch, cell, c, *, index=None, seconds=100.):
    value = dict(cell=cell, c=c, split=batch[0]['split'], n=len(batch), seed=batch[0]['seed'],
        first_id=batch[0]['id'], last_id=batch[-1]['id'], seconds=seconds, prompt_width=100,
        budget_terminated=False, decode_steps=31, output_tokens=30*len(batch),
        max_output_tokens=30, final_reply_tokens=5*len(batch))
    if index is not None:
        value['batch_index'] = index
    return value


@pytest.fixture(scope='module')
def bundle(frozen, tmp_path_factory):
    root = tmp_path_factory.mktemp('continued-study')
    old = deepcopy(frozen)
    raw = root/'results/raw/source'; raw.mkdir(parents=True)
    out = root/'results/phase3'; out.mkdir(parents=True)
    for name in (*FROZEN_PATHS, *CONTINUATION):
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic frozen version: '+name)
    sources, labels = [], []
    for r in old['requests']:
        if r['split'] != 'test':
            continue
        source = {**r, 'model': MODEL, 'model_revision': old['metrics']['revision'],
            'response': '<think>unfinished', 'finish_reason': 'length',
            'rendered_prompt_sha256': next(t['rendered_prompt_sha256'] for t in old['tracking_rows'] if t['id']==r['id'])}
        sources.append(source)
        labels.append({**source, **score_generation(source['response'], r['gold'], 'length')})
    for name, rows in [('Qwen3-14B.jsonl', sources), ('scored.jsonl', labels)]:
        write_rows(root/'results/raw/phase1b-full'/name, rows)
    for row in old['tracking_rows']:
        row.update(leak=False, excluded_from_rates=True)
    old['manifest']['source_sha256'] = {name: file_hash(root/name) for name in FROZEN_PATHS}
    old['metrics']['manifest'] = deepcopy(old['manifest'])
    stage1 = root/'results/raw/phase1-stage2/manifest.json'; stage1.parent.mkdir(parents=True)
    write_json(stage1, old['manifest'])
    stats = {key: np.full((40, 17408), value) for key, value in
        [('d',1.), ('delta',1.), ('tutor_mean',1.), ('neutral_mean',0.), ('tutor_sd',1.), ('neutral_sd',1.)]}
    old['neurons'] = choose_neurons({**stats, 'n_prompts':400})
    np.savez_compressed(out/'screen-statistics.npz', **stats)
    for name in ('manifest', 'metrics', 'selection', 'neurons'):
        write_json(out/(name+'.json'), old[name])
    write_rows(raw/'generations.jsonl', old['generation_rows'])
    write_rows(raw/'tracking.jsonl', old['tracking_rows'])
    np.savez_compressed(raw/'trajectories.npz', scores=np.empty((0,6)), token_ids=np.empty(0,dtype=np.int32),
        offsets=np.zeros(401,dtype=np.int64), ids=[r['id'] for r in old['tracking_rows']])
    package = load_source_package(root, 'source', old['attempts'])
    resume = root/'results/raw/resume'; resume.mkdir()
    manifest = dict(schema_version=1, source_run_id='source', allocation_run_id='resume',
        source_manifest=package['source_manifest'], source_artifacts_sha256=package['source_artifacts_sha256'],
        continuation_sha256={n:file_hash(root/n) for n in CONTINUATION}, payload_sha256=package['payload_sha256'],
        timing_method=TIMING_METHOD, rate_usd_per_gpu_second=RATE, cap_usd=7., parent_validation_required=True)
    previous = sum(a['wall_seconds'] for a in old['attempts'])
    batch_size, setup, startup = 27, 100., 10.
    pilot_batch = package['dev_requests'][:batch_size]
    pilot_record = record(pilot_batch, 'baseline', 0.)
    plan = project_test_plan(pilot_record['seconds'], batch_size, previous+startup+setup+100.+2.)
    selection = {**old['selection'], **plan, 'test_generation_started':True,
                 'pilot_rows_excluded_from_study':batch_size}
    batches = make_test_batches(package['test_requests'], plan['test_templates'], batch_size)
    def row(r, cell, c, seed):
        return make_generation_row(r, cell=cell, c=c, source_run_id='source', allocation_run_id='resume',
            response='<think>work</think>Try another step.', output_tokens=30, end_token_id=151645,
            batch_seed=seed, rendered_prompt_sha256=package['prompt_hashes'][r['id']], final_reply_tokens=5)
    pilots = [row(r, 'baseline', 0., pilot_batch[0]['seed']) | dict(
        planning_only=True, analysis_excluded=True, pilot_kind=TIMING_METHOD) for r in pilot_batch]
    tests, records = [], []
    for cell in CELLS:
        c = 0. if cell=='baseline' else selection['c_star']
        for index, batch in enumerate(batches):
            tests.extend(row(r, cell, c, batch[0]['seed']) for r in batch)
            records.append(record(batch, cell, c, index=index, seconds=10.))
    function = setup+100.+sum(b['seconds'] for b in records)+20.
    start = 10000.
    metrics = dict(schema_version=1, run_id='source', allocation_run_id='resume', status='completed',
        stage='finished', model=MODEL, revision=old['metrics']['revision'], gpu='L40S',
        source_manifest=old['manifest'], source_selection=old['selection'], resume_manifest=manifest,
        sampling=SAMPLING.copy(), batches=records, output_rows=len(tests), planning_pilot=pilot_record,
        test_generation_started=True, parent_validation_required=True, timing_method=TIMING_METHOD,
        previous_seconds=previous, startup_elapsed_seconds=startup, rate_usd_per_gpu_second=RATE, cap_usd=7.,
        work_reserve_bytes=WORK_RESERVE, capacity_rounding='exact positive integer, capped at 64',
        oom_policy='fail closed, no split/rebatch after timing pilot', allocation_started_unix=start,
        absolute_deadline_unix=allocation_limits(previous,start)['absolute_deadline_unix'],
        versions={p:'synthetic-1' for p in ('torch','transformers','numpy','accelerate')} | {'modal':'1.6.2'},
        original_rendered_prompt_hashes_verified=300, eos_token_ids=[151645,151643], cuda_device='NVIDIA L40S',
        greedy_cache_parity_passed=True, hook_math_runtime_passed=True, batch_size=batch_size,
        kv_bytes_per_token=163840, max_prompt_tokens=100, capacity_total_bytes=48*1024**3,
        capacity_free_bytes=WORK_RESERVE+batch_size*163840*(100+3072)+1000,
        setup_seconds=setup, selection=selection, test_batch_source_ids=[[r['id'] for r in b] for b in batches],
        function_seconds=function, lifetime_gpu_seconds_at_checkpoint=previous+startup+function,
        estimated_gpu_usd=(previous+startup+function)*RATE)
    write_rows(resume/'planning-pilot.jsonl', pilots); write_rows(resume/'generations.jsonl',tests)
    for name, value in [('resume-manifest',manifest), ('launch-manifest',manifest), ('source-manifest',old['manifest']),
                        ('source-selection',old['selection']), ('neurons',old['neurons']), ('selection',selection), ('metrics',metrics)]:
        write_json(resume/(name+'.json'), value)
    remote_names = ('resume-manifest.json','source-manifest.json','source-selection.json','neurons.json',
                    'selection.json','metrics.json','planning-pilot.jsonl','generations.jsonl')
    result = {**metrics, 'artifacts_sha256':{n:file_hash(resume/n) for n in remote_names},
        'accounting':dict(previous_seconds=previous, startup_elapsed_seconds=startup, function_seconds=function+.001,
            lifetime_gpu_seconds_at_return=previous+startup+function+.001, rate_usd_per_gpu_second=RATE,
            cap_usd=7., final_allocation_wall_time='local lifetime ledger authoritative')}
    write_json(resume/'resume-result.json', result)
    elapsed = startup+function+5.
    attempt = dict(run_id='resume', source_run_id='source', status='completed', previous_seconds=previous,
        gpu='L40S', rate_usd_per_hour=RATE*3600, started_unix=start, ended_unix=start+elapsed,
        wall_seconds=elapsed, observed_wall_seconds=elapsed, function_seconds=function+.002,
        zero_containers_verified_unix=start+elapsed-1., call_id='fc-synthetic',
        accounting_basis='allocation wall time through verified shutdown', **allocation_limits(previous,start))
    write_json(out/'gpu-attempts.json', old['attempts']+[attempt])
    write_json(resume/'containers-after.json', dict(checked_unix=attempt['zero_containers_verified_unix'],containers=[]))
    write_json(resume/'local-execution.json', dict(attempt=attempt, allocation_gpu_usd_upper=(previous+elapsed)*RATE,
        recovery_errors=[], parent_validation_required=True))
    return root


@pytest.fixture
def project(bundle, tmp_path):
    root = tmp_path/'project'; shutil.copytree(bundle, root)
    return root


def validate(root):
    return api().validate(root=root, source_run_id='source', resume_run_id='resume')


def mutate(root, name, change, *, raw=True, rehash=True):
    path = root/('results/raw/resume' if raw else 'results/phase3')/name
    obj = json.loads(path.read_text()); change(obj); write_json(path,obj)
    if raw and rehash and name not in ('resume-result.json','local-execution.json','containers-after.json','launch-manifest.json'):
        result_path = path.parent/'resume-result.json'; result=json.loads(result_path.read_text())
        if name=='metrics.json':
            result.update(obj)
        result['artifacts_sha256'][name]=file_hash(path)
        write_json(result_path,result)


def mutate_rows(root, filename, change):
    path=root/'results/raw/resume'/filename
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    change(rows); write_rows(path,rows)
    mutate(root,'resume-result.json',lambda r:r['artifacts_sha256'].__setitem__(filename,file_hash(path)))


def test_valid_joined_grid_and_pilot_exclusion(project):
    before={str(p.relative_to(project)):file_hash(p) for p in project.rglob('*') if p.is_file()}
    result=validate(project)
    assert len(result['test_rows'])==1400 and len(result['planning_rows'])==27
    assert result['execution_verification']['batch_size']==27
    output=project/'integrated'
    report=api().integrate(root=project,source_run_id='source',resume_run_id='resume',output_dir=output)
    assert report['n_generation_rows']==1400 and report['n_dev_responses_excluded']==300
    assert report['source_metrics']['status']=='failed' and report['h14']['status']=='complete'
    assert report['execution_verification']['completed'] and not report['execution_verification']['h14_missing']
    assert report['execution_verification']['planning_pilot_rows_excluded']==27
    assert report['execution_verification']['lifetime_gpu_usd'] <= 7.
    assert report['bootstrap']['resamples']==2000 and report['bootstrap']['seed']==0
    assert report['gold_ge_10']['n_generation_rows']==1400
    assert set(p.name for p in output.iterdir())=={'summary.md','summary.json','trajectories.csv','integration.json'}
    assert 'c*: 2' in (output/'summary.md').read_text()
    assert len((output/'trajectories.csv').read_text().splitlines())==181
    assert before=={n:file_hash(project/n) for n in before}
    provenance=json.loads((output/'integration.json').read_text())
    assert provenance['source_manifest']['source_sha256']==report['source_manifest']['source_sha256']
    assert provenance['study_kind']=='continued_logical_study'


@pytest.mark.parametrize('change',[
    lambda a:a[-1].update(source_run_id='other'),
    lambda a:a[-1].update(previous_seconds=5000.),
    lambda a:a[0].update(wall_seconds=0.),
    lambda a:a[-1].update(status='running'),
    lambda a:a.append(dict(run_id='later',status='completed',wall_seconds=1.,ended_unix=20000.)),
    lambda a:a.append(dict(a[-1],run_id='second')),
    lambda a:a[-1].update(wall_seconds=10000.),
])
def test_bad_lineage_and_closed_ledger_rejected(project,change):
    mutate(project,'gpu-attempts.json',change,raw=False)
    with pytest.raises(ValueError): validate(project)


@pytest.mark.parametrize('name,change',[
    ('resume-result.json',lambda r:r.update(status='budget_stopped')),
    ('metrics.json',lambda r:r.update(incomplete_batches=[{}])),
    ('metrics.json',lambda r:r.update(eos_token_ids=[151645])),
    ('metrics.json',lambda r:r.update(sampling=SAMPLING | {'max_new_tokens':512})),
    ('metrics.json',lambda r:r['selection'].update(c_star=8)),
    ('metrics.json',lambda r:r['selection'].update(projected_test_seconds=1.)),
    ('metrics.json',lambda r:r.update(batch_size=24)),
    ('metrics.json',lambda r:r.update(setup_seconds=10000.)),
    ('metrics.json',lambda r:r['planning_pilot'].update(output_tokens=1)),
    ('metrics.json',lambda r:r['test_batch_source_ids'][0].reverse()),
    ('metrics.json',lambda r:r['batches'][1].update(batch_index=0)),
    ('metrics.json',lambda r:r['batches'][0].update(n=26)),
    ('local-execution.json',lambda r:r.update(recovery_errors=['missing'])) ,
    ('local-execution.json',lambda r:r['attempt'].update(call_id='different')),
    ('containers-after.json',lambda r:r.update(checked_unix=1.)),
    ('containers-after.json',lambda r:r.update(containers=[dict(app_name='grader-mode-phase3-resume',app_id='ap-bad')])),
    ('launch-manifest.json',lambda r:r.update(payload_sha256='0'*64)),
    ('resume-manifest.json',lambda r:r['source_artifacts_sha256'].update({'extra.json':'0'*64})),
])
def test_invalid_metadata_rejected(project,name,change):
    mutate(project,name,change)
    with pytest.raises(ValueError): validate(project)


@pytest.mark.parametrize('filename,change',[
    ('planning-pilot.jsonl',lambda r:r[0].update(planning_only=False)),
    ('planning-pilot.jsonl',lambda r:r[0].update(analysis_excluded=False)),
    ('planning-pilot.jsonl',lambda r:r[0].update(leak=True)),
    ('planning-pilot.jsonl',lambda r:r[0].update(batch_seed=1)),
    ('planning-pilot.jsonl',lambda r:r[0].update(rendered_prompt_sha256='0'*64)),
    ('planning-pilot.jsonl',lambda r:r[0].update(output_tokens=3,finish_reason='length',end_token_id=None)),
    ('generations.jsonl',lambda r:r.append(deepcopy(r[0]))),
    ('generations.jsonl',lambda r:r.pop()),
    ('generations.jsonl',lambda r:r[0].update(split='dev')),
    ('generations.jsonl',lambda r:r[0].update(analysis_excluded=True)),
    ('generations.jsonl',lambda r:r[0].update(planning_only=True)),
    ('generations.jsonl',lambda r:r[0].update(seed=1)),
    ('generations.jsonl',lambda r:r[0].update(batch_seed=1)),
    ('generations.jsonl',lambda r:r[0].update(rendered_prompt_sha256='0'*64)),
    ('generations.jsonl',lambda r:r[0].update(allocation_run_id='other')),
    ('generations.jsonl',lambda r:r[0].update(leak=True)),
    ('generations.jsonl',lambda r:r[0].update(end_token_id=151644)),
])
def test_bad_pilot_or_test_rows_rejected(project,filename,change):
    mutate_rows(project,filename,change)
    with pytest.raises(ValueError): validate(project)


def test_pilot_cannot_be_appended_to_study(project):
    pilot=[json.loads(l) for l in (project/'results/raw/resume/planning-pilot.jsonl').read_text().splitlines()]
    mutate_rows(project,'generations.jsonl',lambda rows:rows.extend(pilot))
    with pytest.raises(ValueError): validate(project)


@pytest.mark.parametrize('name',['DESIGN.md','src/phase3_resume.py','results/phase3/selection.json',
    'results/raw/resume/generations.jsonl'])
def test_changed_artifact_or_version_rejected(project,name):
    path=project/name; path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError): validate(project)


def test_cli_uses_root_and_writes_no_original_metadata(project,monkeypatch):
    output=project/'cli-output'
    monkeypatch.setattr(sys,'argv',['phase3_resume_analysis','--root',str(project),
        '--source-run-id','source','--resume-run-id','resume','--output-dir',str(output)])
    api().main()
    assert json.loads((output/'summary.json').read_text())['study_kind']=='continued_logical_study'


def test_output_refuses_raw_directories_and_existing_files(project):
    with pytest.raises(ValueError):
        api().integrate(root=project,source_run_id='source',resume_run_id='resume',output_dir=project/'results/raw/source')
    output=project/'occupied'; output.mkdir(); (output/'summary.json').write_text('preserve')
    with pytest.raises((ValueError,FileExistsError)):
        api().integrate(root=project,source_run_id='source',resume_run_id='resume',output_dir=output)
    assert (output/'summary.json').read_text()=='preserve'
    assert not (output/'integration.json').exists()


def seal(root, metrics):
    """Maintain redundant metadata/hashes so tamper tests reach rederivation."""
    raw=root/'results/raw/resume'
    for name,key in [('source-manifest','source_manifest'),('source-selection','source_selection'),
                     ('resume-manifest','resume_manifest'),('selection','selection')]:
        write_json(raw/(name+'.json'),metrics[key])
    write_json(raw/'launch-manifest.json',metrics['resume_manifest'])
    write_json(raw/'metrics.json',metrics)
    result=json.loads((raw/'resume-result.json').read_text())
    result.update(metrics)
    result['artifacts_sha256']={n:file_hash(raw/n) for n in result['artifacts_sha256']}
    write_json(raw/'resume-result.json',result)


@pytest.mark.parametrize('field,value',[
    ('projected_test_seconds',1.),('projected_gpu_usd',1.),('batches_per_cell',1),
    ('headroom',1.),('reserve_seconds',0),('c_star',8),('dose_threshold_met',False),
    ('spent_gpu_seconds_at_test_gate',5348.528),('pilot_rows_excluded_from_study',0),
])
def test_consistently_rehashed_selection_still_requires_exact_budget_and_dose(project,field,value):
    raw=project/'results/raw/resume'
    metrics=json.loads((raw/'metrics.json').read_text())
    metrics['selection'][field]=value
    seal(project,metrics)
    with pytest.raises(ValueError): validate(project)


def test_valid_two_then_one_projection_retains_frozen_batching(project):
    raw=project/'results/raw/resume'
    attempts=json.loads((project/'results/phase3/gpu-attempts.json').read_text())
    package=load_source_package(project,'source',attempts[:-1])
    metrics=json.loads((raw/'metrics.json').read_text())
    metrics['planning_pilot']['seconds']=110.
    plan=project_test_plan(110.,27,metrics['previous_seconds']+10.+100.+110.+2.)
    assert plan['test_templates']==1 and len(plan['alternatives'])==2
    metrics['selection']={**package['source_selection'],**plan,'test_generation_started':True,
                          'pilot_rows_excluded_from_study':27}
    batches=make_test_batches(package['test_requests'],1,27)
    metrics['test_batch_source_ids']=[[r['id'] for r in b] for b in batches]
    rows=[json.loads(l) for l in (raw/'generations.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['template_idx']==0]
    seeds={r['id']:b[0]['seed'] for b in batches for r in b}
    for row in rows: row['batch_seed']=seeds[row['phase1b_id']]
    metrics['batches']=[record(batch,cell,0. if cell=='baseline' else 2,index=i,seconds=10.)
        for cell in CELLS for i,batch in enumerate(batches)]
    metrics['output_rows']=700
    write_rows(raw/'generations.jsonl',rows)
    seal(project,metrics)
    result=validate(project)
    assert len(result['test_rows'])==700
    assert result['selection']['timing_template_reduction'] is True
    assert result['execution_verification']['grid_counts']=={c:100 for c in CELLS}


def test_valid_all_four_dose_fallback_retains_500_development_rows(project):
    raw=project/'results/raw/resume'; original=project/'results/raw/source'; out=project/'results/phase3'
    dev=[json.loads(l) for l in (original/'generations.jsonl').read_text().splitlines()]
    for row in dev:
        row['response']='<think>work</think>\\boxed{42}'
        row.update(score_generation(row['response'],row['gold'],'stop'))
    oldmetrics=json.loads((out/'metrics.json').read_text())
    for c in (4,8):
        for row in [deepcopy(r) for r in dev if r['generation_cell']=='dose_1']:
            row.update(c=c,generation_cell=f'dose_{c}',id=f'{row["phase1b_id"]}:dose_{c}:c{c}')
            dev.append(row)
        oldmetrics['batches'].append({**oldmetrics['batches'][1],'cell':f'dose_{c}','c':c})
    write_rows(original/'generations.jsonl',dev)
    oldselection=json.loads((out/'selection.json').read_text())
    oldselection.update(c_star=8,dose_threshold_met=False,dose_trials=[
        dict(c=c,n=100,n_valid=100,leak=1.,reduction=0.) for c in (1,2,4,8)])
    oldmetrics['output_rows']=500
    write_json(out/'metrics.json',oldmetrics); write_json(out/'selection.json',oldselection)
    attempts=json.loads((out/'gpu-attempts.json').read_text())
    package=load_source_package(project,'source',attempts[:-1])
    metrics=json.loads((raw/'metrics.json').read_text())
    metrics['source_selection']=oldselection
    metrics['selection'].update(oldselection)
    plan=project_test_plan(metrics['planning_pilot']['seconds'],27,
                           metrics['selection']['spent_gpu_seconds_at_test_gate'])
    metrics['selection'].update(plan,test_generation_started=True)
    metrics['resume_manifest'].update(payload_sha256=package['payload_sha256'],
        source_artifacts_sha256=package['source_artifacts_sha256'])
    rows=[json.loads(l) for l in (raw/'generations.jsonl').read_text().splitlines()]
    for row in rows:
        if row['generation_cell']!='baseline':
            row.update(c=8,id=f'{row["phase1b_id"]}:{row["generation_cell"]}:c8')
    for batch in metrics['batches']:
        if batch['cell']!='baseline': batch['c']=8
    write_rows(raw/'generations.jsonl',rows)
    seal(project,metrics)
    result=validate(project)
    assert len(result['development_rows'])==500 and result['selection']['c_star']==8
    assert result['selection']['dose_threshold_met'] is False


@pytest.mark.parametrize('change',[
    lambda a:a[0].pop('zero_containers_verified_unix'),
    lambda a:a[0].pop('started_unix'),
    lambda a:a[0].update(wall_seconds=348.),
    lambda a:a[-1].update(zero_containers_verified_unix=1.),
])
def test_every_lifetime_attempt_requires_paid_shutdown_tail(project,change):
    mutate(project,'gpu-attempts.json',change,raw=False)
    with pytest.raises(ValueError): validate(project)


def test_missing_artifact_hash_rejected(project):
    mutate(project,'resume-result.json',lambda r:r['artifacts_sha256'].pop('planning-pilot.jsonl'))
    with pytest.raises(ValueError): validate(project)


@pytest.mark.parametrize('key,value',[
    ('payload_sha256','0'*64),
    ('source_artifacts_sha256',{}),
    ('continuation_sha256',{}),
])
def test_consistent_manifest_tampering_is_rederived_not_trusted(project,key,value):
    raw=project/'results/raw/resume'
    metrics=json.loads((raw/'metrics.json').read_text())
    metrics['resume_manifest'][key]=value
    seal(project,metrics)
    with pytest.raises(ValueError): validate(project)


def test_valid_length_finished_pilot_remains_excluded(project):
    raw=project/'results/raw/resume'
    rows=[json.loads(l) for l in (raw/'planning-pilot.jsonl').read_text().splitlines()]
    rows[0].update(response='<think>unfinished',output_tokens=3072,finish_reason='length',
                   end_token_id=None,final_reply_tokens=0)
    rows[0].update(score_generation(rows[0]['response'],rows[0]['gold'],'length'))
    write_rows(raw/'planning-pilot.jsonl',rows)
    metrics=json.loads((raw/'metrics.json').read_text())
    metrics['planning_pilot'].update(decode_steps=3072,max_output_tokens=3072,
        output_tokens=sum(r['output_tokens'] for r in rows),
        final_reply_tokens=sum(r['final_reply_tokens'] for r in rows))
    seal(project,metrics)
    checked=validate(project)
    assert checked['planning_rows'][0]['truncated'] is True
    assert checked['execution_verification']['n_test_rows']==1400


def test_source_must_record_explicit_failed_projection_gate(project):
    raw=project/'results/raw/resume'; out=project/'results/phase3'
    original=json.loads((out/'metrics.json').read_text()); original['status']='budget_stopped'
    write_json(out/'metrics.json',original)
    attempts=json.loads((out/'gpu-attempts.json').read_text()); attempts[-2]['status']='budget_stopped'
    write_json(out/'gpu-attempts.json',attempts)
    package=load_source_package(project,'source',attempts[:-1])
    metrics=json.loads((raw/'metrics.json').read_text())
    metrics['resume_manifest'].update(payload_sha256=package['payload_sha256'],
        source_artifacts_sha256=package['source_artifacts_sha256'])
    seal(project,metrics)
    with pytest.raises(ValueError,match='failed'):
        validate(project)
