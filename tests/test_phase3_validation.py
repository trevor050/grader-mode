"""Synthetic persisted Phase 3 execution contracts; no GPU or real outcomes."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from src.models import MODELS
from src.phase3 import (MODEL, RATE, build_requests, choose_neurons,
                        score_generation, summarize_trajectory)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture(scope='module')
def execution():
    sample=[{'problem_idx':1000+3*i,'question':f'{i+2} plus 3?', 'gold':42} for i in range(200)]
    manifest={'phase':'phase3','dataset':'openai/gsm8k','config':'main','split':'test',
              'revision':'dataset-revision','sample_seed':0,'sample':sample,
              'preregistration_commit':'0e6a330','cap_usd':7.,
              'source_sha256':{'DESIGN.md':'a'*64}}
    requests=build_requests(sample)
    stats={k:np.full((2,50),v) for k,v in
           [('d',1.),('delta',1.),('tutor_mean',1.),('neutral_mean',0.),('tutor_sd',1.),('neutral_sd',1.)]}
    stats['n_prompts']=400
    neurons=choose_neurons(stats)
    source_raw=[]; source_scored=[]; tracks=[]; generations=[]; batches=[]
    values=np.arange(12,dtype=np.float32).reshape(2,6)
    for r in requests:
        prompt=digest(json.dumps(r['messages'],sort_keys=True))
        if r['split']=='test':
            text='<think>consider the problem</think>\\boxed{42}'
            raw={**r,'phase':'phase1b','model':MODEL,'model_revision':MODELS[MODEL],
                 'response':text,'finish_reason':'stop','rendered_prompt_sha256':prompt}
            source_raw.append(raw)
            source_scored.append({**raw,**score_generation(text,42,'stop')})
            tracks.append({k:r[k] for k in ('id','problem_idx','selection_rank','template_idx','gold')}
                          |dict(leak=True,excluded_from_rates=False,closed_reasoning=True,n_reasoning_tokens=2,
                                rendered_prompt_sha256=prompt,source_response_sha256=digest(text),
                                **summarize_trajectory(values)))
    for split,cell,c in [('dev','baseline',0.),('dev','dose_1',1.),('dev','dose_2',2.),
                         *[('test',cell,0. if cell=='baseline' else 2.)
                           for cell in ('baseline','tutor',*(f'control_{i}' for i in range(5)))]]:
        group=[r for r in requests if r['split']==split and r['template_idx']<(1 if split=='dev' else 2)]
        for r in group:
            final='Try another step.' if cell=='dose_2' and r['selection_rank']<=20 else '\\boxed{42}'
            text=f'<think>consider</think>{final}'
            generations.append({**r,'id':f'{r["id"]}:{cell}:c{c:g}',
                                'generation_cell':cell,'intervention':'tutor' if cell.startswith('dose_') else cell,
                                'c':c,'run_id':'synthetic-phase3','model':MODEL,'model_revision':MODELS[MODEL],
                                'response':text,'output_tokens':30,'final_reply_tokens':10,'finish_reason':'stop',
                                'end_token_id':151645,'batch_seed':group[0]['seed'],'budget_terminated':False,
                                'rendered_prompt_sha256':digest(json.dumps(r['messages'],sort_keys=True)),
                                **score_generation(text,42,'stop')})
        batches.append(dict(split=split,cell=cell,c=c,n=len(group),seed=group[0]['seed'],
                            first_id=group[0]['id'],last_id=group[-1]['id'],seconds=100.,
                            budget_terminated=False,decode_steps=31))
    selection=dict(c_star=2,dose_threshold_met=True,dev_baseline_leak=1.,
                   dose_trials=[dict(c=1,n=100,n_valid=100,leak=1.,reduction=0.),
                                dict(c=2,n=100,n_valid=100,leak=.8,reduction=.2)],
                   test_templates=2,test_generation_started=True,timing_template_reduction=False,
                   timing_seconds_per_request=1.2,spent_gpu_seconds_at_test_gate=1000.,
                   projected_test_seconds=1680.,projected_gpu_usd=(1000+1680+300)*RATE)
    metrics=dict(status='completed',run_id='synthetic-phase3',model=MODEL,revision=MODELS[MODEL],
                 manifest=deepcopy(manifest),selection=deepcopy(selection),output_rows=len(generations),
                 tracking_rows=400,batches=batches,sampling={'max_new_tokens':3072},eos_token_ids=[151645,151643])
    return dict(tracking_rows=tracks,generation_rows=generations,neurons=neurons,selection=selection,
                manifest=manifest,metrics=metrics,source_raw=source_raw,source_scored=source_scored)


@pytest.fixture
def case(execution):
    return deepcopy(execution)


def validate(case,**kwargs):
    from src.phase3_validation import validate_execution
    return validate_execution(**case,**kwargs)


def test_full_persisted_execution_validates(case):
    result=validate(case)
    assert result['completed'] is True
    assert result['n_tracking_rows']==400
    assert result['n_dev_rows']==300 and result['n_test_rows']==1400
    assert result['n_source_labels_verified']==400


@pytest.mark.parametrize('field,value',[
    ('status','running'),('status','budget_stopped'),('status','failed'),
    ('revision','unpinned'),('model','Qwen/Qwen3-8B'),('budget_terminated',True),
    ('output_rows',0),('tracking_rows',399),
])
def test_execution_metrics_rejected(case,field,value):
    case['metrics'][field]=value
    with pytest.raises(ValueError):validate(case)


@pytest.mark.parametrize('field,value',[
    ('id','other'),('phase1b_id','other'),('seed',True),('selection_rank',2),
    ('split','test'),('messages',[]),('model_revision','unpinned'),
    ('intervention','control_5'),('c',4.),('batch_seed',-1),
    ('rendered_prompt_sha256','x'*64),('output_tokens',3073),
    ('finish_reason','deadline'),('budget_terminated',True),
])
def test_generation_provenance_rejected(case,field,value):
    case['generation_rows'][0][field]=value
    with pytest.raises(ValueError):validate(case)


@pytest.mark.parametrize('field,value',[
    ('leak',False),('excluded_from_rates',True),('source_response_sha256','b'*64),
    ('rendered_prompt_sha256','c'*64),('closed_reasoning',False),('id','other'),
])
def test_tracking_source_mismatch(case,field,value):
    case['tracking_rows'][0][field]=value
    with pytest.raises(ValueError):validate(case)


def test_tracking_grid_and_missing_exclusion_rejected(case):
    case['tracking_rows'].pop()
    with pytest.raises(ValueError):validate(case)
    case['tracking_rows'].append(deepcopy(case['tracking_rows'][0]))
    with pytest.raises(ValueError):validate(case)


@pytest.mark.parametrize('which',['source_raw','source_scored'])
def test_changed_local_phase1b_sources_rejected(case,which):
    case[which][0]['response']+=' altered'
    with pytest.raises(ValueError):validate(case)


@pytest.mark.parametrize('tokens',[0,30,3071,3073])
def test_shortened_generation_length_rejected(case,tokens):
    case['generation_rows'][0].update(finish_reason='length',output_tokens=tokens,truncated=True)
    with pytest.raises(ValueError):validate(case)


@pytest.mark.parametrize('field,value',[
    ('c_star',1),('dose_threshold_met',False),('dev_baseline_leak',.9),
    ('test_templates',True),('test_generation_started',False),('timing_template_reduction',True),
    ('timing_seconds_per_request',.5),('projected_test_seconds',10),
    ('spent_gpu_seconds_at_test_gate',20000),('projected_gpu_usd',8),
])
def test_selection_mismatch_or_invalid_gate_rejected(case,field,value):
    case['selection'][field]=value
    case['metrics']['selection']=deepcopy(case['selection'])
    with pytest.raises(ValueError):validate(case)


def test_dev_trial_summary_and_exact_grid_rejected(case):
    case['selection']['dose_trials'][0]['n_valid']=99
    case['metrics']['selection']=deepcopy(case['selection'])
    with pytest.raises(ValueError):validate(case)
    case['selection']['dose_trials'][0]['n_valid']=100
    case['metrics']['selection']=deepcopy(case['selection'])
    case['generation_rows'].pop(0)
    case['metrics']['output_rows']-=1
    with pytest.raises(ValueError):validate(case)


@pytest.mark.parametrize('field,value',[
    ('d',-1.),('delta',2.),('tutor_sd',0.),('layer',40),('neuron',17408),('id','9.9'),
])
def test_neuron_statistics_and_coordinates_rejected(case,field,value):
    case['neurons']['tutor'][0][field]=value
    with pytest.raises(ValueError):validate(case)


def test_controls_exclude_tutor_match_layers_and_signs(case):
    case['neurons']['controls'][0][0]=deepcopy(case['neurons']['tutor'][0])
    with pytest.raises(ValueError):validate(case)


def test_batch_seed_and_rendered_prompt_consistency(case):
    case['generation_rows'][1]['batch_seed']+=1
    with pytest.raises(ValueError):validate(case)


def test_tracking_only_requires_explicit_budget_stop(case):
    case['generation_rows']=None
    with pytest.raises(ValueError):validate(case,allow_tracking_only=True)
    case['metrics'].update(status='budget_stopped',output_rows=0,batches=[],selection=None)
    case['selection']={}
    with pytest.raises(ValueError):validate(case)
    result=validate(case,allow_tracking_only=True)
    assert result['completed'] is False and result['h14_missing'] is True


def test_raw_ragged_trajectories_recompute_summaries(case,tmp_path):
    path=tmp_path/'trajectories.npz'
    scores=np.tile(np.arange(12,dtype=np.float32).reshape(2,6),(400,1))
    np.savez_compressed(path,scores=scores,token_ids=np.arange(800),
                        offsets=np.arange(0,801,2),ids=[r['id'] for r in case['tracking_rows']])
    assert validate(case,trajectories=path)['n_trajectories_verified']==400
    case['tracking_rows'][0]['slopes'][0]+=1
    with pytest.raises(ValueError,match='trajector'):validate(case,trajectories=path)


def setup_cli(case,tmp_path,monkeypatch):
    import src.build_prompts
    root=tmp_path/'project'; root.mkdir()
    for name,key in [('results/raw/phase1b-full/Qwen3-14B.jsonl','source_raw'),
                     ('results/raw/phase1b-full/scored.jsonl','source_scored')]:
        target=root/name; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(''.join(json.dumps(r)+'\n' for r in case[key]))
    (root/'DESIGN.md').write_text('frozen synthetic preregistration')
    case['manifest']['source_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in ('DESIGN.md','results/raw/phase1b-full/Qwen3-14B.jsonl','results/raw/phase1b-full/scored.jsonl')}
    original=root/'results/raw/phase1-stage2/manifest.json';original.parent.mkdir(parents=True)
    original.write_text(json.dumps(case['manifest']))
    case['metrics']['manifest']=deepcopy(case['manifest'])
    raw=tmp_path/'raw';raw.mkdir();out=tmp_path/'output';out.mkdir()
    (raw/'tracking.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in case['tracking_rows']))
    if case['generation_rows']:
        (raw/'generations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in case['generation_rows']))
    for name in ('selection','metrics','manifest','neurons'):
        (out/f'{name}.json').write_text(json.dumps(case[name]))
    monkeypatch.setattr(src.build_prompts,'ROOT',root)
    return raw,out,root


def test_cli_loads_outputdir_metadata_and_verifies_local_sources(case,tmp_path,monkeypatch):
    import sys
    from src.analyze_3 import main
    raw,out,_=setup_cli(case,tmp_path,monkeypatch)
    monkeypatch.setattr(sys,'argv',['analyze_3','--input',str(raw),'--output-dir',str(out)])
    main()
    report=json.loads((out/'summary.json').read_text())
    assert report['execution_verification']['completed'] is True
    assert report['execution_verification']['n_source_files_verified']==3
    assert report['h14']['status']=='complete'
    assert report['bootstrap']['resamples']==2000 and report['bootstrap']['seed']==0
    assert len((out/'trajectories.csv').read_text().splitlines())==181


def test_cli_tracking_only_budgetstop_never_claims_completed(case,tmp_path,monkeypatch):
    import sys
    from src.analyze_3 import main
    case['generation_rows']=None
    case['selection']={}
    case['metrics'].update(status='budget_stopped',selection=None,output_rows=0,batches=[])
    raw,out,_=setup_cli(case,tmp_path,monkeypatch)
    monkeypatch.setattr(sys,'argv',['analyze_3','--input',str(raw),'--output-dir',str(out),'--allow-tracking-only'])
    main()
    report=json.loads((out/'summary.json').read_text())
    assert report['execution_verification']['completed'] is False
    assert report['h14']['status']=='missing'
    assert 'completed: false' in (out/'summary.md').read_text()


def test_cli_rejects_changed_source_file_before_output(case,tmp_path,monkeypatch):
    import sys
    from src.analyze_3 import main
    raw,out,root=setup_cli(case,tmp_path,monkeypatch)
    (root/'DESIGN.md').write_text('changed')
    monkeypatch.setattr(sys,'argv',['analyze_3','--input',str(raw),'--output-dir',str(out)])
    with pytest.raises(SystemExit):main()
    assert not (out/'summary.json').exists()


def sync_selection(case):
    case['metrics']['selection']=deepcopy(case['selection'])
    case['metrics']['output_rows']=len(case['generation_rows'])


def test_valid_fallback_8_requires_all_four_failed_trials(case):
    # No dose reaches the threshold; every generation is still retained.
    for row in case['generation_rows']:
        if row['split']=='dev':
            row['response']='<think>consider</think>\\boxed{42}'
            row.update(score_generation(row['response'],42,'stop'))
        elif row['intervention']!='baseline':
            row['c']=8.
            row['id']=f'{row["phase1b_id"]}:{row["generation_cell"]}:c8'
    dose1=[deepcopy(r) for r in case['generation_rows'] if r['generation_cell']=='dose_1']
    for c in (4,8):
        for row in deepcopy(dose1):
            row.update(c=float(c),generation_cell=f'dose_{c}',id=f'{row["phase1b_id"]}:dose_{c}:c{c}')
            case['generation_rows'].append(row)
        batch=deepcopy(case['metrics']['batches'][1]);batch.update(cell=f'dose_{c}',c=float(c))
        case['metrics']['batches'].insert(c//4+2,batch)
    for b in case['metrics']['batches']:
        if b['split']=='test' and b['cell']!='baseline':b['c']=8.
    case['selection'].update(c_star=8,dose_threshold_met=False,
        dose_trials=[dict(c=c,n=100,n_valid=100,leak=1.,reduction=0.) for c in (1,2,4,8)])
    sync_selection(case)
    assert validate(case)['n_dev_rows']==500


def test_valid_reduced_test_grid_uses_timing_gate(case):
    case['generation_rows'][:]=[r for r in case['generation_rows'] if r['split']=='dev' or r['template_idx']==0]
    for batch in case['metrics']['batches']:
        if batch['split']=='test':
            batch['n']=100
            batch['last_id']=batch['last_id'].replace(':t1:',':t0:')
    spent=11500.
    case['selection'].update(test_templates=1,timing_template_reduction=True,
                            spent_gpu_seconds_at_test_gate=spent,
                            projected_test_seconds=840.,projected_gpu_usd=(spent+840+300)*RATE)
    sync_selection(case)
    assert validate(case)['n_test_rows']==700


def test_dev_rate_recomputed_with_empty_final_denominator(case):
    for row in [r for r in case['generation_rows'] if r['generation_cell']=='dose_2'][-10:]:
        row.update(response='<think>unfinished',finish_reason='length',output_tokens=3072,
                   final_reply_tokens=0,end_token_id=None)
        row.update(score_generation(row['response'],42,'length'))
    batch=next(b for b in case['metrics']['batches'] if b['cell']=='dose_2');batch['decode_steps']=3072
    trial=case['selection']['dose_trials'][1]
    trial.update(n_valid=90,leak=70/90,reduction=1-70/90)
    sync_selection(case)
    assert validate(case)['completed']


@pytest.mark.parametrize('value',[None,0,151644,True])
def test_stop_requires_registered_eos_token(case,value):
    case['generation_rows'][0]['end_token_id']=value
    with pytest.raises(ValueError):validate(case)


def test_batches_cannot_put_development_after_test(case):
    case['metrics']['batches'].append(case['metrics']['batches'].pop(0))
    with pytest.raises(ValueError):validate(case)


def test_missing_tracking_exclusion_is_not_inferred(case):
    case['tracking_rows'][0].pop('excluded_from_rates')
    with pytest.raises(ValueError):validate(case)
