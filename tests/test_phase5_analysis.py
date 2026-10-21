"""Synthetic Phase 5 complete grids, pairing, provenance and fallback gates."""
import itertools

import numpy as np
import pytest

from tests.test_phase5_score import row, with_response, continuation, GEMMA, NEMOTRON, GLM


def grid(n=3, drop=0, second_model=NEMOTRON):
    from src.build_prompts import request_seed
    from src.score_4 import PREFILLS
    rows=[]
    for model in (GEMMA,second_model):
        gemma=model==GEMMA
        cells = [('main',t,f,'none') for t,f in itertools.product((False,True),('plain','explicit'))] if gemma or drop<3 else []
        prefills = ('empty','reminder') if (drop>=1 and gemma or drop>=2) else tuple(PREFILLS)
        cells += [('prefill',True,'explicit',p) for p in prefills]
        for p,t,(arm,think,fmt,prefill) in itertools.product(range(n),(0,1),cells):
            yes=(arm=='main' and think and fmt=='explicit') or (arm=='prefill' and prefill!='reminder')
            rows.append(row(model,id=f'{p}:{t}:{arm}:{think}:{fmt}:{prefill}',problem_id=p,
                question=f'Problem {p}?',gold=5 if p==0 else 42,template_idx=t,arm=arm,
                thinking_mode='on' if think else 'off',format=fmt,prefill=prefill,
                prefill_text=PREFILLS[prefill] if arm=='prefill' else '',
                response=continuation(model,'Answer: 43' if yes else 'Try again.',thinking=think,arm=arm),
                seed=request_seed(f'phase5:p{p:04d}:t{t}')))
    return rows


def manifest(n=3):
    from src.models_5 import REGISTRATION_COMMIT
    return {'registration_commit':REGISTRATION_COMMIT,'sample':[{'problem_idx':p,'question':f'Problem {p}?','gold':5 if p==0 else 42} for p in range(n)]}


def test_full_grid_metrics_hypotheses_sensitivity_and_pending_audit():
    from src.analyze_5 import analyze
    report=analyze(grid(),expected_problems=3,problem_manifest=manifest())
    assert report['n_responses']==96
    assert report['bootstrap']['resamples']==2000 and report['bootstrap']['seed']==0
    assert not report['inference']['eligible']
    assert report['gold_ge_10']['n_problems']==2
    for model,item in report['models'].items():
        h='H19' if model.startswith('google/') else 'H20'
        for c in ('empty','neutral','irrelevant'):
            con=item['contrasts']['violation'][f'{h}_{c}_minus_reminder']
            assert con['estimate']==1 and con['ci95']==[1,1] and con['holds'] is None
            assert not item['contrasts']['leak'][f'{h}_{c}_minus_reminder']['tested']
        assert 'cap_hit_no_final' in item['cells'][0]['metrics']
        if model.startswith('google/'):
            assert item['contrasts']['violation']['H18']['estimate']==1
            assert item['contrasts']['violation']['on_plain_minus_off_plain']['tested'] is False


def test_problem_pairing_exact_draws_and_order_invariance():
    from src.analyze_5 import analyze
    rows=grid(4)
    for r in rows:
        yes=r['arm']=='main' and r['thinking_mode']=='on' and r['format']=='explicit' and r['problem_id'] in (1,2)
        final='Answer: 99' if yes else 'Try again.'
        if r['model'].startswith('google/'):
            with_response(r,((('<|channel>thought\n' if r['arm']=='main' else '')+'reason<channel|>') if r['thinking_mode']=='on' else '')+final+'<turn|>')
    report=analyze(rows,expected_problems=4)
    got=report['models']['google/gemma-4-12B-it']['contrasts']['violation']['H18']
    draws=np.array([0,1,1,0])[np.random.default_rng(0).integers(0,4,size=(2000,4))].mean(axis=1)
    assert got['estimate']==.5 and got['ci95']==np.quantile(draws,[.025,.975]).tolist()
    assert report==analyze(list(reversed(rows)),expected_problems=4)


@pytest.mark.parametrize('drop',[1,2,3])
def test_whole_cell_drop_order_explicit_only(drop):
    from src.analyze_5 import analyze
    with pytest.raises(ValueError,match='grid|drop'):
        analyze(grid(drop=drop),expected_problems=3)
    result=analyze(grid(drop=drop),expected_problems=3,allow_cell_drops=True)
    assert result['grid']['drop_stage']==drop
    con=result['models']['google/gemma-4-12B-it']['contrasts']['violation']
    assert con['H19_neutral_minus_reminder']['status']=='unavailable'
    assert con['H19_joint']['tested'] is False
    assert con['H19_reduced']['tested'] is True
    # Dropping second-model controls before Gemma controls violates the order.
    illegal=[r for r in grid() if not (r['model'].startswith('nvidia/') and r['prefill'] in ('neutral','irrelevant'))]
    with pytest.raises(ValueError,match='grid|drop'):
        analyze(illegal,expected_problems=3,allow_cell_drops=True)


@pytest.mark.parametrize('mutation',['missing','duplicate','gold','question','run','phase','revision','effort',
    'temperature','top_p','top_k','max_tokens','seed','missing_pin','template','stage','outcome','budget'])
def test_bad_mixed_or_shortened_grid_rejected(mutation):
    from src.analyze_5 import analyze
    rows=grid()
    if mutation=='missing': rows.pop()
    elif mutation=='duplicate': rows.append(rows[0].copy())
    elif mutation=='gold': rows[0]['gold']=4
    elif mutation=='question': rows[0]['question']='wrong'
    elif mutation=='run': rows[0]['run_id']='other'
    elif mutation=='phase': rows[0]['phase']='phase4'
    elif mutation=='revision': rows[0]['model_revision']='wrong'
    elif mutation=='effort': rows[-1]['reasoning_effort']='low'
    elif mutation=='seed': rows[0]['seed']=999
    elif mutation=='missing_pin': rows[0].pop('temperature')
    elif mutation=='template': rows[0]['template_idx']=2
    elif mutation=='stage': rows[0]['stage']='pilot'
    elif mutation=='outcome': rows[0]['violation']=True
    elif mutation=='budget': rows[0]['budget_shortened']=True
    else: rows[0][mutation]=0
    with pytest.raises(ValueError):
        analyze(rows,expected_problems=3)


def test_full_200_requires_original_manifest_and_audit_before_inference():
    from src.analyze_5 import analyze, _original_sample
    with pytest.raises(ValueError,match='original|sample'):
        analyze(grid(200))
    with pytest.raises(ValueError,match='original|sample|manifest'):
        analyze(grid(200),problem_manifest=manifest(200))
    rows=grid(200)
    original=list(_original_sample().items())
    for r in rows:
        pid,(gold,question)=original[r['problem_id']]
        r['seed']=__import__('src.build_prompts',fromlist=['request_seed']).request_seed(f"phase5:p{pid:04d}:t{r['template_idx']}")
        r.update(problem_id=pid,gold=gold,question=question)
    report=analyze(rows)
    assert not report['inference']['eligible']
    assert report['grid']['original_sample_checked']
    assert 'problem_manifest_not_checked' in report['inference']['reasons']


def test_explicit_pilot_mode_never_inference_eligible():
    from src.analyze_5 import analyze
    rows=grid(1)
    for r in rows: r['stage']='pilot'
    result=analyze(rows,expected_problems=1,pilot=True)
    assert not result['inference']['eligible']
    assert 'pilot' in result['inference']['reasons']


def test_complete_original_sample_stays_pending_until_real_labels_supplied():
    from src.analyze_5 import analyze, _original_sample
    from src.phase5_parser_agreement import audit_sample
    original = _original_sample()
    rows = grid(200)
    mapping = dict(enumerate(original.items()))
    for r in rows:
        pid,(gold,question) = mapping[r['problem_id']]
        r['seed']=__import__('src.build_prompts',fromlist=['request_seed']).request_seed(f"phase5:p{pid:04d}:t{r['template_idx']}")
        r.update(problem_id=pid, gold=gold, question=question)
    frozen = {'registration_commit':row()['registration_commit'],'sample':[dict(problem_idx=p,gold=g,question=q) for p,(g,q) in original.items()]}
    pending = analyze(rows, problem_manifest=frozen)
    assert pending['grid']['original_sample_checked']
    assert 'parser_audit_pending' in pending['inference']['reasons']
    # Explicit synthetic test labels exercise the API, never write hand labels.
    sample = audit_sample(rows)
    labels = [dict(id=r['id'],attempt='Answer:' in r['final_reply'],
        committed_number='43' if 'Answer:' in r['final_reply'] else None) for r in sample]
    attempt_only = analyze(rows,problem_manifest=frozen,labels=labels)
    assert attempt_only['parser_agreement']['status'] == 'complete'
    assert attempt_only['parser_agreement']['committed_number']['status'] == 'complete'
    assert not attempt_only['inference']['eligible']
    assert attempt_only['inference']['reasons'] == ['splitter_audit_pending']
    assert pending['inference']['reasons'] == ['parser_audit_pending', 'splitter_audit_pending']
    # These literal finals describe the synthetic generator above, not a copy
    # of the automatic splitter's output used as purported human verification.
    for label in labels:
        label['final_reply'] = 'Answer: 43' if label['attempt'] else 'Try again.'
    complete = analyze(rows,problem_manifest=frozen,labels=labels)
    assert complete['inference']['eligible']
    assert complete['models']['google/gemma-4-12B-it']['contrasts']['violation']['H18']['holds']
    # No accuracy threshold was preregistered: report a checked disagreement
    # without inventing an additional inference cutoff.
    labels[0]['final_reply'] = 'A deliberately different synthetic final.'
    disagreement = analyze(rows,problem_manifest=frozen,labels=labels)
    assert disagreement['inference']['eligible']
    assert disagreement['parser_agreement']['splitter']['n_disagreements'] == 1


def test_full_size_pilot_with_all_synthetic_labels_is_still_descriptive():
    from src.analyze_5 import analyze, _original_sample
    from src.phase5_parser_agreement import audit_sample
    rows = grid(200)
    original = list(_original_sample().items())
    for r in rows:
        pid,(gold,question) = original[r['problem_id']]
        r['seed']=__import__('src.build_prompts',fromlist=['request_seed']).request_seed(f"phase5:p{pid:04d}:t{r['template_idx']}")
        r.update(problem_id=pid,gold=gold,question=question,stage='pilot',pilot=True)
    frozen = {'registration_commit':row()['registration_commit'],'sample':[dict(problem_idx=p,gold=g,question=q) for p,(g,q) in original]}
    labels = [dict(id=r['id'],attempt='Answer:' in r['final_reply'],
        committed_number='43' if 'Answer:' in r['final_reply'] else None,
        final_reply='Answer: 43' if 'Answer:' in r['final_reply'] else 'Try again.')
        for r in audit_sample(rows)]
    report = analyze(rows,problem_manifest=frozen,labels=labels,pilot=True)
    assert report['parser_agreement']['splitter']['status'] == 'complete'
    assert report['inference'] == dict(eligible=False,reasons=['pilot'],
        parser_audit_required_by_preregistration=True,
        splitter_audit_required_by_preregistration=True,
        analysis_before_hand_audit_authorized=False)
    for data in (report,report['gold_ge_10']):
        for item in data['models'].values():
            for contrasts in item['contrasts'].values():
                assert all(c['holds'] is None and not c['tested'] for c in contrasts.values())


def test_all_contrast_components_and_paired_ci_match_manual_arrays():
    from src.analyze_5 import analyze
    from src.score_5 import score_rows
    rows=grid(4)
    for r in rows:
        p,t=r['problem_id'],r['template_idx']
        if r['arm']=='main':
            threshold={(False,'plain'):1,(True,'plain'):2,(False,'explicit'):2,(True,'explicit'):4}
            yes=p < threshold[(r['thinking_mode']=='on',r['format'])]-t
        else:
            yes=p < {'empty':4,'reminder':1,'neutral':2,'irrelevant':3}[r['prefill']]-t
        final='Answer: 42' if yes else 'Try again.'
        r['response']=continuation(r['model'],final,thinking=r['thinking_mode']=='on',arm=r['arm'])
        with_response(r)
    scored=score_rows(rows)
    report=analyze(rows,expected_problems=4)
    draws=np.random.default_rng(0).integers(0,4,size=(2000,4))
    for model,item in report['models'].items():
        for metric in ('violation','leak'):
            def cell(arm,think,fmt,prefill='none'):
                return np.array([np.mean([r[metric] for r in scored if r['model']==model and r['problem_id']==p and
                    (r['arm'],r['thinking'],r['format'],r['prefill'])==(arm,think,fmt,prefill)]) for p in range(4)])
            expected={}
            expected['H18' if model==GEMMA else 'H21']=cell('main',True,'explicit')-cell('main',False,'explicit')-cell('main',True,'plain')+cell('main',False,'plain')
            h='H19' if model.startswith('google/') else 'H20'
            for comparator in ('empty','neutral','irrelevant'):
                expected[f'{h}_{comparator}_minus_reminder']=cell('prefill',True,'explicit',comparator)-cell('prefill',True,'explicit','reminder')
            for name,values in expected.items():
                got=item['contrasts'][metric][name]
                assert got['estimate']==values.mean()
                assert got['ci95']==np.quantile(values[draws].mean(axis=1),[.025,.975]).tolist()


def test_cli_writes_pilot_reports_and_blind_sample(tmp_path):
    import json
    import subprocess
    import sys
    from src.analyze_5 import render_markdown
    source=tmp_path/'raw.jsonl'
    source.write_text(''.join(json.dumps(r)+'\n' for r in grid(1)))
    out=tmp_path/'out'
    subprocess.run([sys.executable,'-m','src.analyze_5','--input',str(source),
        '--output-dir',str(out),'--pilot','--expected-problems','1'],check=True)
    report=json.loads((out/'summary.json').read_text())
    assert not report['inference']['eligible']
    assert 'cap_hit_no_final' in (out/'summary.md').read_text()
    assert 'H18' in render_markdown(report)
    sample=[json.loads(s) for s in (out/'parser-audit-sample.jsonl').read_text().splitlines()]
    assert len(sample)==32 and all(set(r)=={'id','problem','gold','final_reply'} for r in sample)
    splitter=[json.loads(s) for s in (out/'splitter-audit-sample.jsonl').read_text().splitlines()]
    assert [r['id'] for r in splitter] == [r['id'] for r in sample]
    assert report['parser_agreement']['splitter']['sample_path'] == str(out/'splitter-audit-sample.jsonl')
    assert 'Splitter audit: pending; 0/80 human finals' in (out/'summary.md').read_text()


@pytest.mark.parametrize('field,value',[('max_tokens',4096.0),('top_k',-1.0),('pilot',True)])
def test_generation_pin_types_and_pilot_flag_are_strict(field,value):
    from src.analyze_5 import analyze
    rows=grid()
    for r in rows: r[field]=value
    with pytest.raises(ValueError):
        analyze(rows,expected_problems=3)


def test_partial_pilot_has_counts_but_never_unpaired_intervals():
    from src.analyze_5 import analyze
    rows=grid(1)[:3]
    result=analyze(rows,expected_problems=1,pilot=True)
    assert not result['inference']['eligible']
    item=result['models']['google/gemma-4-12B-it']
    assert sum(c['n_responses'] for c in item['cells'])==3
    assert all(c['metrics']['violation']['ci95']==[None,None] for c in item['cells'])
    assert item['contrasts']['violation']['H18']['status']=='unavailable'


@pytest.mark.parametrize('field,value', [('tokenizer_revision','wrong'),
    ('sampling',{'temperature':.7,'top_p':.95,'top_k':-1,'max_tokens':4096,'skip_special_tokens':True}),
    ('package_versions',{'vllm':'wrong'})])
def test_mixed_provenance_and_tokenizer_settings_rejected(field,value):
    from src.analyze_5 import analyze
    rows=grid()
    rows[0][field]=value
    with pytest.raises(ValueError):
        analyze(rows,expected_problems=3)


def test_h21_is_same_interaction_as_h18_and_whole_stage3_drop():
    from src.analyze_5 import analyze
    report=analyze(grid(),expected_problems=3)
    h18=report['models'][GEMMA]['contrasts']['violation']['H18']
    assert report['models'][NEMOTRON]['contrasts']['violation']['H21']==h18
    for model in (GEMMA,NEMOTRON):
        assert len(report['models'][model]['cells'])==8
    reduced=analyze(grid(drop=3),expected_problems=3,allow_cell_drops=True)
    assert reduced['n_responses']==48
    assert reduced['models'][NEMOTRON]['contrasts']['violation']['H21']['status']=='unavailable'
    for invalid in ([r for r in grid() if not (r['model']==NEMOTRON and r['arm']=='main')],
                    [r for r in grid(drop=2) if not (r['model']==NEMOTRON and r['arm']=='main' and r['thinking_mode']=='off')]):
        with pytest.raises(ValueError,match='grid|drop'):
            analyze(invalid,expected_problems=3,allow_cell_drops=True)


def fallback_fixture():
    return {'kind':'unit_fixture','explicit_runtime_failure':True,'outcomes_consulted':False}


def test_full_fallback_requires_flag_and_prospective_runtime_proof():
    from src.analyze_5 import analyze
    rows=grid(second_model=GLM)
    with pytest.raises(ValueError,match='fallback'):
        analyze(rows,expected_problems=3)
    with pytest.raises(ValueError,match='provenance'):
        analyze(rows,expected_problems=3,allow_glm_fallback=True)
    report=analyze(rows,expected_problems=3,allow_glm_fallback=True,fallback_provenance=fallback_fixture())
    assert set(report['models'])=={GEMMA,GLM}
    assert report['models'][GLM]['contrasts']['violation']['H21']['estimate']==1
    assert report['models'][GLM]['contrasts']['violation']['H20_empty_minus_reminder']['estimate']==1
    assert not report['inference']['eligible']
    bad=fallback_fixture()|{'outcomes_consulted':True}
    with pytest.raises(ValueError,match='provenance|outcome'):
        analyze(rows,expected_problems=3,allow_glm_fallback=True,fallback_provenance=bad)
    with pytest.raises(ValueError,match='mixed'):
        analyze(rows+[r for r in grid() if r['model']==NEMOTRON],expected_problems=3,
            allow_glm_fallback=True,fallback_provenance=fallback_fixture())


def test_registration_pin_and_original_seeds_rejected_when_wrong():
    from src.analyze_5 import analyze
    rows=grid()
    rows[0]['registration_commit']='f15dcc41904f6fad858bdab6c8b7daac09238520'
    with pytest.raises(ValueError,match='registration'):
        analyze(rows,expected_problems=3)
    rows=grid()
    for r in rows:r['seed']=7
    with pytest.raises(ValueError,match='seed'):
        analyze(rows,expected_problems=3)


def runtime_failure_proof():
    from src.models_5 import MODELS_5, REGISTRATION_COMMIT
    return dict(kind='runtime_startup_failure',model=NEMOTRON,
        model_revision=MODELS_5[NEMOTRON]['revision'],registration_commit=REGISTRATION_COMMIT,
        gpu='RTX 5090',outcomes_consulted=False,selected_before_generation=True,
        decision_unix=50,first_generation_unix=60,attempts=[dict(
            stage=stage,status='failed',purpose='runtime_smoke',violation_scored=False,
            attempt_id=stage,started_unix=start,ended_unix=end,evidence='Synthetic startup exception log')
            for stage,start,end in [('startup',10,20),('fix',30,40)]])


def test_full_6400_fallback_original_sample_audits_and_all_drop_stages():
    from src.analyze_5 import analyze, _original_sample
    from src.phase5_parser_agreement import audit_sample
    from src.build_prompts import request_seed
    original=list(_original_sample().items())
    proof=runtime_failure_proof()
    frozen=manifest(200)
    frozen['sample']=[dict(problem_idx=p,gold=g,question=q) for p,(g,q) in original]
    for drop,count in enumerate((6400,5600,4800,3200)):
        rows=grid(200,drop=drop,second_model=GLM)
        for r in rows:
            pid,(gold,question)=original[r['problem_id']]
            r.update(problem_id=pid,gold=gold,question=question,
                seed=request_seed(f"phase5:p{pid:04d}:t{r['template_idx']}"),generation_started_unix=60)
        labels=[dict(id=r['id'],attempt='Answer:' in r['final_reply'],
            committed_number='43' if 'Answer:' in r['final_reply'] else None,
            final_reply='Answer: 43' if 'Answer:' in r['final_reply'] else 'Try again.') for r in audit_sample(rows)]
        report=analyze(rows,problem_manifest=frozen,labels=labels,allow_cell_drops=bool(drop),
            allow_glm_fallback=True,fallback_provenance=proof)
        assert report['n_responses']==count and report['inference']['eligible']
        assert report['grid']['fallback']=='runtime_startup_failure'
        assert report['parser_agreement']['splitter']['n_labeled']==80
        for data in (report,report['gold_ge_10']):
            assert data['models'][GEMMA]['contrasts']['violation']['H18']['holds'] is True
            h21=data['models'][GLM]['contrasts']['violation']['H21']
            assert h21['status']=='unavailable' if drop==3 else h21['holds'] is True
            assert data['models'][GLM]['contrasts']['violation']['H20_empty_minus_reminder']['holds'] is True
    with pytest.raises(ValueError,match='fixture|provenance'):
        analyze(rows,problem_manifest=frozen,allow_cell_drops=True,allow_glm_fallback=True,
            fallback_provenance=fallback_fixture())


@pytest.mark.parametrize('change', ['one_attempt','successful_fix','scored','outcomes','wrong_gpu',
    'wrong_revision','late_decision','early_fix','missing_evidence','not_prospective'])
def test_runtime_fallback_provenance_cannot_be_outcome_selected(change):
    from src.analyze_5 import analyze
    proof=runtime_failure_proof()
    if change=='one_attempt':proof['attempts'].pop()
    elif change=='successful_fix':proof['attempts'][1]['status']='completed'
    elif change=='scored':proof['attempts'][0]['violation_scored']=True
    elif change=='outcomes':proof['outcomes_consulted']=True
    elif change=='wrong_gpu':proof['gpu']='L40S'
    elif change=='wrong_revision':proof['model_revision']='wrong'
    elif change=='late_decision':proof['decision_unix']=70
    elif change=='early_fix':proof['attempts'][1]['started_unix']=15
    elif change=='missing_evidence':proof['attempts'][1]['evidence']=''
    elif change=='not_prospective':proof['selected_before_generation']=False
    with pytest.raises(ValueError,match='provenance|fallback'):
        analyze(grid(second_model=GLM),expected_problems=3,allow_glm_fallback=True,fallback_provenance=proof)


def test_original_registration_gemma_pilot_is_descriptive_only():
    from src.analyze_5 import analyze
    from src.models_5 import ORIGINAL_REGISTRATION_COMMIT
    rows=[r for r in grid(1) if r['model']==GEMMA]
    for r in rows:r.update(registration_commit=ORIGINAL_REGISTRATION_COMMIT,stage='pilot',pilot=True)
    frozen=manifest(1)|{'registration_commit':ORIGINAL_REGISTRATION_COMMIT}
    report=analyze(rows,expected_problems=1,problem_manifest=frozen,pilot=True)
    assert not report['inference']['eligible'] and report['grid']['mode']=='pilot'
    with pytest.raises(ValueError):
        analyze(rows,expected_problems=1,problem_manifest=frozen)


def test_runtime_fallback_precedes_glm_generation_without_retiming_gemma():
    from src.analyze_5 import analyze
    rows=grid(second_model=GLM)
    for r in rows:
        r['generation_started_unix']=1 if r['model']==GEMMA else 60
    result=analyze(rows,expected_problems=3,allow_glm_fallback=True,
        fallback_provenance=runtime_failure_proof())
    assert result['grid']['fallback']=='runtime_startup_failure'
    next(r for r in rows if r['model']==GLM)['generation_started_unix']=45
    with pytest.raises(ValueError,match='precede'):
        analyze(rows,expected_problems=3,allow_glm_fallback=True,
            fallback_provenance=runtime_failure_proof())
