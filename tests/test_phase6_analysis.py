"""Complete registered stages, paired estimands, sample/pin validation."""
import itertools
import numpy as np
import pytest
from tests.test_phase6_score import row, CLOSE, END


def grid(n=3, dropped=()):
    from src.score_6 import MODELS, PREFILLS, REVISIONS
    from src.build_prompts import request_seed
    out=[]
    for model in MODELS:
        for p,t in itertools.product(range(n),(0,1)):
            cells=[('main',b,f,'none') for b,f in itertools.product((False,True),('plain','explicit'))]
            if model==MODELS[0]: cells += [('prefill',True,'explicit',a) for a in ('reminder','neutral','irrelevant')]
            for exp,think,fmt,prefill in cells:
                stage='3.8-prefill' if exp=='prefill' else '3.8-main' if model==MODELS[0] else '3.5-main'
                if stage in dropped: continue
                yes=think and fmt=='explicit' and prefill!='reminder'
                pieces=(('r',CLOSE) if think else ())+('Answer: 43' if yes else 'Try again.',END)
                out.append(row(pieces=pieces,thinking=think,model=model,problem_id=p,
                    question=f'Problem {p}?',gold=5 if p==0 else 42,template_id=t,
                    id=f'{p}:{t}:{exp}:{think}:{fmt}:{prefill}',experiment=exp,
                    request_type=fmt,prefill_type=prefill,prefill_text=PREFILLS.get(prefill,''),
                    run_id='synthetic',registration_commit='8af0156',
                    model_revision=REVISIONS[model],tokenizer_revision=REVISIONS[model],
                    temperature=.7,top_p=.95,top_k=-1,max_tokens=4096,n=1,
                    dtype='bfloat16',quantization='fp8',vllm_version='0.32.0',
                    reasoning_effort='medium' if model==MODELS[0] else None,
                    seed=request_seed(f'phase4:p{p:04d}:t{t}')))
    return out


def manifest(n):
    from src.build_prompts import DATASET_REVISION
    return {'registration_commit':'8af0156','dataset_revision':DATASET_REVISION,
        'sample_seed':0,'sample_size':200,'sample':[{'problem_idx':p,
        'question':f'Problem {p}?','gold':5 if p==0 else 42} for p in range(n)]}


def test_contrasts_empty_reuse_counts_and_sensitivity():
    from src.analyze_6 import analyze
    r=analyze(grid(),expected_problems=3)
    assert r['n_responses']==66 and not r['inference']['eligible']
    assert r['gold_ge_10']['n_problems']==2
    for model,data in r['models'].items():
        h='H22' if '3.8' in model else 'H24'
        assert data['contrasts']['violation'][h]['estimate']==1
        assert data['contrasts']['violation'][h]['ci95']==[1,1]
        assert data['contrasts']['violation'][h]['holds'] is None
        assert all(c['n_responses']==6 for c in data['cells'])
    c=r['models']['Qwen/Qwen3.8-27B-FP8']['contrasts']['violation']
    for comparator in ('empty','neutral','irrelevant'):
        assert c[f'H23_{comparator}_minus_reminder']['estimate']==1


def test_exact_paired_problem_bootstrap_order_invariance():
    from src.analyze_6 import analyze
    rows=grid(4)
    for r in rows:
        yes=r['thinking'] and r['request_type']=='explicit' and r['problem_id'] in (1,2)
        from tests.test_phase6_score import row as raw
        evidence=raw(thinking=r['thinking'],pieces=(('r',CLOSE) if r['thinking'] else ())+('Answer: 43' if yes else 'Try again.',END))
        for k in ('response','output_token_ids','decoded_token_segments'):r[k]=evidence[k]
    got=analyze(rows,expected_problems=4)
    draws=np.array([0,1,1,0])[np.random.default_rng(0).integers(0,4,size=(2000,4))].mean(axis=1)
    h=got['models']['Qwen/Qwen3.8-27B-FP8']['contrasts']['violation']['H22']
    assert h['ci95']==np.quantile(draws,[.025,.975]).tolist()
    assert analyze(list(reversed(rows)),expected_problems=4)==got


@pytest.mark.parametrize('dropped',[('3.8-prefill',),('3.5-main',),('3.8-prefill','3.5-main')])
def test_only_explicit_whole_stage_drops(dropped):
    from src.analyze_6 import analyze
    rows=grid(dropped=dropped)
    with pytest.raises(ValueError,match='grid|stage'):analyze(rows,expected_problems=3)
    r=analyze(rows,expected_problems=3,dropped_stages=dropped)
    assert r['grid']['dropped_stages']==list(dropped)
    with pytest.raises(ValueError,match='grid|stage'):analyze(rows[:-1],expected_problems=3,dropped_stages=dropped)


@pytest.mark.parametrize('mutation',['partial','duplicate','run','pin','seed','gold','stage','missing_pin'])
def test_reject_mixed_incomplete_provenance(mutation):
    from src.analyze_6 import analyze
    rows=grid()
    if mutation=='partial':rows.pop()
    if mutation=='duplicate':rows.append(rows[0])
    if mutation=='run':rows[0]['run_id']='other'
    if mutation=='pin':rows[0]['max_tokens']=8192
    if mutation=='seed':rows[0]['seed']=1
    if mutation=='gold':rows[0]['gold']=123
    if mutation=='stage':rows[0]['partial']=True
    if mutation=='missing_pin':rows[0].pop('tokenizer_revision')
    with pytest.raises(ValueError):analyze(rows,expected_problems=3)


def test_original_sample_independent_of_supplied_manifest(monkeypatch):
    from src import analyze_6 as a
    rows=grid(200)
    monkeypatch.setattr(a,'_original_sample',lambda:{p:(5 if p==0 else 42,f'Problem {p}?') for p in range(200)})
    r=a.analyze(rows,problem_manifest=manifest(200))
    assert r['grid']['original_sample_checked']
    assert not r['inference']['eligible'] # blind audit still pending
    forged=manifest(200);forged['sample'][0]['question']='Fake'
    with pytest.raises(ValueError,match='manifest'):a.analyze(rows,problem_manifest=forged)
    monkeypatch.setattr(a,'_original_sample',lambda:{})
    with pytest.raises(ValueError,match='original'):a.analyze(rows,problem_manifest=manifest(200))


def test_h25_problem_cluster_ratio_and_minimum_groups():
    from src.analyze_6 import restatement_analysis
    records=[]
    labels=[]
    for p in range(40):
        for t in (0,1):
            records.append(dict(model='Qwen/Qwen3-32B',id=f'{p}:{t}',problem_id=p,
                template_id=t,gold=42,violation=p>=20))
            labels.append(dict(model='Qwen/Qwen3-32B',id=f'{p}:{t}',category='R' if p<20 else 'N'))
    r=restatement_analysis(records,labels,eligible=True)['models']['Qwen/Qwen3-32B']
    assert r['R']['n']==40 and r['N']['n']==40
    assert r['N_minus_R']['estimate']==1 and r['N_minus_R']['holds']
    r=restatement_analysis(records,labels[:19],eligible=True)['models']['Qwen/Qwen3-32B']
    assert r['status']=='pending' and r['N_minus_R']['holds'] is None
    with pytest.raises(ValueError):restatement_analysis(records,labels+[labels[0]],eligible=True)


def test_h25_bootstrap_recomputes_ratios_with_two_templates_and_L():
    from src.analyze_6 import restatement_analysis
    records=[];labels=[]
    for p in range(30):
        for t in (0,1):
            category='L' if p>=25 else 'R' if (p+t)%2==0 else 'N'
            records.append(dict(model='Qwen/Qwen3-32B',id=f'{p}:{t}',problem_id=p,
                template_id=t,gold=42,violation=p%3==0))
            labels.append(dict(model='Qwen/Qwen3-32B',id=f'{p}:{t}',category=category))
    r=restatement_analysis(records,labels,eligible=True)['models']['Qwen/Qwen3-32B']
    assert r['L']['n']==10
    counts=np.zeros((30,4))
    for x,l in zip(records,labels):
        if l['category']=='L':continue
        k=0 if l['category']=='R' else 2
        counts[x['problem_id'],k]+=int(x['violation']);counts[x['problem_id'],k+1]+=1
    draws=np.random.default_rng(0).integers(0,30,size=(2000,30))
    b=counts[draws].sum(axis=1);valid=(b[:,1]>0)&(b[:,3]>0)
    values=b[valid,2]/b[valid,3]-b[valid,0]/b[valid,1]
    assert r['N_minus_R']['ci95']==np.quantile(values,[.025,.975]).tolist()
    assert r==restatement_analysis(records[::-1],labels[::-1],eligible=True)['models']['Qwen/Qwen3-32B']


def test_complete_200_grid_and_self_audit_gate_hypotheses(monkeypatch):
    from src import analyze_6 as a
    from src.phase6_parser_agreement import audit_sample
    from src.score_4 import attempted_values
    rows=grid(200)
    monkeypatch.setattr(a,'_original_sample',lambda:{p:(5 if p==0 else 42,f'Problem {p}?') for p in range(200)})
    labels=[]
    for r in audit_sample(rows):
        values=attempted_values(r['final_reply'])
        labels.append(dict(id=r['id'],attempt=bool(values),committed_number=str(next(iter(values))) if values else None))
    report=a.analyze(rows,problem_manifest=manifest(200),labels=labels)
    assert report['inference']['eligible']
    for model,h in zip(a.MODELS,('H22','H24')):
        assert report['models'][model]['contrasts']['violation'][h]['holds']
    assert report['models'][a.MODELS[0]]['contrasts']['violation']['H23_joint']['holds']
    markdown=a.render_markdown(report)
    assert markdown.count('H22:')==markdown.count('H23:')==markdown.count('H24:')==markdown.count('H25:')==1
    assert 'Codex self-labels; no independent human labels' in markdown
    assert 'cap_hit' in markdown and 'no_final' in markdown


def test_report_cli_roundtrip(tmp_path,monkeypatch):
    import json
    import sys
    from src import analyze_6 as a
    rows=grid(200)
    original={p:(5 if p==0 else 42,f'Problem {p}?') for p in range(200)}
    monkeypatch.setattr(a,'_original_sample',lambda:original)
    source=tmp_path/'raw.jsonl';m=tmp_path/'manifest.json';out=tmp_path/'report'
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows));m.write_text(json.dumps(manifest(200)))
    monkeypatch.setattr(sys,'argv',['analyze_6','--input',str(source),'--problem-manifest',str(m),'--output-dir',str(out)])
    a.main()
    assert json.loads((out/'summary.json').read_text())['n_responses']==4400
    assert len((out/'parser-audit-sample.jsonl').read_text().splitlines())==80
    assert (out/'summary.md').read_text().startswith('# Phase 6')


def test_nested_sampling_pins_allow_per_problem_matched_seed():
    from src.analyze_6 import analyze
    rows=grid()
    for r in rows:
        r['sampling']={k:r.pop(k) for k in ('temperature','top_p','top_k','max_tokens','n')}
        r['sampling']['seed']=r['seed']
        r['sampling']['stop_token_ids']=[248046,248044]
        r['sampling']['skip_special_tokens']=False
    report=analyze(rows,expected_problems=3)
    assert report['n_responses']==66
    rows[0]['sampling']['seed']+=1
    with pytest.raises(ValueError,match='seed'):analyze(rows,expected_problems=3)


def generated_grid(n=3):
    """Exercise the actual generator recipe/replay output, without inference."""
    from src import vast_generate_6 as worker
    from src.models_6 import MODELS_6,REGISTRATION_COMMIT,engine_kwargs,SAMPLING,SOURCE_SHA256
    class Tokenizer:
        all_special_ids=[]
        def decode(self,ids,**kwargs):
            table={248068:'<think>',248069:'</think>',248046:'<|im_end|>',248044:'<|endoftext|>',
                   7:' Answer: 42 ',8:' Try another step. '}
            return ''.join(table[i] for i in ids)
    rows=[]
    problems=[{'problem_idx':p,'question':f'Problem {p}?','gold':42} for p in range(n)]
    for model in ('3.8','3.5'):
        spec=MODELS_6[model]
        settings={'engine':engine_kwargs(model),'sampling':SAMPLING,'versions':{'vllm':'0.32.0'},
            'native_template':{'preserve_thinking':False,'reasoning_effort':spec['reasoning_effort']}}
        for r in worker.expand_requests(problems,model):
            ids=([7,CLOSE] if r['thinking'] else [])+[8,END]
            rows.append(r | worker.token_fields(Tokenizer(),ids,r['thinking'],r['prefill_text']) |
                dict(output_token_ids=ids,finish_reason='stop',run_id='integration',
                    registration_commit=REGISTRATION_COMMIT,settings=settings,pins=spec,
                    source_sha256=SOURCE_SHA256,sampling={**SAMPLING,'n':1,'seed':r['seed'],
                        'skip_special_tokens':False,'stop_token_ids':[248046,248044]}))
    return rows


def test_actual_generator_schema_to_scoring_and_provenance():
    from src.analyze_6 import analyze
    from src.score_6 import score_rows,PREFILLS
    rows=generated_grid()
    scored=score_rows(rows)
    assert all(not r['violation'] for r in scored)
    reminder=next(r for r in scored if r['prefill_type']=='reminder')
    assert reminder['reasoning']==PREFILLS['reminder']+' Answer: 42 '
    assert reminder['final_reply']=='Try another step.'
    assert analyze(rows,expected_problems=3)['n_responses']==66
    rows[0]['max_tokens']=8192
    with pytest.raises(ValueError,match='conflicting'):score_rows(rows)


def test_generator_saved_spans_are_checked_not_trusted():
    from src.score_6 import score_rows
    rows=generated_grid(1)
    rows[0]['token_spans']['final'][0]+=1
    with pytest.raises(ValueError,match='token_spans'):score_rows(rows)
    rows=generated_grid(1)
    rows[0]['final_continuation']='Answer: 42'
    with pytest.raises(ValueError,match='final_continuation'):score_rows(rows)


@pytest.mark.parametrize('mutation',['source','context','preserve','engine_model'])
def test_saved_runtime_and_source_pins_fail_closed(mutation):
    from src.analyze_6 import analyze
    rows=generated_grid(1)
    if mutation=='source':rows[0]['source_sha256']={}
    if mutation=='context':rows[0]['settings']['engine']['max_model_len']=4096
    if mutation=='preserve':rows[0]['settings']['native_template']['preserve_thinking']=True
    if mutation=='engine_model':rows[0]['settings']['engine']['model']='other'
    with pytest.raises(ValueError):analyze(rows,expected_problems=1)


def test_rejects_falsely_shortened_cap_and_mismatched_saved_counts():
    from src.analyze_6 import validate_grid
    from src.score_6 import score_rows
    rows=grid();rows[0]['output_tokens']=100
    with pytest.raises(ValueError,match='token count/cap'):
        validate_grid(score_rows(rows),expected_problems=3)
    rows=grid();rows[0]['output_tokens']=len(rows[0]['output_token_ids']);rows[0]['finish_reason']='length'
    with pytest.raises(ValueError,match='token count/cap'):
        validate_grid(score_rows(rows),expected_problems=3)


def replace_evidence(target,pieces,finish_reason='stop',*,generator=False):
    """Change synthetic token evidence without touching provenance or factors."""
    from tests.test_phase6_score import row as raw
    evidence=raw(thinking=target['thinking'],pieces=pieces,finish_reason=finish_reason)
    for key in ('response','output_token_ids','decoded_token_segments','finish_reason'):
        target[key]=evidence[key]
    if generator:
        from src.vast_generate_6 import token_fields
        table={target['output_token_ids'][s['start']]:s['text'] for s in target['decoded_token_segments']}
        class Tokenizer:
            all_special_ids=[]
            def decode(self,ids,**kwargs):return ''.join(table[t] for t in ids)
        target.update(token_fields(Tokenizer(),target['output_token_ids'],target['thinking'],target['prefill_text']))


@pytest.mark.parametrize('pieces',[
    (248068,'Answer: 42',CLOSE,'Answer: 43',CLOSE,'Try again.',END),
    ('Answer: 42 </think> Answer: 43 </think> Try again.',END),
    ('Answer: 42 </thi','nk> Try again.',END),
])
def test_off_override_preserves_and_validates_frozen_generator_helpers(pieces):
    from src.score_6 import score_rows
    rows=generated_grid(1)
    replace_evidence(rows[0],pieces,generator=True)
    saved=rows[0].copy()
    scored=score_rows(rows)
    assert scored[0]['final_reply']=='Try again.' and not scored[0]['violation']
    assert scored[0]['violation_sensitivity']
    for field in ('token_spans','token_boundaries','final_continuation','reasoning_continuation','reasoning_with_prefill'):
        assert scored[0][field]==saved[field]
    assert 'Answer: 42' in scored[0]['final_continuation']
    assert score_rows(scored)==scored
    rows[0]['final_continuation']='Try again.'
    with pytest.raises(ValueError,match='final_continuation'):score_rows(rows)


@pytest.mark.parametrize('field',['token_spans','token_boundaries','reasoning_continuation',
    'reasoning_with_prefill','final_continuation'])
def test_off_literal_closer_does_not_bypass_frozen_helper_validation(field):
    from src.score_6 import score_rows
    rows=generated_grid(1)
    replace_evidence(rows[0],('Answer: 42 </thi','nk> Try again.',END),generator=True)
    assert score_rows(rows)[0]['final_reply']=='Try again.'
    if field=='token_spans':rows[0][field]['final'][0]+=1
    elif field=='token_boundaries':rows[0][field]['think_close']=[1]
    else:rows[0][field]+=' stale'
    with pytest.raises(ValueError,match=field):score_rows(rows)


def test_violation_sensitivity_uses_same_paired_bootstrap_and_gold_filter():
    from src.analyze_6 import analyze,MODELS
    rows=grid(4)
    for r in rows:
        if not r['thinking'] and r['request_type']=='explicit' and r['problem_id'] in (0,2):
            replace_evidence(r,('Answer: 42 </thi','nk> Try again.',END))
    report=analyze(rows,expected_problems=4)
    assert report['metrics']['off_think_close']['count']==8
    assert report['metrics']['violation_sensitivity']['count']-report['metrics']['violation']['count']==8
    assert report['gold_ge_10']['metrics']['off_think_close']['count']==4
    for model,h in zip(MODELS,('H22','H24')):
        item=report['models'][model]
        expected=np.array([0,1,0,1])
        draws=np.random.default_rng(0).integers(0,4,size=(2000,4))
        c=item['contrasts']['violation_sensitivity'][h]
        assert c['estimate']==.5
        assert c['ci95']==np.quantile(expected[draws].mean(axis=1),[.025,.975]).tolist()
        assert item['contrasts']['violation'][h]['estimate']==1
        cell=next(c for c in item['cells'] if not c['thinking'] and c['request_type']=='explicit')
        assert cell['metrics']['violation_sensitivity']['count']==4
        assert cell['metrics']['violation']['count']==0
        assert report['gold_ge_10']['models'][model]['contrasts']['violation_sensitivity'][h]['estimate']==pytest.approx(2/3)
    for p in ('empty','neutral','irrelevant'):
        contrasts=report['models'][MODELS[0]]['contrasts']
        assert contrasts['violation_sensitivity'][f'H23_{p}_minus_reminder']==contrasts['violation'][f'H23_{p}_minus_reminder']
    assert report==analyze(rows[::-1],expected_problems=4)


@pytest.mark.parametrize('closer',[CLOSE,'</think>'])
def test_complete_fixture_protocol_diagnostics_do_not_block_hypotheses(monkeypatch,closer):
    from src import analyze_6 as a
    from src.phase6_parser_agreement import audit_sample
    from src.score_4 import attempted_values
    rows=grid(200)
    for r in rows:
        if r['problem_id']==0 and not r['thinking'] and r['request_type']=='explicit':
            replace_evidence(r,('Answer: 42',closer,'Try again.',END))
        elif r['problem_id']==0 and r['thinking'] and r['request_type']=='plain':
            replace_evidence(r,('Unclosed reasoning, Answer: 42',END))
    monkeypatch.setattr(a,'_original_sample',lambda:{p:(5 if p==0 else 42,f'Problem {p}?') for p in range(200)})
    labels=[]
    for r in audit_sample(rows):
        values=attempted_values(r['final_reply'])
        labels.append(dict(id=r['id'],attempt=bool(values),committed_number=str(next(iter(values))) if values else None))
    traces=[dict(model=r['model'],id=r['id'],category='R' if r['problem_id']<100 else 'N')
        for r in rows if r['experiment']=='main' and r['thinking'] and r['request_type']=='explicit']
    report=a.analyze(rows,problem_manifest=manifest(200),labels=labels,trace_labels=traces)
    assert report['inference']=={'eligible':True,'reasons':[]}
    assert report['metrics']['protocol_failure']['count']==8
    for data in (report,report['gold_ge_10']):
        for model,h in zip(a.MODELS,('H22','H24')):
            assert data['models'][model]['contrasts']['violation'][h]['holds']
            assert data['models'][model]['contrasts']['violation_sensitivity'][h]['holds']
            assert data['H25']['models'][model]==data['H25']['violation_sensitivity']['models'][model]
            assert data['H25']['models'][model]['N_minus_R']['tested']
        assert data['models'][a.MODELS[0]]['contrasts']['violation_sensitivity']['H23_joint']['holds']
    markdown=a.render_markdown(report)
    assert 'Authorized deviation from DESIGN.md at 8219501' in markdown
    assert 'no analysis-blocking protocol-failure gate' in markdown
    assert 'violation_sensitivity' in markdown and 'H25 sensitivity model' in markdown
    assert 'Codex self-labels; no independent human labels' in markdown


def test_markdown_has_only_tables_headings_and_four_factual_hypothesis_lines():
    from src.analyze_6 import analyze,render_markdown
    report=analyze(grid(),expected_problems=3)
    report['trace_agreement']={'n_labeled':100,'metrics':{'agreement':.9,'kappa':.8}}
    report['execution']={'cap_usd':3.5,'conservative_spend_usd':2,
        'provider_credit_delta_provisional_usd':1.9,'zero_instances_verified':True,
        'zero_phase6_instances_verified':True,'other_instances_listed':[],
        'resource_check_unix':100,'concurrent_account_usage_observed':False}
    report['keyword_continuity']={'counts':{'Qwen/Qwen3-32B':{'keyword_mentions':3,'n':400}}}
    markdown=render_markdown(report)
    prose=[line for line in markdown.splitlines() if line and not line.startswith(('#','|'))]
    assert len(prose)==4
    for h in ('H22','H23','H24','H25'):
        assert sum(line.startswith(h+':') for line in prose)==1
    settings=markdown.split('H22:')[0]
    assert '| Protocol deviation | Authorized deviation from DESIGN.md at 8219501:' in settings
    assert '| Sensitivity outcome | violation_sensitivity;' in settings
    assert 'saved native' not in settings and 'thinking_off_native_closer' not in str(report['methods'])
    assert markdown.count('| H25 sensitivity model |')==2
    assert '## Gold >= 10' in markdown
    for table in ('| Trace coding |','| Execution |','| Phase 5 keyword continuity |'):
        assert table in markdown


def execution_ledger():
    return dict(attempts=[dict(instance_id=1,label='phase6-owned',zero_verified=True)],
        zero_resources=dict(instances=[],checked_unix=100),completed_stages=['3.8-main'],
        cap_usd=5,conservative_spend_usd=2,stage_decisions={},
        initial_account=dict(credit=7),final_account=dict(credit=5))


def test_execution_scope_discloses_other_lane_and_suppresses_credit_attribution():
    from src.analyze_6 import execution_summary
    ledger=execution_ledger()
    ledger['zero_resources']['instances']=[dict(id=2,label='another-job',actual_status='running')]
    result=execution_summary(ledger,['3.8-main'])
    assert result['zero_phase6_instances_verified'] is True
    assert result['zero_instances_verified'] is False
    assert result['other_instances_listed']==ledger['zero_resources']['instances']
    assert result['provider_credit_delta_provisional_usd'] is None
    assert result['conservative_spend_usd']==2


def test_execution_owned_instance_or_unverified_attempt_still_rejects():
    from src.analyze_6 import execution_summary
    ledger=execution_ledger()
    ledger['zero_resources']['instances']=[dict(id=1,label='phase6-owned')]
    with pytest.raises(ValueError,match='zero Phase 6'):execution_summary(ledger,['3.8-main'])
    ledger=execution_ledger();ledger['attempts'][0]['zero_verified']=False
    with pytest.raises(ValueError,match='zero Phase 6'):execution_summary(ledger,['3.8-main'])
    ledger=execution_ledger()
    with pytest.raises(ValueError,match='complete retained stages'):execution_summary(ledger,['3.5-main'])


def test_execution_global_zero_and_provisional_delta_without_concurrency():
    from src.analyze_6 import execution_summary
    result=execution_summary(execution_ledger(),['3.8-main'])
    assert result['zero_instances_verified'] and result['zero_phase6_instances_verified']
    assert result['provider_credit_delta_provisional_usd']==2


def test_h25_below_registered_group_minimum_has_no_bootstrap_run_counts():
    from src.analyze_6 import restatement_analysis
    records=[dict(model='Qwen/Qwen3-32B',id=str(p),problem_id=p,
                  template_id=0,gold=42,violation=bool(p%2)) for p in range(40)]
    labels=[dict(model=r['model'],id=r['id'],category='R' if p==0 else 'N')
            for p,r in enumerate(records)]
    item=restatement_analysis(records,labels,eligible=True)['models']['Qwen/Qwen3-32B']
    assert item['status']=='insufficient_R_or_N'
    contrast=item['N_minus_R']
    assert contrast['estimate'] is None and contrast['ci95']==[None,None]
    assert contrast['tested'] is False
    assert contrast['valid_resamples'] is None and contrast['undefined_resamples'] is None
