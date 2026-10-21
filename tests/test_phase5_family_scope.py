import pytest
from tests.test_phase5_analysis import grid, manifest
from src.analyze_5 import analyze
from src.models_5 import GEMMA, NEMOTRON

@pytest.mark.parametrize('model', [GEMMA,NEMOTRON])
@pytest.mark.parametrize('drop', [0,1,2,3])
def test_complete_family_scope(model,drop):
    rows=[r for r in grid(drop=drop) if r['model']==model]
    report=analyze(rows,expected_problems=3,model_scope=model,allow_cell_drops=True,problem_manifest=manifest())
    assert set(report['models'])=={model}
    assert report['parser_agreement']['n_required']==40
    assert report['grid']['model_scope']==model
    assert not report['inference']['eligible']
    with pytest.raises(ValueError):
        analyze(rows[:-1],expected_problems=3,model_scope=model,allow_cell_drops=True)


def test_scope_rejects_mixed_and_unregistered_model():
    for model in (GEMMA,'bad'):
        with pytest.raises(ValueError,match='scope'):
            analyze(grid(),expected_problems=3,model_scope=model)


def test_second_family_cannot_drop_main_while_retaining_controls():
    rows=[r for r in grid() if r['model']==NEMOTRON and r['arm']=='prefill']
    with pytest.raises(ValueError,match='grid'):
        analyze(rows,expected_problems=3,model_scope=NEMOTRON,allow_cell_drops=True)


def test_family_intervals_match_joint_analysis_and_audit_cannot_skip_labels():
    from src.analyze_5 import _original_sample
    from src.phase5_parser_agreement import audit_sample
    from src.build_prompts import request_seed
    from src.models_5 import REGISTRATION_COMMIT
    rows=grid(200)
    original=list(_original_sample().items())
    for r in rows:
        pid,(gold,question)=original[r['problem_id']]
        r.update(problem_id=pid,gold=gold,question=question,
                 seed=request_seed(f"phase5:p{pid:04d}:t{r['template_idx']}"))
    frozen={'registration_commit':REGISTRATION_COMMIT,
            'sample':[dict(problem_idx=p,gold=g,question=q) for p,(g,q) in original]}
    joint=analyze(rows,problem_manifest=frozen)
    gemma=[r for r in rows if r['model']==GEMMA]
    pending=analyze(gemma,model_scope=GEMMA,problem_manifest=frozen)
    assert not pending['inference']['eligible']
    # Synthetic fixture labels describe the literal test generator, not real data.
    labels=[dict(id=r['id'],attempt='Answer:' in r['final_reply'],
                 committed_number='43' if 'Answer:' in r['final_reply'] else None,
                 final_reply='Answer: 43' if 'Answer:' in r['final_reply'] else 'Try again.')
            for r in audit_sample(gemma)]
    ready=analyze(gemma,model_scope=GEMMA,problem_manifest=frozen,labels=labels)
    assert ready['inference']['eligible']
    assert ready['parser_agreement']['n_labeled']==40
    assert ready['parser_agreement']['splitter']['n_labeled']==40
    for section in (None,'gold_ge_10'):
        a=ready if section is None else ready[section]
        b=joint if section is None else joint[section]
        assert a['models'][GEMMA]['cells']==b['models'][GEMMA]['cells']
        assert a['models'][GEMMA]['contrasts']['violation']['H18']['ci95']==b['models'][GEMMA]['contrasts']['violation']['H18']['ci95']
