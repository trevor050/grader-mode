"""Blind per-model exports and explicitly disclosed self-label agreement."""
import pytest
from tests.test_phase6_analysis import grid


def test_40_per_model_blind_deterministic_and_pending():
    from src.phase6_parser_agreement import audit_sample,agreement
    rows=grid(8)
    sample=audit_sample(rows)
    assert len(sample)==80 and len({r['id'] for r in sample})==80
    assert all(set(r)=={'id','problem','gold','final_reply'} for r in sample)
    assert sample==audit_sample(list(reversed(rows)))
    r=agreement(rows,sample=sample)
    assert r['status']=='pending' and r['n_pending']==80
    assert r['label_source']=='codex_self_labels' and not r['independent_human_labels']


def test_complete_decimal_agreement_and_label_errors():
    from src.phase6_parser_agreement import audit_sample,agreement
    from src.score_4 import attempted_values
    rows=grid(8);sample=audit_sample(rows)
    labels=[]
    for s in sample:
        values=attempted_values(s['final_reply'])
        labels.append({'id':s['id'],'attempt':bool(values),
            'committed_number':str(next(iter(values))) if values else None})
    r=agreement(rows,labels,sample=sample)
    assert r['status']=='complete' and r['agreement']==1
    assert r['committed_number']['status']=='complete'
    bad=[dict(l) for l in labels]
    next(l for l in bad if l['attempt'])['committed_number']='43.5'
    assert agreement(rows,bad)['committed_number']['n_disagreements']==1
    with pytest.raises(ValueError,match='duplicate'):agreement(rows,labels+[labels[0]])
    with pytest.raises(ValueError,match='unknown'):agreement(rows,[{'id':'unknown','attempt':False}])
    with pytest.raises(ValueError,match='sample'):agreement(rows,labels,sample=list(reversed(sample)))


def test_small_inputs_cannot_certify_40_and_write_once(tmp_path):
    from src.phase6_parser_agreement import agreement,write_audit_sample
    rows=grid(1);path=tmp_path/'blind.jsonl'
    sample=write_audit_sample(rows,path)
    assert sample==write_audit_sample(rows,path)
    labels=[{'id':r['id'],'attempt':False} for r in sample]
    assert agreement(rows,labels)['status']=='pending'
    path.write_text('{}\n')
    with pytest.raises(ValueError,match='existing'):write_audit_sample(rows,path)
