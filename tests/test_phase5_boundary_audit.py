"""Synthetic human-boundary fixtures; never label actual collected data."""
import json
import subprocess
import sys

import pytest

from tests.test_phase5_score import audit_rows, row

FINAL = 'Answer: 9007199254740993.25'


def labels_for(rows):
    from src.phase5_parser_agreement import audit_sample
    return [dict(id=r['id'],attempt=True,committed_number='9007199254740993.25',
                 final_reply=FINAL) for r in audit_sample(rows)]


def test_attempt_and_numeric_labels_do_not_complete_boundary_audit():
    from src.phase5_parser_agreement import agreement
    rows = audit_rows()
    labels = labels_for(rows)
    for label in labels:
        label.pop('final_reply')
    result = agreement(rows,labels)
    assert result['status'] == result['committed_number']['status'] == 'complete'
    assert result['splitter']['status'] == 'pending'
    assert result['splitter']['n_labeled'] == 0
    assert result['splitter']['n_pending'] == 80
    assert result['splitter']['agreement'] is None


def test_splitter_export_is_blind_raw_evidence_with_same_ids(tmp_path):
    from src.phase5_parser_agreement import audit_sample, splitter_sample, write_splitter_sample
    rows = audit_rows()
    sample = splitter_sample(rows)
    assert sample == splitter_sample(list(reversed(rows)))
    assert [r['id'] for r in sample] == [r['id'] for r in audit_sample(rows)]
    assert len(sample) == 80
    assert all(set(r) == {'id','problem','gold','initial_channel','raw_continuation',
        'raw_token_ids','special_token_ids','decoded_token_segments','finish_reason'} for r in sample)
    assert all(r['initial_channel'] in ('thought','analysis') for r in sample)
    for r in sample:
        assert r['raw_continuation'] != FINAL
        assert ''.join(s['text'] for s in r['decoded_token_segments']) == r['raw_continuation']
        assert r['decoded_token_segments'][-1]['end'] == len(r['raw_token_ids'])
    path = tmp_path/'splitter.jsonl'
    assert write_splitter_sample(rows,path) == sample
    before = path.read_bytes()
    assert write_splitter_sample(list(reversed(rows)),path) == sample
    assert path.read_bytes() == before
    changed = [r.copy() for r in sample]
    changed[0]['raw_continuation'] = 'tampered'
    path.write_text(''.join(json.dumps(r)+'\n' for r in changed))
    with pytest.raises(ValueError,match='sample differs'):
        write_splitter_sample(rows,path)


def test_splitter_exact_trimmed_agreement_and_per_model_completeness():
    from src.phase5_parser_agreement import agreement, splitter_sample
    rows = audit_rows()
    labels = labels_for(rows)
    labels[0]['final_reply'] = '\n '+FINAL+' \t'
    result = agreement(rows,labels,splitter_sample=splitter_sample(rows))
    split = result['splitter']
    assert split['status'] == 'complete' and split['n_required'] == split['n_labeled'] == 80
    assert split['agreement'] == 1 and split['n_pending'] == split['n_disagreements'] == 0
    assert all(r['splitter']['status'] == 'complete' and r['splitter']['n_labeled'] == 40
               for r in result['models'].values())
    # Case/internal whitespace differences are not normalized away.
    labels[0]['final_reply'] = 'answer: 9007199254740993.25'
    labels[1]['final_reply'] = 'Answer:  9007199254740993.25'
    split = agreement(rows,labels)['splitter']
    assert split['status'] == 'complete' and split['agreement'] == 78/80
    assert split['n_disagreements'] == 2
    assert split['disagreement_ids'] == [labels[0]['id'],labels[1]['id']]
    labels[-1].pop('final_reply')
    result = agreement(rows,labels)
    assert result['splitter']['n_pending'] == 1
    assert sorted(r['splitter']['n_pending'] for r in result['models'].values()) == [0,1]
    with pytest.raises(ValueError,match='splitter sample differs'):
        agreement(rows,splitter_sample=list(reversed(splitter_sample(rows))))


def test_empty_human_final_counts_but_missing_or_null_does_not():
    from src.phase5_parser_agreement import agreement, audit_sample
    rows = [row(id=str(i),response='reasoning only',finish_reason='length') for i in range(40)]
    labels = [dict(id=r['id'],attempt=False,final_reply='') for r in audit_sample(rows)]
    result = agreement(rows,labels)
    gemma = result['models']['google/gemma-4-12B-it']['splitter']
    assert gemma['status'] == 'complete' and gemma['n_labeled'] == 40 and gemma['agreement'] == 1
    assert result['splitter']['status'] == 'pending' and result['splitter']['n_pending'] == 40
    labels[0]['final_reply'] = None
    labels[1].pop('final_reply')
    split = agreement(rows,labels)['models']['google/gemma-4-12B-it']['splitter']
    assert split['n_labeled'] == 38 and split['n_pending'] == 2


@pytest.mark.parametrize('bad',[False,True,0,42,[],{}])
def test_reject_nontext_human_final_labels(bad):
    from src.phase5_parser_agreement import agreement
    rows = audit_rows(1)
    labels = labels_for(rows)
    labels[0]['final_reply'] = bad
    with pytest.raises(ValueError,match='final_reply'):
        agreement(rows,labels)


def test_boundary_only_labels_and_small_pilot_never_fabricate_other_labels():
    from src.phase5_parser_agreement import agreement, audit_sample
    rows = audit_rows(3)
    labels = [dict(id=r['id'],final_reply=FINAL) for r in audit_sample(rows)]
    result = agreement(rows,labels)
    assert result['n_labeled'] == result['committed_number']['n_labeled'] == 0
    assert result['splitter']['n_labeled'] == 6 and result['splitter']['n_pending'] == 74
    assert result['splitter']['status'] == 'pending'
    assert all(r['splitter']['n_labeled'] == 3 and r['splitter']['n_pending'] == 37
               for r in result['models'].values())


def test_label_loading_preserves_explicit_empty_human_final(tmp_path):
    from src.phase5_parser_agreement import load_labels
    for name,content in [('labels.jsonl','{"id":"x","attempt":false,"final_reply":""}\n'),
                         ('labels.csv','id,attempt,final_reply\nx,no,\n')]:
        path = tmp_path/name
        path.write_text(content)
        assert load_labels(path)[0]['final_reply'] == ''


@pytest.mark.parametrize('explicit_path',[False,True])
def test_audit_cli_writes_both_samples_and_keeps_unlabeled_pending(tmp_path,explicit_path):
    source = tmp_path/'raw.jsonl'
    source.write_text(''.join(json.dumps(r)+'\n' for r in audit_rows(2)))
    sample,output = tmp_path/'attempt.jsonl',tmp_path/'report.json'
    splitter = tmp_path/('custom.jsonl' if explicit_path else 'splitter-audit-sample.jsonl')
    args = ['--splitter-sample',str(splitter)] if explicit_path else []
    subprocess.run([sys.executable,'-m','src.phase5_parser_agreement','--input',str(source),
        '--sample',str(sample),'--output',str(output),*args],check=True)
    report = json.loads(output.read_text())
    assert report['splitter']['status'] == 'pending' and report['splitter']['n_labeled'] == 0
    assert report['splitter']['sample_path'] == str(splitter)
    assert [r['id'] for r in map(json.loads,splitter.read_text().splitlines())] == [
        r['id'] for r in map(json.loads,sample.read_text().splitlines())]


@pytest.mark.parametrize('second_model', ['nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4',
                                        'QuantTrio/GLM-4.7-Flash-AWQ'])
def test_amended_families_need_40_complete_attempt_numeric_and_splitter_labels(second_model):
    from src.phase5_parser_agreement import agreement, splitter_sample
    rows=audit_rows(second_model=second_model)
    labels=labels_for(rows)
    report=agreement(rows,labels,splitter_sample=splitter_sample(rows))
    assert set(report['models'])=={'google/gemma-4-12B-it',second_model}
    assert report['status']==report['committed_number']['status']==report['splitter']['status']=='complete'
    for model in report['models'].values():
        assert model['n_labeled']==model['splitter']['n_labeled']==40
        assert model['committed_number']['n_labeled']==40
    labels[0].pop('committed_number')
    report=agreement(rows,labels)
    assert report['committed_number']['status']=='pending'
    assert report['splitter']['status']=='complete'


def test_zero_numeric_eligible_labels_are_complete_with_explicit_denominator():
    from src.phase5_parser_agreement import agreement, audit_sample
    from tests.test_phase5_score import continuation, GEMMA, NEMOTRON
    rows=[row(m,id=str(i),response=continuation(m,'Guide.')) for m in (GEMMA,NEMOTRON) for i in range(40)]
    labels=[dict(id=r['id'],attempt=False,final_reply='Guide.') for r in audit_sample(rows)]
    result=agreement(rows,labels)
    assert result['committed_number']['status']=='complete'
    assert result['committed_number']['n_eligible']==0 and result['committed_number']['agreement'] is None
