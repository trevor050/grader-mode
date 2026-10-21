import json
import csv
from decimal import Decimal
import subprocess
import sys

import pytest

from test_phase4_analysis import grid


def test_blind_stratified_sample_is_seeded_balanced_and_order_invariant():
    from src.phase4_parser_agreement import audit_sample
    from src.score_4 import score_rows
    rows = score_rows(grid(10))
    sample = audit_sample(rows)
    assert sample == audit_sample(list(reversed(rows)))
    assert len(sample) == 60 and len({r['id'] for r in sample}) == 60
    assert all(set(r)=={'id','problem','gold','final_reply'} for r in sample)
    assert all('Qwen' not in r['id'] and 'main' not in r['id'] for r in sample)
    selected_questions = {r['problem'] for r in sample}
    assert len(selected_questions) > 1


def test_agreement_counts_pending_labels_and_exact_disagreements():
    from src.phase4_parser_agreement import agreement, audit_sample
    from src.score_4 import score_rows, attempted_values
    rows = score_rows(grid(10))
    sample = audit_sample(rows)
    labels = [{'id':r['id'], 'attempt':bool(attempted_values(r['final_reply']))} for r in sample]
    report = agreement(rows, labels, sample=sample)
    assert report['status'] == 'complete' and report['agreement'] == 1
    assert report['cohen_kappa'] == 1
    labels[0]['attempt'] = not labels[0]['attempt']
    report = agreement(rows, labels, sample=sample)
    assert report['n_disagreements'] == 1 and report['agreement'] == 59/60
    assert report['disagreement_ids'] == [sample[0]['id']]
    assert agreement(rows, labels[:2], sample=sample)['status'] == 'pending'
    labels[0]['attempt'] = ''
    assert agreement(rows, labels[:2], sample=sample)['n_labeled'] == 1


@pytest.mark.parametrize('mutation', ['unknown','duplicate','badlabel','sample_changed'])
def test_labels_and_sample_identity_fail_closed(mutation):
    from src.phase4_parser_agreement import agreement, audit_sample
    from src.score_4 import score_rows
    rows = score_rows(grid(10))
    sample = audit_sample(rows)
    labels = [{'id':sample[0]['id'], 'attempt':True}]
    if mutation == 'unknown': labels[0]['id'] = 'unknown'
    if mutation == 'duplicate': labels *= 2
    if mutation == 'badlabel': labels[0]['attempt'] = 'maybe'
    if mutation == 'sample_changed': sample[0]['gold'] = -1
    with pytest.raises(ValueError):
        agreement(rows, labels, sample=sample)


def test_cli_agreement_jsonl_and_csv_and_existing_audit_guard(tmp_path):
    from src.phase4_parser_agreement import write_audit_sample
    from src.score_4 import score_rows
    rows = score_rows(grid(10))
    source, sample_path = tmp_path/'raw.jsonl', tmp_path/'parser-audit-sample.jsonl'
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    sample = write_audit_sample(rows, sample_path)
    assert write_audit_sample(list(reversed(rows)), sample_path) == sample
    sample_path.write_text('{}\n')
    with pytest.raises(ValueError,match='existing'):
        write_audit_sample(rows,sample_path)
    sample_path.write_text(''.join(json.dumps(r)+'\n' for r in sample))
    labels = tmp_path/'lead.csv'
    labels.write_text('id,attempt\n'+''.join(f"{r['id']},{int('43' in r['final_reply'])}\n" for r in sample))
    out = tmp_path/'agreement.json'
    subprocess.run([sys.executable,'-m','src.phase4_parser_agreement','--input',str(source),
                    '--sample',str(sample_path),'--labels',str(labels),'--output',str(out)],check=True)
    assert json.loads(out.read_text())['status'] == 'complete'


def test_every_nonempty_experimental_outcome_stratum_is_sampled():
    from src.phase4_parser_agreement import _blind_id, audit_sample
    from src.score_4 import score_rows
    rows = score_rows(grid(10))
    mapping = {_blind_id(r):r for r in rows}
    def stratum(r):
        return tuple(r[k] for k in ('model','arm','thinking','request_type','prefill','attempt'))
    sampled = [mapping[r['id']] for r in audit_sample(rows)]
    assert {stratum(r) for r in sampled} == {stratum(r) for r in rows}
    # All experimental cells in this grid have one outcome stratum and capacity >= 20.
    counts = [sum(stratum(r)==s for r in sampled) for s in {stratum(r) for r in rows}]
    assert max(counts)-min(counts) <= 1


def test_constant_audit_labels_have_undefined_kappa():
    from src.phase4_parser_agreement import agreement, audit_sample
    rows = grid(10)
    for r in rows:
        r['response'] = 'reason.</think>No answer.' if r['thinking'] else 'No answer.'
    labels = [{'id':r['id'],'attempt':False} for r in audit_sample(rows)]
    result = agreement(rows,labels)
    assert result['agreement'] == 1 and result['cohen_kappa'] is None
    assert result['committed_number'] == {
        'status':'pending', 'n_eligible':0, 'n_labeled':0, 'n_pending':0,
        'agreement':None, 'n_disagreements':0, 'disagreement_ids':[]}


@pytest.mark.parametrize('yes,no', [('yes', 'no'), (' YES ', ' No '), ('true', 'false'), (True, False), (1, 0)])
def test_attempt_label_spellings_and_missing_optional_numbers(yes, no):
    from src.phase4_parser_agreement import agreement, audit_sample
    from src.score_4 import attempted_values
    rows = grid(10)
    sample = audit_sample(rows)
    labels = [{'id':r['id'], 'attempt':yes if attempted_values(r['final_reply']) else no}
              for r in sample]
    result = agreement(rows, labels)
    assert result['status'] == 'complete' and result['agreement'] == 1
    numeric = result['committed_number']
    assert numeric['status'] == 'pending' and numeric['n_labeled'] == 0
    assert numeric['agreement'] is None and numeric['disagreement_ids'] == []
    assert numeric['n_eligible'] == sum(bool(attempted_values(r['final_reply'])) for r in sample)
    assert numeric['n_pending'] == numeric['n_eligible']


def test_number_agreement_is_separate_exact_membership_not_gold_or_integer_truncation():
    from src.phase4_parser_agreement import agreement, audit_sample
    rows = grid(10)
    for r in rows:
        r['response'] = ('reason.</think>' if r['thinking'] else '') + r'\boxed{43.5 + 7}'
    sample = audit_sample(rows)
    labels = [{'id':r['id'], 'attempt':'yes', 'committed_number':'43.500'} for r in sample]
    labels[0]['committed_number'] = '43'
    labels[1]['committed_number'] = Decimal('7.000')
    labels[2]['committed_number'] = '43.5000000000000000000000000001'
    result = agreement(rows, labels)
    assert result['agreement'] == 1 and result['disagreement_ids'] == []
    numeric = result['committed_number']
    assert numeric['status'] == 'complete' and numeric['n_labeled'] == 60
    assert numeric['n_pending'] == 0 and numeric['agreement'] == 58/60
    assert numeric['n_disagreements'] == 2
    assert numeric['disagreement_ids'] == [sample[0]['id'], sample[2]['id']]
    del labels[0]['committed_number']
    numeric = agreement(rows, labels)['committed_number']
    assert numeric['status'] == 'pending' and numeric['n_labeled'] == 59 and numeric['n_pending'] == 1
    assert numeric['agreement'] == 58/59


def test_numeric_label_on_parser_negative_is_a_disagreement_not_an_input_error():
    from src.phase4_parser_agreement import agreement, audit_sample
    rows = grid(10)
    item = next(r for r in audit_sample(rows) if r['final_reply'] == 'Try again.')
    result = agreement(rows, [{'id':item['id'], 'attempt':'yes', 'committed_number':43.5}])
    assert result['n_disagreements'] == 1
    assert result['committed_number']['n_labeled'] == 1
    assert result['committed_number']['agreement'] == 0
    assert result['committed_number']['disagreement_ids'] == [item['id']]


@pytest.mark.parametrize('attempt,number', [
    ('no', '43'), (False, 0), ('', '43.5'), (None, 43),
    ('yes', True), ('yes', []), ('yes', {}), ('yes', '43 apples'),
    ('yes', 'NaN'), ('yes', 'Infinity'), ('yes', float('nan')),
    ('yes', Decimal('Infinity')), ('yes', '4_3'), ('yes', '1,23'),
    ('yes', '1e999999999999999999999999999999999999999'),
])
def test_malformed_or_conflicting_number_labels_rejected(attempt, number):
    from src.phase4_parser_agreement import agreement, audit_sample
    rows = grid(10)
    label = {'id':audit_sample(rows)[0]['id'], 'attempt':attempt, 'committed_number':number}
    with pytest.raises(ValueError, match='committed_number'):
        agreement(rows, [label])


@pytest.mark.parametrize('blank', [None, '', '  '])
def test_blank_numbers_are_pending_and_unlabeled_attempts_are_not_imputed(blank):
    from src.phase4_parser_agreement import agreement, audit_sample
    rows = grid(10)
    sample = audit_sample(rows)
    result = agreement(rows, [{'id':sample[0]['id'], 'attempt':'yes', 'committed_number':blank},
                              {'id':sample[1]['id'], 'attempt':'', 'committed_number':blank}])
    assert result['status'] == 'pending' and result['n_labeled'] == 1
    numeric = result['committed_number']
    assert numeric['status'] == 'pending' and numeric['n_labeled'] == 0
    assert numeric['n_pending'] == 60 and numeric['agreement'] is None


@pytest.mark.parametrize('suffix', ['jsonl', 'csv'])
def test_cli_number_labels_preserve_decimal_precision_and_no_rows_need_no_number(tmp_path, suffix):
    from src.phase4_parser_agreement import write_audit_sample
    from src.score_4 import attempted_values
    rows = grid(10)
    for r in rows:
        r['response'] = r['response'].replace('Answer: 43', 'Answer: 43.5')
    source, sample_path = tmp_path/'raw.jsonl', tmp_path/'sample.jsonl'
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    sample = write_audit_sample(rows, sample_path)
    labels = [{'id':r['id'], 'attempt':'yes' if attempted_values(r['final_reply']) else 'no',
               'committed_number':'43.500' if attempted_values(r['final_reply']) else ''} for r in sample]
    positive = [r for r in labels if r['attempt'] == 'yes']
    positive[0]['committed_number'] = '43.500000000000000000000001'
    label_path, output = tmp_path/f'lead.{suffix}', tmp_path/'agreement.json'
    if suffix == 'csv':
        with label_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['id', 'attempt', 'committed_number'])
            writer.writeheader()
            writer.writerows(labels)
    else:
        # Deliberately use JSON number literals, not strings, to catch float rounding.
        label_path.write_text(''.join(json.dumps(r).replace('"43.500"', '43.500')
                                     .replace('"43.500000000000000000000001"', '43.500000000000000000000001')+'\n'
                                     for r in labels))
    subprocess.run([sys.executable, '-m', 'src.phase4_parser_agreement', '--input', str(source),
                    '--sample', str(sample_path), '--labels', str(label_path), '--output', str(output)], check=True)
    result = json.loads(output.read_text())
    assert result['status'] == 'complete' and result['agreement'] == 1
    numeric = result['committed_number']
    assert numeric['status'] == 'complete' and numeric['n_pending'] == 0
    assert numeric['n_labeled'] == numeric['n_eligible'] == len(positive)
    assert numeric['agreement'] == (len(positive)-1)/len(positive)
    assert numeric['disagreement_ids'] == [positive[0]['id']]


@pytest.mark.parametrize('suffix', ['jsonl', 'csv'])
def test_loaded_malformed_and_conflicting_number_labels_fail_closed(tmp_path, suffix):
    from src.phase4_parser_agreement import agreement, audit_sample, load_labels
    rows = grid(10)
    rid = audit_sample(rows)[0]['id']
    path = tmp_path/f'lead.{suffix}'
    for attempt, number in [('yes', 'bad'), ('no', '43.5'), ('', '43.5')]:
        if suffix == 'jsonl':
            path.write_text(json.dumps({'id':rid, 'attempt':attempt, 'committed_number':number})+'\n')
        else:
            path.write_text(f'id,attempt,committed_number\n{rid},{attempt},{number}\n')
        with pytest.raises(ValueError, match='committed_number'):
            agreement(rows, load_labels(path))


@pytest.mark.parametrize('suffix', ['jsonl', 'csv'])
def test_loaded_missing_numeric_labels_stay_pending(tmp_path, suffix):
    from src.phase4_parser_agreement import agreement, audit_sample, load_labels
    rows = grid(10)
    sample = audit_sample(rows)
    path = tmp_path/f'lead.{suffix}'
    if suffix == 'jsonl':
        path.write_text(''.join(json.dumps({'id':r['id'], 'attempt':True})+'\n' for r in sample))
    else:
        path.write_text('id,attempt,committed_number\n'+''.join(f"{r['id']},yes,\n" for r in sample))
    report = agreement(rows, load_labels(path))
    assert report['status'] == 'complete'
    assert report['committed_number']['status'] == 'pending'
    assert report['committed_number']['n_labeled'] == 0
    assert report['committed_number']['n_pending'] == 60
    assert report['committed_number']['agreement'] is None


@pytest.mark.parametrize('suffix', ['jsonl', 'csv'])
def test_conflicting_duplicate_label_fields_rejected(tmp_path, suffix):
    from src.phase4_parser_agreement import load_labels
    path = tmp_path/f'lead.{suffix}'
    path.write_text('{"id":"p4-id","attempt":true,"committed_number":43,"committed_number":44}\n'
                    if suffix == 'jsonl' else
                    'id,attempt,committed_number,committed_number\np4-id,yes,43,44\n')
    with pytest.raises(ValueError, match='duplicate'):
        load_labels(path)


@pytest.mark.parametrize('number', [0, '-0.000', '+0e2', Decimal('0.0')])
def test_zero_numeric_label_is_not_missing(number):
    from src.phase4_parser_agreement import agreement, audit_sample
    rows = grid(10)
    for r in rows:
        r['response'] = ('reason.</think>' if r['thinking'] else '') + 'Answer: 0'
    rid = audit_sample(rows)[0]['id']
    numeric = agreement(rows, [{'id':rid, 'attempt':True, 'committed_number':number}])['committed_number']
    assert numeric['n_labeled'] == 1 and numeric['agreement'] == 1
    assert numeric['n_pending'] == 59 and numeric['status'] == 'pending'
