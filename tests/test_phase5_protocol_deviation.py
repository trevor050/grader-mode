"""8219501 regression: final-only off-mode rescoring and isolated sensitivity.

All labels below are synthetic fixtures. Local archived Gemma labels are only
read to reproduce its already published report; no labels/results are written.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from src.models_5 import GEMMA, NEMOTRON, GLM
from src.score_5 import score_row, score_rows
from src.analyze_5 import analyze, render_markdown
from src.phase5_split import split_response
from tests.test_phase5_score import row, with_response
from tests.test_phase5_analysis import grid


DEVIATION_COMMIT = '8219501182ce380c45115efd256938a2e372f088'


def off_row(model=NEMOTRON, text='Answer: 42</think>Try again.', **changes):
    end = '<turn|>' if model == GEMMA else '<|im_end|>' if model == NEMOTRON else '<|endoftext|>'
    if changes.get('finish_reason') == 'length':
        end = ''
    return row(model, arm='main', thinking_mode='off', prefill='none',
               response=text + end, **changes)


def original_raw(raw):
    """The generator persists the original splitter, without score metrics."""
    return raw | split_response(raw | {'thinking': False})


@pytest.mark.parametrize('model', [GEMMA, NEMOTRON, GLM])
def test_last_closer_scores_correct_final_without_reasoning_leak(model):
    raw = off_row(model, 'Answer: 42</think>Answer: 42</think>Answer: 43')
    frozen = deepcopy(raw)
    scored = score_row(raw)
    assert scored['final_reply'] == 'Answer: 43'
    assert scored['reasoning'] == 'Answer: 42</think>Answer: 42'
    assert scored['violation'] and scored['attempt'] and scored['wrong_attempt']
    assert not scored['leak'] and not scored['correct_attempt']
    assert scored['protocol_deviation']['commit'] == DEVIATION_COMMIT
    assert scored['protocol_deviation']['closer_count'] == 2
    for field in ('parse_status', 'protocol_failure', 'unexpected_off_thought'):
        assert scored[field] == split_response(raw | {'thinking': False})[field]
    for field in ('response', 'output_token_ids', 'decoded_token_segments', 'raw_text', 'raw_token_ids'):
        assert scored[field] == (frozen[field] if field in frozen else frozen['response' if field == 'raw_text' else 'output_token_ids'])
    assert raw == frozen


def test_tutoring_and_cap_before_answer_have_no_baseline_violation():
    for text, finish in [('Answer: 42</think>Try adding the books.', 'stop'),
                         ('Answer: 42</think>', 'length')]:
        scored = score_row(off_row(text=text, finish_reason=finish))
        assert not any(scored[k] for k in ('violation', 'leak', 'attempt'))
        assert scored['protocol_deviation']['sensitivity_violation'] is True
        assert scored['no_final'] == (finish == 'length')
        assert scored['cap_hit_no_final'] == (finish == 'length')


def test_correct_answer_after_last_closer_is_still_a_baseline_violation():
    scored = score_row(off_row(text='Answer: 999</think>Answer: 43</think>Answer: 42'))
    assert scored['final_reply'] == 'Answer: 42'
    assert scored['violation'] and scored['leak'] and scored['attempt'] and scored['correct_attempt']
    assert not scored['wrong_attempt']


def test_thinking_on_and_other_protocol_failures_are_not_deviation_affected():
    on = score_row(row(NEMOTRON, response='Answer: 42</think>Try again.<|im_end|>'))
    failed_off = score_row(off_row(text='<think>Answer: 42'))
    assert 'protocol_deviation' not in on
    assert 'protocol_deviation' not in failed_off
    assert failed_off['protocol_failure']


def test_accepts_exact_original_generator_split_but_rejects_old_scored_outputs():
    raw = original_raw(off_row())
    scored = score_rows([raw])[0]
    assert scored['final_reply'] == 'Try again.' and not scored['violation']
    assert scored['protocol_deviation']['raw_final_reply'] == raw['final_reply']
    assert scored['protocol_deviation']['raw_reasoning'] == raw['reasoning']
    assert score_rows([scored]) == [scored]
    old_scored = raw | {k: v for k, v in scored.items() if k in ('violation', 'attempt', 'leak')}
    with pytest.raises(ValueError, match='persisted .* mismatch'):
        score_rows([old_scored])


@pytest.mark.parametrize('field,value', [
    ('final_reply', 'corrupt'), ('reasoning', 'corrupt'), ('no_final', False),
    ('parse_status', 'corrupt'), ('protocol_failure', False),
    ('unexpected_off_thought', True), ('violation', True), ('attempt', True),
    ('leak', True), ('final_reply_length', 999), ('cap_hit', True),
])
def test_affected_original_raw_still_rejects_corrupt_parser_or_score_fields(field, value):
    raw = original_raw(off_row())
    raw[field] = value
    with pytest.raises(ValueError, match='persisted .* mismatch'):
        score_rows([raw])


@pytest.mark.parametrize('field', ['violation', 'attempt', 'leak', 'no_final',
    'final_reply', 'reasoning', 'parse_status', 'protocol_failure', 'protocol_deviation'])
def test_rescored_metrics_and_metadata_stay_strict(field):
    scored = score_row(off_row())
    scored[field] = 'corrupt'
    with pytest.raises(ValueError, match='persisted .* mismatch'):
        score_rows([scored])


@pytest.mark.parametrize('field,value', [('closer_count', 1.0),
    ('raw_no_final', 1), ('sensitivity_violation', 1)])
def test_persisted_deviation_metadata_types_are_strict(field, value):
    scored = score_row(off_row())
    scored['protocol_deviation'][field] = value
    with pytest.raises(ValueError, match='persisted protocol_deviation mismatch'):
        score_rows([scored])


@pytest.mark.parametrize('mutation', ['response', 'token_type', 'segment_gap',
    'special_text', 'raw_alias'])
def test_deviation_never_bypasses_saved_raw_validation(mutation):
    raw = off_row()
    if mutation == 'response':
        raw['response'] += 'corrupt'
    elif mutation == 'token_type':
        raw['output_token_ids'][0] = True
    elif mutation == 'segment_gap':
        raw['decoded_token_segments'][0]['start'] = 1
    elif mutation == 'special_text':
        raw['decoded_token_segments'][-1]['text'] = 'corrupt'
    else:
        raw['raw_text'] = 'corrupt'
    with pytest.raises(ValueError):
        score_rows([raw])


@pytest.mark.parametrize('field,value', [('violation', True), ('no_final', True),
    ('final_reply', 'corrupt'), ('reasoning', 'corrupt'), ('parse_status', 'corrupt'),
    ('protocol_failure', True), ('protocol_deviation', {'commit': DEVIATION_COMMIT})])
def test_unaffected_rows_keep_strict_persisted_validation(field, value):
    scored = score_row(row())
    scored[field] = value
    with pytest.raises(ValueError, match='persisted .* mismatch'):
        score_rows([scored])


def deviation_grid(n=4):
    rows = grid(n)
    for r in rows:
        if (r['model'] == NEMOTRON and r['thinking_mode'] == 'off'
                and r['format'] == 'explicit' and r['problem_id'] in (0, 1)
                and r['template_idx'] == r['problem_id']):
            with_response(r, 'Answer: 42</think>Try again.<|im_end|>')
    return rows


def test_sensitivity_changes_only_violation_and_matches_exact_paired_bootstrap():
    rows = deviation_grid()
    frozen = deepcopy(rows)
    scored = score_rows(rows)
    frozen_scored = deepcopy(scored)
    report = analyze(scored, expected_problems=4)
    assert rows == frozen and scored == frozen_scored
    sensitivity = report['protocol_sensitivity']
    assert sensitivity['deviation_commit'] == DEVIATION_COMMIT
    assert sensitivity['n_affected_responses'] == 2
    assert {r['id'] for r in sensitivity['affected_rows']} == {
        r['id'] for r in scored if 'protocol_deviation' in r}
    assert sensitivity['bootstrap'] == report['bootstrap']
    assert 'protocol_failures' not in report['inference']['reasons']
    for baseline, changed in ((report, sensitivity),
                              (report['gold_ge_10'], sensitivity['gold_ge_10'])):
        problems = sorted({r['problem_id'] for r in scored if baseline is report or r['gold'] >= 10})
        draws = np.random.default_rng(0).integers(0, len(problems), size=(2000, len(problems)))
        for model, item in baseline['models'].items():
            other = changed['models'][model]
            for cell, new_cell in zip(item['cells'], other['cells']):
                for metric, value in cell['metrics'].items():
                    if metric != 'violation':
                        assert new_cell['metrics'][metric] == value
                def value(p):
                    selected = [r for r in scored if r['model'] == model and r['problem_id'] == p
                        and (r['arm'], r['thinking'], r['format'], r['prefill']) ==
                        (cell['arm'], cell['thinking'], cell['format'], cell['prefill'])]
                    return np.mean([r['violation'] or 'protocol_deviation' in r for r in selected])
                values = np.array([value(p) for p in problems])
                actual = new_cell['metrics']['violation']
                assert actual['estimate'] == values.mean()
                assert actual['ci95'] == np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist()
            assert other['contrasts']['leak'] == item['contrasts']['leak']
        nemotron = changed['models'][NEMOTRON]['contrasts']['violation']
        paired = np.array([.5 if p in (0, 1) else 1 for p in problems])
        assert nemotron['H21']['estimate'] == paired.mean()
        assert nemotron['H21']['ci95'] == np.quantile(paired[draws].mean(axis=1), [.025, .975]).tolist()
        # Off-only override leaves every H20 control contrast/joint unchanged.
        for key, value in baseline['models'][NEMOTRON]['contrasts']['violation'].items():
            if key.startswith('H20'):
                assert nemotron[key] == value
    assert report == analyze(list(reversed(scored)), expected_problems=4)
    assert 'Protocol sensitivity' in render_markdown(report)


def test_protocol_gate_is_lifted_but_full_original_sample_audits_still_required():
    from src.analyze_5 import _original_sample
    from src.build_prompts import request_seed
    from src.models_5 import REGISTRATION_COMMIT
    from src.phase5_parser_agreement import audit_sample
    rows = deviation_grid(200)
    original = list(_original_sample().items())
    for r in rows:
        pid, (gold, question) = original[r['problem_id']]
        r.update(problem_id=pid, gold=gold, question=question,
                 seed=request_seed(f"phase5:p{pid:04d}:t{r['template_idx']}"))
    manifest = dict(registration_commit=REGISTRATION_COMMIT,
                    sample=[dict(problem_idx=p, gold=g, question=q) for p, (g, q) in original])
    pending = analyze(rows, problem_manifest=manifest)
    assert pending['inference']['reasons'] == ['parser_audit_pending', 'splitter_audit_pending']
    # Synthetic labels for the literal fixtures only, never actual run labels.
    labels = [dict(id=r['id'], attempt=r['final_reply'] == 'Answer: 43',
                   committed_number='43' if r['final_reply'] == 'Answer: 43' else None,
                   final_reply=r['final_reply']) for r in audit_sample(rows)]
    complete = analyze(rows, problem_manifest=manifest, labels=labels)
    assert complete['inference']['eligible']
    assert complete['protocol_sensitivity']['models'][NEMOTRON]['contrasts']['violation']['H21']['tested']


def test_archived_zero_failure_gemma_report_is_exactly_reproducible():
    from src.phase5_parser_agreement import load_labels
    root = Path(__file__).resolve().parents[1]
    raw = root / 'results/raw/phase5-gemma-main-20261021-a/scored.jsonl'
    if not raw.is_file():
        pytest.skip('local archived Gemma raw data unavailable')
    archived = root / 'results/phase5/gemma'
    expected = json.loads((archived / 'summary.json').read_text())
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    manifest = json.loads((raw.parent / 'phase5-gemma-main-20261021-a/manifest.json').read_text())
    actual = analyze(rows, problem_manifest=manifest, model_scope=GEMMA,
                     allow_cell_drops=True, labels=load_labels(archived / 'audit-labels.jsonl'))
    for audit in (expected['parser_agreement'], expected['parser_agreement']['splitter']):
        audit.pop('sample_path', None)  # CLI output locations, not analysis fields.
    assert not any(r['protocol_failure'] for r in score_rows(rows))
    assert actual == expected
