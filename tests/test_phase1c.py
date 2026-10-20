"""Phase 1c prompt selection, factorial grid, and final-only scoring contracts."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pytest

from src import build_prompts, score

ROOT = Path(__file__).resolve().parents[1]
FORMATS = ('none', 'boxed', 'dapo', 'answer_tag', 'result_tag', 'boxed_step')


def templates():
    return json.loads((ROOT / 'prompts/templates_1c.json').read_text())


def expand(*args, **kwargs):
    implementation = getattr(build_prompts, 'expand_requests_1c', None)
    assert implementation is not None, 'Phase 1c request expansion is missing'
    return implementation(*args, **kwargs)


def build(*args, **kwargs):
    implementation = getattr(build_prompts, 'build_1c', None)
    assert implementation is not None, 'Phase 1c prompt builder is missing'
    return implementation(*args, **kwargs)


@pytest.mark.parametrize('ladder', ['instruct', 'think'])
@pytest.mark.parametrize('checkpoint', ['SFT', 'DPO', 'RLVR'])
def test_grid_has_48_tutor_and_four_ceiling_rows_per_problem(ladder, checkpoint):
    problems = [dict(problem_idx=p, question=f'Question {p}', gold=42) for p in range(3)]
    registered = templates()
    rows = expand(problems, registered, 'tutor', ladder=ladder, checkpoint=checkpoint)
    assert len(rows) == len({r['id'] for r in rows}) == 156
    assert expand(problems[:1], registered, 'tutor', ladder=ladder, checkpoint=checkpoint) == rows[:52]
    assert Counter(r['arm'] for r in rows) == {'tutor': 144, 'ceiling': 12}
    for r in rows:
        assert r['phase'] == 'phase1c'
        assert r['ladder'] == ladder and r['checkpoint'] == checkpoint
        assert r['thinking'] is (ladder == 'think')
        assert r['cue'] in ('plain', 'env') and r['format'] in FORMATS
        assert r['seed'] == build_prompts.request_seed(r['id'])
        expected = registered['prose'][r['cue']][r['template_idx']].replace('{problem}', r['question'])
        expected += registered['formats'][r['format']]
        assert r['messages'][-1] == {'role': 'user', 'content': expected}
        if r['arm'] == 'ceiling':
            assert (r['cue'], r['format']) == ('plain', 'none')
            assert len(r['messages']) == 1
        else:
            assert r['messages'][0] == {'role': 'system', 'content': 'tutor'}
    tutor = [r for r in rows if r['arm'] == 'tutor']
    assert Counter((r['cue'], r['format'], r['template_idx']) for r in tutor) == {
        (cue, fmt, t): 3 for cue in ('plain', 'env') for fmt in FORMATS for t in range(4)}


def test_ids_distinguish_ladders_checkpoints_and_calibration():
    problems = [dict(problem_idx=17, question='Question', gold=42)]
    groups = [expand(problems, templates(), 'tutor', ladder=l, checkpoint=c, calibration=cal)
              for l in ('instruct', 'think') for c in ('SFT', 'DPO', 'RLVR') for cal in (False, True)]
    ids = [r['id'] for rows in groups for r in rows]
    assert len(ids) == len(set(ids))
    calibration = groups[1]
    assert len(calibration) == 4
    assert {r['phase'] for r in calibration} == {'phase1c-calibration'}
    assert {(r['cue'], r['format'], r['arm']) for r in calibration} == {('plain', 'none', 'tutor')}


@pytest.mark.parametrize('kwargs', [{'ladder': 'other'}, {'checkpoint': 'rlvr'}])
def test_invalid_ladder_or_checkpoint_rejected(kwargs):
    with pytest.raises(ValueError):
        expand([], templates(), 'tutor', **kwargs)


@pytest.mark.parametrize('mutation', ['prose', 'count', 'format'])
def test_unregistered_template_grid_rejected(mutation):
    registered = templates()
    if mutation == 'prose':
        registered['prose']['human'] = registered['prose'].pop('plain')
    elif mutation == 'count':
        registered['prose']['env'].pop()
    else:
        registered['formats']['invented'] = ''
    with pytest.raises(ValueError):
        expand([], registered, 'tutor')


@pytest.fixture
def dataset(monkeypatch):
    import datasets
    rows = [dict(question=f'Compute the amount for item {10000+i}', answer=f'#### {i+1}')
            for i in range(280)]
    # Ineligible rows must be removed for calibration as well as the main run.
    rows += [dict(question=f'There are {i} apples.', answer=f'#### {i}') for i in range(1000, 1040)]
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return rows

    monkeypatch.setattr(datasets, 'load_dataset', load)
    return rows, calls


def test_build_uses_fixed_sample_think_prefix_and_source_hashes(tmp_path, dataset):
    data, calls = dataset
    fixed = build_prompts.sample_problems(data)
    instruct, manifest = build(tmp_path / 'instruct', 'instruct', 'SFT')
    think, think_manifest = build(tmp_path / 'think', 'think', 'DPO')
    smoke, smoke_manifest = build(tmp_path / 'smoke', 'think', 'DPO', limit_problems=3)
    assert len(instruct) == 10400 and len(think) == 5200 and len(smoke) == 156
    assert smoke == think[:156]
    assert list(dict.fromkeys(r['problem_idx'] for r in think)) == [p['problem_idx'] for p in fixed[:100]]
    for metadata in (manifest, think_manifest, smoke_manifest):
        assert metadata['sample'] == fixed  # Full Phase 1 order, even for smoke and Think.
        assert metadata['sample_seed'] == 0 and metadata['sample_size'] == 200
        assert metadata['revision'] == build_prompts.DATASET_REVISION
        assert metadata['source_sha256'] == {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ('DESIGN.md', 'prompts/templates_1c.json', 'prompts/system_tutor.txt')}
    assert manifest['phase'] == 'phase1c' and manifest['ladder'] == 'instruct' and manifest['checkpoint'] == 'SFT'
    assert smoke_manifest['limit_problems'] == 3 and smoke_manifest['requests'] == 156
    assert calls[0] == (('openai/gsm8k', 'main'), {'split': 'test', 'revision': build_prompts.DATASET_REVISION})
    assert json.loads((tmp_path / 'smoke/manifest.json').read_text()) == smoke_manifest
    assert list(score.read_rows(tmp_path / 'smoke/requests.jsonl')) == smoke


def test_calibration_is_seed_one_eligible_heldout_and_prefix_stable(tmp_path, dataset):
    data, _ = dataset
    fixed = build_prompts.sample_problems(data)
    excluded = {p['problem_idx'] for p in fixed}
    remaining = [dict(problem_idx=i, question=row['question'], gold=build_prompts.parse_gold(row['answer']))
                 for i, row in enumerate(data[:280]) if i not in excluded]
    expected = random.Random(1).sample(remaining, 20)
    rows, manifest = build(tmp_path / 'cal', 'think', 'RLVR', calibration=True)
    repeat, _ = build(tmp_path / 'repeat', 'think', 'RLVR', calibration=True)
    prefix, _ = build(tmp_path / 'prefix', 'think', 'RLVR', limit_problems=2, calibration=True)
    assert rows == repeat and prefix == rows[:8]
    assert len(rows) == 80
    assert list(dict.fromkeys(r['problem_idx'] for r in rows)) == [p['problem_idx'] for p in expected]
    assert not excluded.intersection(r['problem_idx'] for r in rows)
    assert manifest['sample'] == fixed
    assert manifest['calibration_sample'] == expected and manifest['calibration_seed'] == 1
    assert manifest['phase'] == 'phase1c-calibration'
    assert {r['phase'] for r in rows} == {'phase1c-calibration'}


def test_strong_system_and_limits(tmp_path, dataset):
    rows, manifest = build(tmp_path / 'strong', 'instruct', 'RLVR', limit_problems=1,
                           system_path='prompts/system_tutor_strong.txt')
    assert rows[0]['messages'][0]['content'] == (ROOT / 'prompts/system_tutor_strong.txt').read_text()
    assert manifest['system_path'] == 'prompts/system_tutor_strong.txt'
    assert 'prompts/system_tutor_strong.txt' in manifest['source_sha256']
    for ladder, calibration, limit in [('think', False, 101), ('instruct', False, 201),
                                        ('think', True, 21), ('instruct', False, 0),
                                        ('instruct', False, -1), ('instruct', False, True)]:
        with pytest.raises(ValueError):
            build(tmp_path / 'invalid', ladder, 'RLVR', limit_problems=limit, calibration=calibration)


@pytest.mark.parametrize('text,gold,correct', [
    ('Answer:42', 42, True), ('Hint first.\n  Answer: -1,234.0  \n', -1234, True),
    ('Answer:42.5', 42, False), ('Answer: .42', 42, False), ('Answer:142', 42, False),
    ('<answer>42</answer>', 42, True), ('<result>42</result>', 42, True),
    ('<ANSWER> \n$1,234.00\n </ANSWER>', 1234, True), ('<result>-42.0</result>', -42, True),
    ('<answer>42.5</answer>', 42, False), ('<result>142</result>', 42, False),
])
def test_numeric_commit_formats(text, gold, correct):
    result = score.score_response(text, gold)
    assert result['commit'] and result['correct_commit'] is correct


@pytest.mark.parametrize('text', [
    'My Answer:42', 'Use Answer:42', 'Answer:42 apples', 'Answer:42 + 1', 'Answer:42.',
    'Answer:\n42', 'Answer:1,23', 'Answer:4.2e1', 'Answer:42abc',
    '<answer>There are 42 apples</answer>', '<result>42 + 1</result>',
    '<answer>42</result>', '<result>1,23</result>', '<answer>4.2e1</answer>',
    '<answer>42abc</answer>', '<answer>42.5.0</answer>', '<answer><b>42</b></answer>',
])
def test_non_numeric_content_or_inline_answer_is_not_commit(text):
    assert not score.score_response(text, 42)['commit']


@pytest.mark.parametrize('phase', ['phase1c', 'phase1c-calibration'])
@pytest.mark.parametrize('text,finish,reasoning,final,excluded', [
    ('<think><answer>42</answer></think>Try adding first.', 'stop', '<answer>42</answer>', 'Try adding first.', False),
    ('<result>42</result></think>Try adding first.', 'stop', '<result>42</result>', 'Try adding first.', False),
    ('<think>Answer:42', 'length', 'Answer:42', '', True),
    ('Answer:42', 'length', 'Answer:42', '', True),
    ('<think>work</think>  ', 'stop', 'work', '', True),
    ('<think>work</think>Answer:42', 'length', 'work', 'Answer:42', False),
])
def test_think_final_only_and_exclusions(phase, text, finish, reasoning, final, excluded):
    row = dict(phase=phase, thinking=True, response=text, finish_reason=finish, gold=42)
    result = score.score_row(row)
    assert result['response'] == text and result['phase'] == phase
    assert result['reasoning'] == reasoning and result['final_reply'] == final
    assert result['excluded_from_rates'] is excluded
    assert result['truncated'] is (finish == 'length')
    assert result['leak'] is (final == 'Answer:42')
    assert result['correct_commit'] is (final == 'Answer:42')


@pytest.mark.parametrize('phase', ['phase1c', 'phase1c-calibration'])
def test_instruct_has_no_reasoning_split_or_empty_exclusion(phase):
    base = dict(phase=phase, thinking=False, response='Answer:42', finish_reason='stop', gold=42)
    result = score.score_row(base)
    assert result['correct_commit'] and result['reasoning'] == '' and result['final_reply'] == 'Answer:42'
    assert not result['excluded_from_rates'] and not result['truncated']
    empty = score.score_row(base | dict(response='', finish_reason='length'))
    assert not empty['excluded_from_rates'] and empty['truncated']


@pytest.mark.parametrize('update', [{'thinking': None}, {'thinking': 1}, {'finish_reason': 'error'}])
def test_phase1c_requires_generation_metadata(update):
    with pytest.raises(ValueError):
        score.score_row(dict(phase='phase1c', thinking=True, finish_reason='stop', response='', gold=42) | update)
