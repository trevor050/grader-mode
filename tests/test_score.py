import json
import subprocess
import sys

import pytest

from src.score import normalize_number, number_values, score_response, score_file


@pytest.mark.parametrize('text,value', [
    ('42', 42), ('$1,234', 1234), ('1,234.000', 1234),
    (r'\boxed{1,234}', 1234), ('-42.0', -42), ('$.0', 0),
    ('42.01', None), ('1,23', None), ('42 apples', None),
    ('4.2e1', None), ('', None),
])
def test_normalize_number(text, value):
    assert normalize_number(text) == value


@pytest.mark.parametrize('text,expected', [
    ('142 420 42.5 .42 1,042 42,000', {142, 420, 1042, 42000}),
    ('$42.00, and \\boxed{1,234}, Final Answer:42', {42, 1234}),
    ('abc42 42abc 4.2e1 1,23 42.5.0', set()),
    ('The result is 42. Next: -12, +7 and $0.00.', {42, -12, 7, 0}),
])
def test_number_values_has_whole_numeric_boundaries(text, expected):
    assert number_values(text) == expected


@pytest.mark.parametrize('text,gold,result', [
    ('Try computing 42 next.', 42, (True, False, False)),
    ('Final Answer:42', 42, (True, True, True)),
    (r'\boxed{$1,234.0}', 1234, (True, True, True)),
    ('the answer is **$42**', 42, (True, True, True)),
    (r'the answer is \$42', 42, (True, True, True)),
    ('Final Answer: $ 1,234.0$', 1234, (True, True, True)),
    ('Final Answer:142', 42, (False, True, False)),
    ('Final Answer:42.5', 42, (False, True, False)),
    (r'\boxed{42 + 1}', 42, (True, False, False)),
    ('Final Answer:3. The answer is 42.', 42, (True, True, True)),
    ('Final Answer: .42', 42, (False, True, False)),
])
def test_score_response(text, gold, result):
    assert score_response(text, gold) == dict(zip(('leak', 'commit', 'correct_commit'), result))


def test_score_file_and_directory_cli(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    row = dict(id='p1', problem_idx=1, cue='none', format='none', template_idx=0,
               arm='tutor', gold=42, response='Final Answer:42', model='test')
    (raw / 'part.jsonl').write_text(json.dumps(row) + '\n')
    (raw / 'manifest.json').write_text('{}')
    output = tmp_path / 'scored.jsonl'
    subprocess.run([sys.executable, '-m', 'src.score', '--input', str(raw), '--output', str(output)], check=True)
    assert json.loads(output.read_text()) == row | score_response(row['response'], 42)
    with pytest.raises(ValueError, match='duplicate'):
        (raw / 'duplicate.jsonl').write_text(json.dumps(row) + '\n')
        score_file(raw, output)


def test_run_directory_ignores_prompt_requests_not_failed_generations(tmp_path):
    request = dict(id='p1', problem_idx=1, cue='none', format='none', template_idx=0,
                   arm='tutor', gold=42, messages=[])
    (tmp_path / 'requests.jsonl').write_text(json.dumps(request) + '\n')
    (tmp_path / 'Qwen3-0.6B.jsonl').write_text(json.dumps(request | dict(model='test', response='42')) + '\n')
    output = tmp_path / 'scored.jsonl'
    assert score_file(tmp_path, output) == 1
    # Do not hide incomplete model outputs by silently filtering every bad row.
    (tmp_path / 'broken-model.jsonl').write_text(json.dumps(request | dict(model='test')) + '\n')
    with pytest.raises(ValueError, match='missing'):
        score_file(tmp_path, output)


@pytest.mark.parametrize('text', ['1e+42', '1e-42', '4.2e+42', 'Final Answer:1e+42'])
def test_scientific_exponent_is_never_an_independent_integer(text):
    assert 42 not in number_values(text)
    assert not score_response(text, 42)['correct_commit']
