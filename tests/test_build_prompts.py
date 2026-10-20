import json
from pathlib import Path
import pytest
from src.build_prompts import expand_requests, sample_problems, parse_gold, request_seed

ROOT = Path(__file__).resolve().parents[1]

def test_gold_is_integer_and_not_solution_numbers():
    assert parse_gold('work 99\n#### $1,234.0') == 1234
    with pytest.raises(ValueError):
        parse_gold('work 42 without delimiter')
    with pytest.raises(ValueError):
        parse_gold('#### 1.25')

def test_question_filter_uses_whole_normalized_number_tokens():
    rows = [
        {'question': 'There are 42 apples', 'answer': '#### 42'},
        {'question': 'There are 142 apples', 'answer': '#### 42'},
        {'question': 'It costs $1,234.0', 'answer': '#### 1234'},
        {'question': 'There are 42.5 apples', 'answer': '#### 42'},
    ]
    chosen = sample_problems(rows, count=2)
    assert {p['problem_idx'] for p in chosen} == {1, 3}

def test_sample_is_deterministic_and_grid_is_matched():
    data = [{'question': f'Compute the missing amount for item {i+1000}', 'answer': f'#### {i+1}'} for i in range(220)]
    sample = sample_problems(data)
    assert sample == sample_problems(data)
    assert len({p['problem_idx'] for p in sample}) == 200
    templates = json.loads((ROOT / 'prompts/templates.json').read_text())
    system = (ROOT / 'prompts/system_tutor.txt').read_text()
    full = expand_requests(sample, templates, system)
    smoke = expand_requests(sample[:10], templates, system)
    assert len(full) == 8000
    assert len(smoke) == 400
    assert smoke == full[:400]
    assert len({r['id'] for r in full}) == len(full)
    for r in full:
        assert r['seed'] == request_seed(r['id'])
        assert 0 <= r['seed'] < 2**32
        assert r['messages'][-1]['role'] == 'user'
        if r['arm'] == 'ceiling':
            assert len(r['messages']) == 1
            assert r['cue'] == r['format'] == 'none'
        else:
            assert r['messages'][0] == {'role': 'system', 'content': system}
