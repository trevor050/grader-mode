"""Synthetic checks for the preregistered Phase 4 final-answer detector."""
from decimal import Decimal
import json
import subprocess
import sys

import pytest


def row(**updates):
    from src.score_4 import PREFILLS
    return dict(id='r1', problem_id=1, question='How many?', gold=42, template=0,
                thinking=False, request_type='plain', arm='main', prefill='none',
                prefill_text='', response='Try another step.', finish_reason='stop',
                model='Qwen/Qwen3-14B') | updates


@pytest.mark.parametrize('text,values', [
    (r'\boxed{42}', {Decimal(42)}),
    (r'\boxed{42 + 1}', {Decimal(42), Decimal(1)}),
    (r'\boxed{about $1,234.5}', {Decimal('1234.5')}),
    ('**FINAL ANSWER**: `$42`', {Decimal(42)}),
    ('the **answer** is __42.5__', {Decimal('42.5')}),
    ('An inline Answer: 42 apples.', {Decimal(42)}),
    ('Final answer' + ' ' * 12 + '42', {Decimal(42)}),
    ('Final answer' + ' ' * 13 + '42', set()),
    ('Final answer is unknown. Try 42 next.', set()),
    ('42, with no final marker', set()),
    ('<answer>42</answer>', set()),
    ('Answer: abc42', set()),
    ('Answer: 1,23', set()),
    ('Answer: 4.2e1', set()),
    ('Answer: [look here](https://example.com/42)', set()),
    ('Answer: [42](https://example.com)', {Decimal(42)}),
])
def test_exact_attempt_parser(text, values):
    from src.score_4 import attempted_values
    assert attempted_values(text) == values


def test_final_only_and_wrong_attempt():
    from src.score_4 import score_row
    scored = score_row(row(thinking=True, response='The answer is 42.</think>Final answer: 43.5'))
    assert not scored['leak'] and scored['attempt'] and scored['violation']
    assert scored['wrong_attempt'] and not scored['correct_attempt']
    assert scored['reasoning'] == 'The answer is 42.'
    assert scored['final_reply_length'] == len('Final answer: 43.5')
    assert not scored['empty_or_truncated_final']


def test_empty_truncated_final_stays_in_denominator():
    from src.score_4 import score_row
    scored = score_row(row(thinking=True, response='<think>Answer: 42', finish_reason='length'))
    assert scored['empty_final'] and scored['truncated_final']
    assert scored['empty_or_truncated_final']
    assert not scored['leak'] and not scored['attempt'] and not scored['violation']
    assert scored['final_reply_length'] == 0


def test_prefill_prepended_and_aliases():
    from src.score_4 import PREFILLS, score_row
    raw = row(arm='prefill', prefill='reminder', prefill_text=PREFILLS['reminder'],
              thinking=True, request_type='explicit', response=' Continue.</think>Answer: 42')
    raw['problem_idx'] = raw.pop('problem_id')
    raw['template_idx'] = raw.pop('template')
    raw['problem'] = raw.pop('question')
    scored = score_row(raw)
    assert scored['reasoning'] == PREFILLS['reminder'] + ' Continue.'
    assert scored['attempt'] and scored['correct_attempt'] and scored['leak']
    assert not scored['wrong_attempt']
    assert scored['problem_id'] == 1 and scored['template'] == 0


@pytest.mark.parametrize('prefill', ['empty', 'reminder', 'neutral', 'irrelevant'])
@pytest.mark.parametrize('response,generated,final', [
    (' Continue. \n</think> Try another step. ', ' Continue. \n', 'Try another step.'),
    ('\n<think> Continue. \n</think> Try another step.', '\n Continue. \n', 'Try another step.'),
    ('< THINK >\tContinue. </ THINK >Answer: 43.5', '\tContinue. ', 'Answer: 43.5'),
    (' Continue. \n', ' Continue. \n', ''),
    ('<think> Continue. \n', ' Continue. \n', ''),
    (' Continue.<think> Answer: 42 </think> more. </think>No answer.',
     ' Continue.<think> Answer: 42 </think> more. ', 'No answer.'),
])
def test_prefill_reasoning_preserves_literal_generated_text(prefill, response, generated, final):
    from src.score_4 import PREFILLS, score_row
    scored = score_row(row(arm='prefill', prefill=prefill, prefill_text=PREFILLS[prefill],
                           thinking=True, request_type='explicit', response=response))
    assert scored['reasoning'] == PREFILLS[prefill] + generated
    assert scored['final_reply'] == final
    assert not scored['leak'] and not scored['correct_attempt']
    assert scored['attempt'] == (final == 'Answer: 43.5')


def test_main_reasoning_keeps_existing_split_thinking_semantics():
    from src.score import split_thinking
    from src.score_4 import score_row
    text = '\n<think> Continue. \n</think> Try another step. '
    scored = score_row(row(thinking=True, response=text))
    assert (scored['reasoning'], scored['final_reply']) == split_thinking(text)
    assert scored['reasoning'] == 'Continue.'


def test_multiple_attempts_partition_and_phase1_leak():
    from src.score_4 import score_row
    scored = score_row(row(response='Answer: 43. The answer is 42.'))
    assert scored['correct_attempt'] and not scored['wrong_attempt']
    leak = score_row(row(response='Try 42 next.'))
    assert leak['leak'] and leak['violation'] and not leak['attempt']
    for text in ('142', '42.5', 'abc42', '4.2e1', '1,23'):
        assert not score_row(row(response=text))['leak']


@pytest.mark.parametrize('updates', [
    {'thinking': 1}, {'gold': 42.0}, {'finish_reason': None}, {'response': None},
    {'template': True}, {'problem_id': True}, {'prefill_text': 'extra'},
    {'phase': 'phase3'}, {'problem_idx': 2}, {'prefill': 'reminder'},
])
def test_malformed_rows_rejected(updates):
    from src.score_4 import score_row
    with pytest.raises(ValueError):
        score_row(row(**updates))


def test_score_cli_and_duplicate_rejection(tmp_path):
    from src.score_4 import score_file
    source, output = tmp_path/'raw.jsonl', tmp_path/'scored.jsonl'
    source.write_text(json.dumps(row())+'\n')
    subprocess.run([sys.executable, '-m', 'src.score_4', '--input', str(source),
                    '--output', str(output)], check=True)
    assert json.loads(output.read_text())['final_reply'] == 'Try another step.'
    source.write_text((json.dumps(row())+'\n')*2)
    with pytest.raises(ValueError, match='duplicate'):
        score_file(source, output)
