"""Token-aware split tests, including ordinary text quoting marker spellings."""
import pytest

from src.models_5 import GEMMA, NEMOTRON, GLM, model_spec


class Tokenizer:
    """Small reversible tokenizer using each family's real protocol IDs."""
    def __init__(self, model=GEMMA):
        self.special = model_spec(model)['special_tokens']
        self.all_special_ids = list(self.special.values())

    def encode(self, text, add_special_tokens=False, literal=False):
        ids = []
        while text:
            matched = next((s for s in self.special if text.startswith(s)), None) if not literal else None
            if matched:
                ids.append(self.special[matched]); text = text[len(matched):]
            else:
                ids.append(ord(text[0]) + 1000000); text = text[1:]
        return ids

    def decode(self, ids, skip_special_tokens=False, **kw):
        inverse = {v:k for k,v in self.special.items()}
        return ''.join(inverse[i] if i in inverse else chr(i - 1000000) for i in ids)

    def convert_tokens_to_ids(self, s):
        return self.special.get(s)


def split(model, text, **kw):
    from src.phase5_split import split_reasoning
    tok = Tokenizer(model)
    return split_reasoning(model, tok.encode(text), tok, **kw)


def terminator(model):
    return {GEMMA:'<turn|>', NEMOTRON:'<|im_end|>', GLM:'<|user|>'}[model]


def test_gemma_prompt_opened_preserves_prefill_and_whitespace():
    r = split(GEMMA, '  reasoning\n<channel|> Final answer: 42.<turn|>',
              initial_channel='thought', prefill_text='literal prefill')
    assert r['thinking'] == 'literal prefill  reasoning\n'
    assert r['final_reply'] == 'Final answer: 42.' and r['parse_status'] == 'ok'
    assert not r['no_final'] and not r['protocol_failure']
    assert r['raw_token_ids'] and r['raw_text'].endswith('<turn|>')


def test_gemma_native_thought_and_direct_off():
    r = split(GEMMA, '<|channel>thought\n  work<channel|>Guide.<turn|>', initial_channel='auto')
    assert r['thinking'] == '  work' and r['final_reply'] == 'Guide.'
    r = split(GEMMA, 'Guide.<turn|>', initial_channel='final', thinking=False)
    assert r['final_reply'] == 'Guide.' and not r['unexpected_off_thought']


@pytest.mark.parametrize('model', [GEMMA, NEMOTRON, GLM])
def test_off_thought_is_visible(model):
    text = ('<|channel>thought\nanswer 42<channel|>' if model == GEMMA else '<think>answer 42</think>')
    r = split(model, text + 'Guide.' + terminator(model), initial_channel='final', thinking=False)
    assert r['thinking'] == 'answer 42' and r['final_reply'] == 'Guide.'
    assert r['unexpected_off_thought'] and r['protocol_failure']


@pytest.mark.parametrize('model', [GEMMA, NEMOTRON, GLM])
def test_reasoning_only_cap_does_not_leak_into_final(model):
    r = split(model, 'answer is 42', initial_channel='thought', finish_reason='length')
    assert r['thinking'] == 'answer is 42' and r['final_reply'] == '' and r['no_final']
    assert r['parse_status'] == 'truncated_reasoning' and not r['protocol_failure']


@pytest.mark.parametrize('model', [NEMOTRON, GLM])
def test_native_prompt_opened_reasoning_and_direct_off(model):
    r = split(model, '  work\n</think>Guide.' + terminator(model), prefill_text='literal prefill')
    assert r['thinking'] == 'literal prefill  work\n' and r['final_reply'] == 'Guide.'
    assert r['parse_status'] == 'ok' and not r['protocol_failure']
    r = split(model, 'Guide.' + terminator(model), thinking=False)
    assert r['final_reply'] == 'Guide.' and r['thinking'] == '' and r['parse_status'] == 'ok'


@pytest.mark.parametrize('model', [NEMOTRON, GLM])
@pytest.mark.parametrize('suffix', ['', 'terminator'])
def test_native_genuine_closer_required(model, suffix):
    r = split(model, 'answer 42' + (terminator(model) if suffix else ''))
    assert r['thinking'] == 'answer 42' and r['no_final'] and r['final_reply'] == ''
    assert r['parse_status'] == 'unclosed_reasoning' and r['protocol_failure']


@pytest.mark.parametrize('model,text,channel', [
    (GEMMA, 'reason<channel|>a<channel|>b<turn|>', 'thought'),
    (GEMMA, '<|channel>analysis\nwrong<channel|>a<turn|>', 'auto'),
    (GEMMA, 'work<|tool_response>', 'thought'),
    (GEMMA, 'a<turn|>extra', 'final'),
])
def test_malformed_gemma_outputs_are_not_successful_replies(model, text, channel):
    r = split(model, text, initial_channel=channel)
    assert r['protocol_failure'] and r['no_final'] and r['final_reply'] == ''


@pytest.mark.parametrize('model', [NEMOTRON, GLM])
@pytest.mark.parametrize('text', [
    'work<think>nested</think>42',
    'work</think>Guide.</think>',
    'work</think>Guide.<think>more',
    'work<tool_call>calculator</tool_call>',
    'work</think><tool_call>calculator</tool_call>',
    'work<tool_response>42</tool_response>',
])
def test_native_nested_malformed_and_tool_outputs_are_visible_failures(model, text):
    r = split(model, text + terminator(model))
    assert r['protocol_failure'] and r['no_final'] and r['final_reply'] == ''
    assert r['raw_text'] == text + terminator(model)
    assert r['parse_status'] not in ('ok', 'truncated_final', 'truncated_reasoning')


@pytest.mark.parametrize('model', [NEMOTRON, GLM])
def test_native_tokens_after_terminator_and_tool_stop_are_failures(model):
    r = split(model, 'work</think>Guide.' + terminator(model) + 'extra')
    assert r['parse_status'] == 'tokens_after_terminator' and r['no_final']
    tool = '[TOOL_CALLS]' if model == NEMOTRON else '<|observation|>'
    r = split(model, 'work</think>42' + tool)
    assert r['parse_status'] == 'unexpected_tool_handoff' and r['no_final']


@pytest.mark.parametrize('model', [GEMMA, NEMOTRON, GLM])
def test_quoted_marker_text_is_not_a_token_boundary(model):
    from src.phase5_split import split_reasoning
    tok = Tokenizer(model)
    body = 'quote </think> <think> <channel|> <tool_call> are text, answer 42'
    closer = '<channel|>' if model == GEMMA else '</think>'
    ids = tok.encode(body, literal=True) + tok.encode(closer + 'Guide.' + terminator(model))
    r = split_reasoning(model, ids, tok, initial_channel='thought')
    assert r['thinking'] == body and r['final_reply'] == 'Guide.'
    r = split_reasoning(model, tok.encode(body, literal=True), tok,
                        initial_channel='thought', finish_reason='length')
    assert r['no_final'] and r['thinking'] == body
    r = split_reasoning(model, tok.encode(body, literal=True) + tok.encode(terminator(model)),
                        tok, thinking=False)
    assert r['final_reply'] == body and r['parse_status'] == 'ok'


@pytest.mark.parametrize('model', [GEMMA, NEMOTRON, GLM])
def test_truncated_final_is_preserved(model):
    prefix = 'work</think>' if model != GEMMA else ''
    r = split(model, prefix + 'Guide.', initial_channel='thought' if prefix else 'final',
              finish_reason='length')
    assert r['final_reply'] == 'Guide.' and r['parse_status'] == 'truncated_final'


def test_invalid_inputs_rejected_instead_of_coerced():
    from src.phase5_split import split_reasoning
    for ids in (None, 'text', [True], [1.0], [-1]):
        with pytest.raises(ValueError): split_reasoning(GEMMA, ids, Tokenizer())
    for model in (NEMOTRON, GLM):
        with pytest.raises(ValueError):
            split_reasoning(model, [], Tokenizer(model), initial_channel='auto')
        with pytest.raises(ValueError):
            split_reasoning(model, [], Tokenizer(model), initial_channel='final', thinking=True)


@pytest.mark.parametrize('model', [GEMMA, NEMOTRON, GLM])
@pytest.mark.parametrize('incomplete_specials', [False, True])
def test_saved_id_wrapper_replays_without_transformers_and_avoids_factor_collision(model, incomplete_specials):
    from src.phase5_split import split_response, decoded_segments
    tok = Tokenizer(model)
    if incomplete_specials:
        tok.all_special_ids = []  # Native think tokens really have special=False.
    closer = '<channel|>' if model == GEMMA else '</think>'
    ids = tok.encode('work' + closer + 'Guide.' + terminator(model))
    row = {'model':model, 'output_token_ids':ids, 'response':tok.decode(ids), 'thinking':True,
           'initial_channel':'thought', 'prefill_text':'prefill ', 'finish_reason':'stop',
           'decoded_token_segments':decoded_segments(ids, tok), 'special_token_ids':tok.all_special_ids}
    a = split_response(row, tok); b = split_response(row)
    assert a == b and a['reasoning'] == 'prefill work' and a['final_reply'] == 'Guide.'
    assert 'thinking' not in a and row['thinking'] is True
    row['decoded_token_segments'][0]['text'] = 'tampered'
    with pytest.raises(ValueError): split_response(row)


@pytest.mark.parametrize('model', [NEMOTRON, GLM])
def test_saved_ordinary_quoted_closer_cannot_create_a_final(model):
    from src.phase5_split import decoded_segments, split_response
    tok = Tokenizer(model)
    tok.all_special_ids = []
    text = 'quote </think> and <tool_call> with answer 42'
    ids = tok.encode(text, literal=True)
    row = {'model':model, 'thinking':True, 'initial_channel':'thought', 'prefill_text':'',
           'finish_reason':'length', 'output_token_ids':ids, 'response':text,
           'decoded_token_segments':decoded_segments(ids, tok, model), 'special_token_ids':[]}
    result = split_response(row)
    assert result['reasoning'] == text and result['final_reply'] == '' and result['no_final']
    assert result['parse_status'] == 'truncated_reasoning'
