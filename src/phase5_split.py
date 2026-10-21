"""Strict token-ID Gemma/native think-tag splitters, isolated from Phase 1–4 scoring.

``split_reasoning(model_key, output_ids, tokenizer, initial_channel=...,
thinking=..., prefill_text=..., finish_reason=...)`` returns audit text and
final-only scoring text. All boundary decisions use pinned protocol token IDs.
Ordinary tokens spelling a quoted marker never become protocol structure.
The prompt's prefill belongs only in ``thinking``; raw output is untouched.
"""
from src.models_5 import MODELS_5, model_spec


def split_reasoning(model_key, token_ids, tokenizer, *, initial_channel=None,
                    thinking=True, prefill_text='', finish_reason='stop'):
    spec = model_spec(model_key)
    if (not isinstance(token_ids, (list, tuple))
            or any(type(i) is not int or i < 0 for i in token_ids)):
        raise ValueError('token_ids must be a sequence of nonnegative integer IDs')
    if type(thinking) is not bool or not isinstance(prefill_text, str):
        raise ValueError('thinking must be boolean and prefill_text must be text')
    if finish_reason not in ('stop', 'length'):
        raise ValueError('finish_reason must be stop or length')
    ids = list(token_ids)
    decode = lambda seq: tokenizer.decode(seq, skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)
    raw = decode(ids)
    special = set(getattr(tokenizer, 'all_special_ids', [])) | set(spec['special_tokens'].values())
    channel = initial_channel
    if channel is None:
        channel = ('thought' if spec['format'] == 'think_tags' else 'auto') if thinking else 'final'
    allowed = ('thought', 'final') if spec['format'] == 'think_tags' else ('auto', 'thought', 'final')
    if channel not in allowed or (prefill_text and channel != 'thought'):
        raise ValueError('initial channel/prefill is incompatible with the registered format')
    if spec['format'] == 'think_tags' and thinking and channel != 'thought':
        raise ValueError('Native thinking-on prompts already contain an open reasoning block')
    if not thinking and channel != 'final':
        raise ValueError('off continuations start after the native empty thought block')
    result = {'thinking': prefill_text, 'final_reply': '', 'parse_status': 'ok',
              'no_final': True, 'protocol_failure': False, 'unexpected_off_thought': False,
              'raw_token_ids': ids, 'raw_text': raw}

    def fail(status):
        result.update(final_reply='', no_final=True, protocol_failure=True, parse_status=status)
        return result

    def finish(final, in_reasoning=False, terminated=False):
        result['final_reply'] = final.strip()
        result['no_final'] = not bool(result['final_reply'])
        if result['unexpected_off_thought']:
            result.update(parse_status='unexpected_off_thought', protocol_failure=True)
        elif in_reasoning:
            result['parse_status'] = 'truncated_reasoning' if finish_reason == 'length' else 'unclosed_reasoning'
            result['protocol_failure'] = finish_reason != 'length'
        elif finish_reason == 'length' and not terminated:
            result['parse_status'] = 'truncated_final' if final.strip() else 'no_final'
        elif not terminated and ids:
            return fail('missing_terminator')
        elif not final.strip():
            result['parse_status'] = 'no_final'
        return result

    def next_special(start):
        return next((i for i in range(start, len(ids)) if ids[i] in special), len(ids))

    if spec['format'] == 'gemma4':
        pos = 0
        if ids and ids[0] == 100:
            if channel == 'thought':
                return fail('duplicate_thought_header')
            end = next_special(1)
            header_and_body = decode(ids[1:end])
            if not header_and_body.startswith('thought\n'):
                return fail('malformed_thought_header')
            result['thinking'] += header_and_body[len('thought\n'):]
            result['unexpected_off_thought'] = not thinking
            channel, pos = 'thought', end
        if channel == 'thought':
            end = next_special(pos)
            result['thinking'] += decode(ids[pos:end])
            if end == len(ids):
                return finish('', in_reasoning=True)
            if ids[end] != 101:
                return fail('unexpected_tool_handoff' if ids[end] in (46,47,48,49,50,51) else 'malformed_thought_boundary')
            pos = end + 1
        end = next_special(pos)
        final = decode(ids[pos:end])
        if end == len(ids):
            return finish(final)
        if ids[end] == 50:
            return fail('unexpected_tool_handoff')
        if ids[end] not in (1, 106):
            return fail('unexpected_final_boundary')
        if end != len(ids) - 1:
            return fail('tokens_after_terminator')
        return finish(final, terminated=True)

    # Nemotron/GLM native prompts already contain <think> (or the off closer).
    # These added-token IDs need not appear in tokenizer.all_special_ids.
    opener, closer = (spec['special_tokens'][t] for t in ('<think>', '</think>'))
    tools = set(spec['tool_token_ids'])
    pos = 0
    if not thinking and ids and ids[0] == opener:
        result['unexpected_off_thought'] = True
        channel, pos = 'thought', 1
    if channel == 'thought':
        end = next_special(pos)
        result['thinking'] += decode(ids[pos:end])
        if end == len(ids):
            return finish('', in_reasoning=True)
        token = ids[end]
        if token in tools:
            return fail('unexpected_tool_handoff')
        if token == opener:
            return fail('nested_reasoning')
        if token in spec['final_token_ids']:
            if end != len(ids) - 1:
                return fail('tokens_after_terminator')
            return finish('', in_reasoning=True, terminated=True)
        if token != closer:
            return fail('malformed_thought_boundary')
        pos = end + 1
    end = next_special(pos)
    final = decode(ids[pos:end])
    if end == len(ids):
        return finish(final)
    token = ids[end]
    if token in tools:
        return fail('unexpected_tool_handoff')
    if token not in spec['final_token_ids']:
        return fail('unexpected_final_boundary')
    if end != len(ids) - 1:
        return fail('tokens_after_terminator')
    return finish(final, terminated=True)


def split_row(row, tokenizer):
    """Re-split a saved raw generation without reinterpreting its prompt state."""
    return split_reasoning(row['model'], row['output_token_ids'], tokenizer,
                           initial_channel=row['initial_channel'], thinking=row['thinking'],
                           prefill_text=row['prefill_text'], finish_reason=row['finish_reason'])


def decoded_segments(token_ids, tokenizer, model_key=None):
    """Save maximal ordinary-token spans and singleton special tokens for replay.

    Decoding individual BPE tokens can corrupt multibyte characters. Decode
    whole spans instead, using exactly the boundaries the splitter may visit.
    No marker-text search or re-encoding of raw text is involved.
    """
    if (not isinstance(token_ids, (list, tuple))
            or any(type(i) is not int or i < 0 for i in token_ids)):
        raise ValueError('Expected nonnegative integer token IDs')
    special = set(tokenizer.all_special_ids)
    if model_key is not None:
        special.update(model_spec(model_key)['special_tokens'].values())
    else:
        # Native think markers can be added tokens without special=True.
        # Identify the family by its pinned channel/think opener, then add
        # only that format's pinned IDs (ordinary IDs overlap across families).
        for spec in MODELS_5.values():
            marker = '<|channel>' if spec['format']=='gemma4' else '<think>'
            if tokenizer.convert_tokens_to_ids(marker)==spec['special_tokens'][marker]:
                special.update(spec['special_tokens'].values())
    segments, pos = [], 0
    while pos < len(token_ids):
        end = pos + 1
        if token_ids[pos] not in special:
            while end < len(token_ids) and token_ids[end] not in special:
                end += 1
        segments.append({'start':pos, 'end':end,
            'text':tokenizer.decode(token_ids[pos:end],skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)})
        pos=end
    return segments


class _SavedDecoder:
    """Validated saved token spans, sufficient for the strict boundary parser."""
    def __init__(self, row):
        ids=row['output_token_ids']
        if not isinstance(ids,list) or any(type(i) is not int or i<0 for i in ids):
            raise ValueError('output_token_ids must be integer IDs')
        specials=row.get('special_token_ids')
        if not isinstance(specials,list) or any(type(i) is not int or i<0 for i in specials):
            raise ValueError('Saved decoding requires special_token_ids')
        spec=model_spec(row['model'])
        self.all_special_ids=sorted(set(specials) | set(spec['special_tokens'].values()))
        special=set(self.all_special_ids)
        known={v:k for k,v in spec['special_tokens'].items()}
        segments=row.get('decoded_token_segments')
        if not isinstance(segments,list) or not isinstance(row.get('response'),str):
            raise ValueError('Saved decoding requires decoded_token_segments and raw response')
        self.spans={():''}
        pos, pieces=0,[]
        for segment in segments:
            if not isinstance(segment,dict) or set(segment)!={'start','end','text'}:
                raise ValueError('Malformed decoded token segment')
            start,end,text=segment['start'],segment['end'],segment['text']
            if (type(start) is not int or type(end) is not int or start!=pos
                    or not start<end<=len(ids) or not isinstance(text,str)):
                raise ValueError('Decoded token segments must exactly partition output IDs')
            if ids[start] in special:
                if end!=start+1 or (ids[start] in known and text!=known[ids[start]]):
                    raise ValueError('Invalid special-token segment')
            elif any(i in special for i in ids[start:end]) or (end<len(ids) and ids[end] not in special):
                raise ValueError('Ordinary-token segments must end at a genuine special ID')
            key=tuple(ids[start:end])
            if key in self.spans and self.spans[key]!=text:
                raise ValueError('Conflicting saved decoding for identical token IDs')
            self.spans[key]=text
            pieces.append(text);pos=end
        if pos!=len(ids) or ''.join(pieces)!=row['response']:
            raise ValueError('Saved token decoding disagrees with raw response')
        self.spans[tuple(ids)]=row['response']

    def decode(self, ids, **kwargs):
        try:
            return self.spans[tuple(ids)]
        except KeyError as exc:
            raise ValueError('Saved decoder does not cover the requested token span') from exc


def split_response(row, tokenizer=None):
    """Scorer API: saved-ID parsing, returning ``reasoning`` (not factor thinking).

    New raw rows include CPU-replayable decoded spans. Older hand fixtures may
    instead pass their pinned tokenizer explicitly. Text-only parsing is never
    an automatic fallback because it would reinterpret quoted markers.
    """
    required={'model','output_token_ids','response','initial_channel','thinking','prefill_text','finish_reason'}
    if not isinstance(row,dict) or required-row.keys():
        raise ValueError('Missing Phase 5 raw splitting fields')
    if ('raw_token_ids' in row and row['raw_token_ids']!=row['output_token_ids']
            or 'raw_text' in row and row['raw_text']!=row['response']):
        raise ValueError('Conflicting raw continuation aliases')
    decoder=_SavedDecoder(row) if tokenizer is None else tokenizer
    result=split_row(row,decoder)
    if result['raw_text']!=row['response']:
        raise ValueError('Raw response disagrees with decoded output token IDs')
    result['reasoning']=result.pop('thinking')
    return result
