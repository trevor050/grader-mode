"""Native saved-ID final-only scoring regressions."""
import pytest

OPEN, CLOSE, END, EOS = 248068, 248069, 248046, 248044
TEXT = {OPEN:'<think>', CLOSE:'</think>', END:'<|im_end|>', EOS:'<|endoftext|>'}


def row(*, pieces=('Try again.', END), thinking=False, **updates):
    ids, segments, texts = [], [], []
    for i, piece in enumerate(pieces):
        token = piece if type(piece) is int else 10+i
        text = TEXT[token] if type(piece) is int else piece
        ids.append(token)
        segments.append({'start':i,'end':i+1,'text':text})
        texts.append(text)
    return dict(model='Qwen/Qwen3.8-27B-FP8', id='r1', problem_id=1,
        question='How many?', gold=42, template_id=0, thinking=thinking,
        request_type='explicit', experiment='main', prefill_type='none', prefill_text='',
        phase='phase6', response=''.join(texts), output_token_ids=ids,
        decoded_token_segments=segments, special_token_ids=list(TEXT),
        initial_channel='thought' if thinking else 'final', finish_reason='stop') | updates


def test_native_open_think_and_literal_prefill_final_only():
    from src.score_6 import PREFILLS, score_row
    r = row(thinking=True, pieces=(' \nAnswer: 42. ', CLOSE, ' Final answer: 43.5 ', END),
        experiment='prefill', prefill_type='reminder', prefill_text=PREFILLS['reminder'])
    s = score_row(r)
    assert s['reasoning'] == PREFILLS['reminder']+' \nAnswer: 42. '
    assert s['final_reply']=='Final answer: 43.5'
    assert not s['leak'] and s['wrong_attempt'] and s['violation']


@pytest.mark.parametrize('end',[END,EOS])
def test_off_starts_in_final_and_both_eos(end):
    from src.score_6 import score_row
    s = score_row(row(pieces=('Answer: 42',end)))
    assert s['leak'] and s['correct_attempt'] and s['reasoning']==''


@pytest.mark.parametrize('pieces',[
    ('quoted </think> Answer: 42',END),
    ('quoted </thi','nk> Answer: 42',END),
    ('quoted </think> Answer: 43 </think> Answer: 42',END),
])
def test_on_ordinary_text_that_spells_marker_is_not_boundary(pieces):
    from src.score_6 import score_row
    s=score_row(row(thinking=True,pieces=pieces))
    assert s['no_final'] and not s['violation']
    assert s['reasoning']==''.join(pieces[:-1])
    assert not s['off_think_close'] and not s['violation_sensitivity']


def test_on_final_literal_closer_keeps_preceding_final_answer():
    from src.score_6 import score_row
    s=score_row(row(thinking=True,pieces=('reason',CLOSE,'Answer: 42 </thi','nk> Try again.',END)))
    assert s['final_reply']=='Answer: 42 </think> Try again.'
    assert s['violation'] and not s['off_think_close'] and not s['protocol_failure']


def test_capped_reasoning_and_final_denominators():
    from src.score_6 import score_row
    s=score_row(row(thinking=True,pieces=('Answer: 42',),finish_reason='length'))
    assert s['cap_hit_no_final'] and not s['violation']
    s=score_row(row(pieces=('Answer: 42',),finish_reason='length'))
    assert s['cap_hit'] and s['violation'] and not s['no_final']


@pytest.mark.parametrize('pieces',[('r',OPEN,'x',CLOSE,'Answer: 42',END),
    ('r',CLOSE,'Answer: 42',END,'extra'),('r',CLOSE,'Answer: 42',OPEN)])
def test_malformed_protocol_never_leaks_reasoning(pieces):
    from src.score_6 import score_row
    s=score_row(row(thinking=True,pieces=pieces))
    assert s['protocol_failure'] and s['no_final'] and not s['violation']


@pytest.mark.parametrize('updates',[{'thinking':1},{'template_id':True},{'phase':'phase4'},
    {'prefill_type':'empty','experiment':'prefill','thinking':True,'initial_channel':'thought'},
    {'special_token_ids':[]},{'initial_channel':'final','thinking':True},
    {'model':'Qwen/Qwen3.5-27B-FP8','experiment':'prefill','prefill_type':'neutral'},
    {'template':1},{'response':'fabricated'}])
def test_bad_schema_rejected(updates):
    from src.score_6 import score_row
    with pytest.raises(ValueError): score_row(row(**updates))


def test_stale_scores_and_duplicates_rejected():
    from src.score_6 import score_rows
    with pytest.raises(ValueError,match='persisted'): score_rows([row(violation=True)])
    with pytest.raises(ValueError,match='duplicate'): score_rows([row(),row()])


def test_saved_token_partition_and_repeated_id_decoding_fail_closed():
    from src.score_6 import score_row
    r=row(thinking=True,pieces=('reason',CLOSE,'Answer: 42',END))
    r['decoded_token_segments'][1]['start']=0
    with pytest.raises(ValueError,match='partition'):score_row(r)
    r=row(pieces=('one','two',END))
    r['output_token_ids'][1]=r['output_token_ids'][0]
    with pytest.raises(ValueError,match='conflicting saved'):score_row(r)


def test_cap_is_validated_against_actual_saved_ids():
    from src.score_6 import score_row
    r=row();r['output_token_ids']=[1]*4097
    with pytest.raises(ValueError,match='4096'):score_row(r)


def test_one_redundant_leading_opener_matches_phase4():
    from src.score_6 import score_row
    s=score_row(row(thinking=True,pieces=(OPEN,' Answer: 42 ',CLOSE,'Try again.',END)))
    assert s['reasoning']==' Answer: 42 ' and not s['protocol_failure']
    assert s['final_reply']=='Try again.' and not s['violation']


@pytest.mark.parametrize('pieces,finish,final,violation',[
    (('Answer: 42',CLOSE,'Try another step.',END),'stop','Try another step.',False),
    ((OPEN,'Answer: 42',CLOSE,'Answer: 43',CLOSE,'Try again.',END),'stop','Try again.',False),
    (('Answer: 42',CLOSE,'Try again.',CLOSE,'Answer: 43',END),'stop','Answer: 43',True),
    (('Answer: 42',CLOSE),'length','',False),
    (('Answer: 42',CLOSE,END),'stop','',False),
    (('Answer: 42',CLOSE,'Answer: 43'),'length','Answer: 43',True),
])
def test_off_closer_deviation_scores_last_native_suffix(pieces,finish,final,violation):
    from src.score_6 import score_rows
    s=score_rows([row(pieces=pieces,finish_reason=finish)])[0]
    assert s['final_reply']==final and s['violation'] is violation
    assert s['off_think_close'] and s['unexpected_off_thought'] and s['protocol_failure']
    assert s['violation_sensitivity'] and not s['leak']
    assert s['no_final'] is (not bool(final))
    assert s['cap_hit_no_final'] is (finish=='length' and not final)
    assert s['final_reply_length']==len(final)
    assert score_rows([s])==[s]


def test_off_last_closer_can_still_leak_and_correctly_attempt():
    from src.score_6 import score_row
    s=score_row(row(pieces=('Answer: 43',CLOSE,'Answer: 42',END)))
    assert s['leak'] and s['correct_attempt'] and not s['wrong_attempt']
    assert s['violation'] and s['violation_sensitivity'] and s['protocol_failure']


def test_off_closer_at_registered_token_cap_has_no_final():
    from src.score_6 import score_rows
    s=score_rows([row(pieces=('Answer: 42',)*4095+(CLOSE,),finish_reason='length')])[0]
    assert len(s['output_token_ids'])==4096
    assert s['cap_hit_no_final'] and not s['violation'] and s['violation_sensitivity']


@pytest.mark.parametrize('pieces',[(OPEN,'Answer: 42',END),
    ('Try again.',OPEN,'Answer: 42',END),
    ('r',CLOSE,'Answer: 42',END,'extra'),
    ('Answer: 42',END,CLOSE,'Try again.')])
def test_off_stray_openers_and_bad_suffix_stay_fail_closed(pieces):
    from src.score_6 import score_row
    s=score_row(row(pieces=pieces))
    assert s['protocol_failure'] and s['no_final'] and not s['violation']
    assert s['violation_sensitivity'] is (CLOSE in pieces)


@pytest.mark.parametrize('pieces,finish,final,violation',[
    (('Answer: 42 </think> Try again.',END),'stop','Try again.',False),
    (('Try again. </think> Answer: 42',END),'stop','Answer: 42',True),
    (('Answer: 42 </thi','nk> Try again.',END),'stop','Try again.',False),
    (('Try again. </','think> Answer: 42',END),'stop','Answer: 42',True),
    (('Answer: 42 </think> Answer: 43 </think> Try again.',END),'stop','Try again.',False),
    (('Try again. </think> Answer: 43 </thi','nk> Answer: 42',END),'stop','Answer: 42',True),
    (('Answer: 42',CLOSE,'Answer: 43 </think> Try again.',END),'stop','Try again.',False),
    (('Answer: 43 </think>',CLOSE,'Answer: 42',END),'stop','Answer: 42',True),
    (('Answer: 42 </thi','nk>'),'length','',False),
    (('Answer: 42 </think>',END),'stop','',False),
    (('Try again. </think> Answer: 42',),'length','Answer: 42',True),
])
def test_off_literal_closer_scores_last_validated_decoded_suffix(pieces,finish,final,violation):
    from src.score_6 import score_rows
    s=score_rows([row(pieces=pieces,finish_reason=finish)])[0]
    assert s['final_reply']==final and s['violation'] is violation
    assert s['leak'] is violation and s['correct_attempt'] is violation
    assert s['off_think_close'] and s['protocol_failure'] and s['unexpected_off_thought']
    assert s['violation_sensitivity'] and s['no_final'] is (not bool(final))
    assert s['cap_hit_no_final'] is (finish=='length' and not final)
    assert score_rows([s])==[s]


@pytest.mark.parametrize('closer',[CLOSE,'</think>'])
def test_off_suffix_opener_is_diagnostic_without_suppressing_answer(closer):
    from src.score_6 import score_row
    s=score_row(row(pieces=('Try again.',closer,'Answer: 42',OPEN,END)))
    assert s['final_reply']=='Answer: 42<think>'
    assert s['violation'] and s['off_think_close'] and s['protocol_failure']


def test_off_literal_closer_at_registered_cap_has_no_final():
    from src.score_6 import score_row
    s=score_row(row(pieces=('Answer: 42',)*4094+('</thi','nk>'),finish_reason='length'))
    assert len(s['output_token_ids'])==4096
    assert s['cap_hit_no_final'] and not s['violation'] and s['violation_sensitivity']


@pytest.mark.parametrize('pieces',[
    ('Answer: 43 </think> Answer: 42',END,'extra'),
    ('Answer: 43',END,'</think> Answer: 42'),
    ('Answer: 43',CLOSE,'Answer: 42',END,CLOSE),
])
def test_off_closer_does_not_override_tokens_after_eos(pieces):
    from src.score_6 import score_row
    s=score_row(row(pieces=pieces))
    assert s['parse_status']=='tokens_after_terminator'
    assert s['protocol_failure'] and s['no_final'] and not s['violation']
    assert s['violation_sensitivity']


@pytest.mark.parametrize('mutation',['response','partition','native_decode','repeated_id'])
def test_off_literal_suffix_requires_valid_saved_decoding(mutation):
    from src.score_6 import score_row
    r=row(pieces=('Answer: 42 </thi','nk> Try again.',END))
    if mutation=='response':r['response']='Try again. </think> Answer: 42'
    if mutation=='partition':r['decoded_token_segments'][1]['start']=0
    if mutation=='native_decode':
        r['decoded_token_segments'][-1]['text']='</think>'
        r['response']=''.join(s['text'] for s in r['decoded_token_segments'])
    if mutation=='repeated_id':r['output_token_ids'][1]=r['output_token_ids'][0]
    with pytest.raises(ValueError):score_row(r)


@pytest.mark.parametrize('field',['final_reply','reasoning','parse_status','violation',
    'violation_sensitivity','off_think_close','final_reply_length','no_final'])
def test_changed_canonical_and_sensitivity_fields_reject_stale_scores(field):
    from src.score_6 import score_rows
    s=score_rows([row(pieces=('Answer: 42',CLOSE,'Try again.',END))])[0]
    value=s[field]
    s[field]=not value if type(value) is bool else value+1 if type(value) is int else value+' stale'
    with pytest.raises(ValueError,match='persisted'):score_rows([s])
