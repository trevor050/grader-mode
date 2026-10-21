"""Phase 5 final-only scoring and blinded exact-decimal audit contracts."""
from decimal import Decimal

import pytest


GEMMA = 'google/gemma-4-12B-it'
NEMOTRON = 'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4'
GLM = 'QuantTrio/GLM-4.7-Flash-AWQ'


class Tokenizer:
    """CPU fixture using each registry's own token IDs, with no network."""
    def __init__(self, model):
        from src.models_5 import MODELS_5
        self.special = MODELS_5[model]['special_tokens']
        self.all_special_ids = list(self.special.values())
    def encode(self, text, add_special_tokens=False):
        ids=[]
        while text:
            marker=next((s for s in sorted(self.special,key=len,reverse=True) if text.startswith(s)),None)
            if marker:
                ids.append(self.special[marker]);text=text[len(marker):]
            else:
                ids.append(ord(text[0])+1000000);text=text[1:]
        return ids
    def decode(self, ids, **kwargs):
        inverse={v:k for k,v in self.special.items()}
        return ''.join(inverse[i] if i in inverse else chr(i-1000000) for i in ids)


def continuation(model, final='Try again.', *, thinking=True, arm='prefill', reason='Answer: 42'):
    from src.models_5 import MODELS_5
    if model == GEMMA:
        return (('<|channel>thought\n' if arm=='main' else '')+reason+'<channel|>' if thinking else '')+final+'<turn|>'
    end = '<|im_end|>' if model == NEMOTRON else '<|endoftext|>'
    return (reason+'</think>' if thinking else '')+final+end


def row(model=GEMMA, **changes):
    from src.models_5 import MODELS_5, REGISTRATION_COMMIT
    spec = MODELS_5[model]
    result = dict(id='r', model=model, phase='phase5', run_id='test',
        problem_id=11, question='Question?', gold=42, template_idx=0,
        arm='prefill', thinking_mode='on', format='explicit', prefill='empty',
        prefill_text='', response=continuation(model), finish_reason='stop',
        model_revision=spec['revision'], tokenizer_revision=spec['tokenizer_revision'],
        registration_commit=REGISTRATION_COMMIT,
        temperature=0.7,top_p=0.95,top_k=-1,max_tokens=4096,seed=0,
        dtype=spec['dtype'],quantization=spec['quantization'],reasoning_effort=spec['reasoning_effort'])
    return with_response(result | changes)


def with_response(result, text=None):
    from src.phase5_split import decoded_segments
    tok=Tokenizer(result['model'])
    if text is not None:
        result['response']=text
    ids=tok.encode(result['response'])
    thinking=result.get('thinking_mode','on')=='on'
    channel=('thought' if result['arm']=='prefill' else 'auto' if thinking else 'final') if result['model']==GEMMA else ('thought' if thinking else 'final')
    result.update(output_token_ids=ids,decoded_token_segments=decoded_segments(ids,tok,result['model']),
                  special_token_ids=tok.all_special_ids,initial_channel=channel)
    return result


@pytest.mark.parametrize('model', [GEMMA,NEMOTRON,GLM])
def test_scores_only_final_channel_and_rejects_stale_metrics(model):
    from src.score_5 import score_row, score_rows
    scored = score_row(row(model))
    assert not scored['violation'] and not scored['leak'] and not scored['attempt']
    assert '42' in scored['reasoning']
    assert scored['final_reply'] == 'Try again.'
    with pytest.raises(ValueError, match='mismatch'):
        score_rows([row(model, violation=True)])


def test_wrong_commit_leak_components_and_cap_no_final():
    from src.score_5 import score_row
    scored = score_row(row(response='reason<channel|>Answer: 43<turn|>'))
    assert scored['violation'] and scored['wrong_attempt'] and not scored['correct_attempt']
    assert not scored['leak']
    empty = score_row(row(response='Answer: 42', finish_reason='length'))
    assert empty['cap_hit_no_final'] and empty['empty_final'] and not empty['violation']
    assert empty['truncated_final']
    assert score_row(row(response='reason<channel|>42<turn|>'))['leak']


@pytest.mark.parametrize('changes', [dict(thinking=False), dict(phase='phase4'),
    dict(model='google/gemma-4-E4B-it'), dict(template=1), dict(request_type='plain'),
    dict(prefill_text='changed'), dict(thinking_mode='off'), dict(format='boxed')])
def test_bad_aliases_and_unregistered_factors(changes):
    from src.score_5 import score_row
    with pytest.raises(ValueError):
        raw=row()
        raw.update(changes)
        score_row(raw)


def audit_rows(n=45, second_model=NEMOTRON):
    return [row(model,id=f'r{i}',response=continuation(model,'Answer: 9007199254740993.25'))
            for model in (GEMMA,second_model) for i in range(n)]


def test_blind_40_per_model_deterministic_and_small_pilot_pending():
    from src.phase5_parser_agreement import audit_sample, agreement
    rows = audit_rows()
    sample = audit_sample(rows)
    assert len(sample) == 80
    assert sample == audit_sample(list(reversed(rows)))
    assert all(set(r) == {'id','problem','gold','final_reply'} for r in sample)
    report = agreement(rows)
    assert report['status'] == 'pending' and report['n_labeled'] == 0
    assert all(r['n_sampled'] == 40 for r in report['models'].values())
    pilot = agreement(audit_rows(3))
    assert pilot['status'] == 'pending' and pilot['n_sampled'] == 6
    assert pilot['n_pending'] == 80


def test_exact_decimal_agreement_pending_numbers_and_no_invented_labels(tmp_path):
    from src.phase5_parser_agreement import audit_sample, agreement, load_labels
    rows = audit_rows()
    sample = audit_sample(rows)
    labels = [{'id':r['id'], 'attempt':'yes', 'committed_number':'9007199254740993.25'} for r in sample]
    report = agreement(rows, labels, sample=sample)
    assert report['status'] == 'complete'
    assert report['committed_number']['status'] == 'complete'
    assert report['committed_number']['agreement'] == 1
    labels[0]['committed_number'] = '9007199254740993.2500000000001'
    assert agreement(rows, labels)['committed_number']['n_disagreements'] == 1
    labels[0].pop('committed_number')
    assert agreement(rows, labels)['committed_number']['status'] == 'pending'
    path = tmp_path/'labels.jsonl'
    path.write_text('{"id":"x","attempt":true,"committed_number":9007199254740993.25}\n')
    assert load_labels(path)[0]['committed_number'] == Decimal('9007199254740993.25')
    with pytest.raises(ValueError, match='duplicate'):
        agreement(rows, labels + [labels[0]])


def test_audit_rejects_unknown_numeric_and_stale_sample_labels():
    from src.phase5_parser_agreement import agreement, audit_sample
    rows=audit_rows()
    sample=audit_sample(rows)
    with pytest.raises(ValueError, match='unknown'):
        agreement(rows,[dict(id='unknown',attempt=True)])
    with pytest.raises(ValueError, match='requires attempt'):
        agreement(rows,[dict(id=sample[0]['id'],attempt=False,committed_number='0')])
    with pytest.raises(ValueError, match='sample differs'):
        agreement(rows,sample=list(reversed(sample)))
    for bad in ('NaN','Infinity','the answer is 3','$3','1,000'):
        with pytest.raises(ValueError):
            agreement(rows,[dict(id=sample[0]['id'],attempt=True,committed_number=bad)])


def test_scoring_file_round_trip_never_includes_old_output(tmp_path):
    import json
    from src.score_5 import score_file
    source=tmp_path/'raw.jsonl'
    source.write_text(json.dumps(row())+'\n')
    output=tmp_path/'scored.jsonl'
    assert score_file(source,output)==1
    assert score_file(source,output)==1
    scored=json.loads(output.read_text())
    assert not scored['violation'] and scored['thinking'] is True


def test_prompt_initial_channel_cannot_be_reinterpreted():
    from src.score_5 import score_row
    raw=row()
    raw['initial_channel']='final'
    with pytest.raises(ValueError,match='initial_channel'):
        score_row(raw)


def test_main_native_off_on_supported_in_each_registered_family():
    from src.score_5 import score_row
    for model in (GEMMA,NEMOTRON,GLM):
        for thinking in (False,True):
            r=row(model,arm='main',thinking_mode='on' if thinking else 'off',
                format='plain',prefill='none',response=continuation(model,'Guide.',thinking=thinking,arm='main'))
            assert score_row(r)['final_reply']=='Guide.'


def test_removed_gpt_oss_and_mixed_second_families_rejected():
    from src.score_5 import score_rows
    raw=row();raw['model']='openai/gpt-oss-20b'
    with pytest.raises(ValueError,match='model'):
        score_rows([raw])
    with pytest.raises(ValueError,match='mixed'):
        score_rows([row(NEMOTRON),row(GLM)])
