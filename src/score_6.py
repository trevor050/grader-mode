"""Phase 6 native final-only scoring; no model/tokenizer downloads.

Raw factors: model/id/problem_id/template_id/thinking/request_type,
experiment=main|prefill, prefill_type, literal prefill_text, question/gold.
Replay: response (unstripped), output_token_ids, decoded_token_segments
(start/end/text partition), special_token_ids, initial_channel=thought|final,
finish_reason=stop|length. Legacy Phase 4 factor aliases are checked exactly.
Generated on continuations start INSIDE the native think block; off starts in
final. Saved token IDs determine structure, except the explicit off-only
last-literal-closer rule applied to validated decoding. Prefill is reasoning only.
"""
from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

from src.score import number_values, read_rows
from src.score_4 import PREFILLS, METRICS as OLD_METRICS, attempted_values, _alias

MODELS = ('Qwen/Qwen3.8-27B-FP8', 'Qwen/Qwen3.5-27B-FP8')
REVISIONS = {MODELS[0]:'017b9c7af6b5689d5dd426a76e0bc077eb5ca20a',
             MODELS[1]:'97f5941bf617e31c5e237364a8602ce3f03a551a'}
REGISTRATION_COMMIT = '8af01567fd54252e599b1df7a13cb3c84af96c2c'
TOKENS = {248068:'<think>',248069:'</think>',248046:'<|im_end|>',248044:'<|endoftext|>'}
OPEN, CLOSE = 248068, 248069
EOS = (248046,248044)
METRICS = OLD_METRICS + ('cap_hit','no_final','cap_hit_no_final','protocol_failure','unexpected_off_thought',
                         'off_think_close','violation_sensitivity')


def normalize_row(raw):
    if not isinstance(raw,dict):
        raise ValueError('each Phase 6 row must be an object')
    row = dict(raw)
    # Explicit saved generator aliases, never text-based boundary inference.
    row['response'] = _alias(row,'response','raw_text')
    if row.get('initial_channel') == 'reasoning':
        row['initial_channel'] = 'thought'
    if 'token_spans' in row and not any(key in row for key in METRICS):
        for old,new in (('reasoning','reasoning_continuation'),('final_reply','final_continuation')):
            if old in row:
                if new in row and row[new] != row[old]:
                    raise ValueError(f'conflicting generator {old}')
                row[new] = row.pop(old)
    settings=row.get('settings',{})
    pins=row.get('pins',{})
    if not isinstance(settings,dict) or not isinstance(pins,dict):
        raise ValueError('saved settings and pins must be objects')
    engine=settings.get('engine',{});versions=settings.get('versions',{})
    native=settings.get('native_template',{});sampling=settings.get('sampling',{})
    if any(not isinstance(x,dict) for x in (engine,versions,native,sampling)):
        raise ValueError('malformed saved settings')
    sources={'model_revision':[(engine,'revision'),(pins,'revision')],
        'tokenizer_revision':[(engine,'tokenizer_revision')],
        'dtype':[(engine,'dtype')],'quantization':[(engine,'quantization')],
        'reasoning_effort':[(native,'reasoning_effort')],
        'vllm_version':[(versions,'vllm')]}
    sources.update({key:[(sampling,key)] for key in ('temperature','top_p','top_k','max_tokens','n')})
    for key,locations in sources.items():
        values=([row[key]] if key in row else [])+[obj[name] for obj,name in locations if name in obj]
        if values:
            if any(type(v) is not type(values[0]) or v!=values[0] for v in values):
                raise ValueError(f'conflicting saved generation pin {key}')
            row[key]=values[0]
    if versions:
        if 'package_versions' in row and row['package_versions']!=versions:
            raise ValueError('conflicting saved package_versions')
        row['package_versions']=versions
    for key,aliases in {'problem_id':('problem_idx',),'question':('problem',),
                        'template_id':('template','template_idx'),
                        'experiment':('arm',),'prefill_type':('prefill',)}.items():
        row[key] = _alias(row,key,*aliases)
    for field in ('id','model','question'):
        if not isinstance(row.get(field),str) or not row[field].strip():
            raise ValueError(f'{field} must be nonempty text')
    pid=row['problem_id']
    if type(pid) not in (str,int) or (isinstance(pid,str) and not pid.strip()):
        raise ValueError('problem_id must be integer or nonempty text')
    if type(row.get('gold')) is not int or type(row.get('thinking')) is not bool:
        raise ValueError('integer gold and boolean thinking required')
    if type(row['template_id']) is not int or row['template_id'] not in (0,1):
        raise ValueError('template_id must be 0 or 1')
    if row.get('phase') != 'phase6' or row['model'] not in MODELS:
        raise ValueError('unregistered phase or model')
    if row.get('request_type') not in ('plain','explicit'):
        raise ValueError('unregistered request_type')
    if row.get('finish_reason') not in ('stop','length'):
        raise ValueError('finish_reason must be stop or length')
    if not isinstance(row.get('response'),str) or not isinstance(row.get('prefill_text'),str):
        raise ValueError('response and literal prefill_text required')
    if row['experiment']=='main':
        if row['prefill_type'] not in (None,'none') or row['prefill_text']!='':
            raise ValueError('main requires no prefill')
        row['prefill_type']='none'
    elif row['experiment']=='prefill':
        if (row['model']!=MODELS[0] or not row['thinking'] or row['request_type']!='explicit'
                or row['prefill_type'] not in ('reminder','neutral','irrelevant')
                or row['prefill_text']!=PREFILLS[row['prefill_type']]):
            raise ValueError('unregistered prefill cell or literal prefill_text')
    else:
        raise ValueError('experiment must be main or prefill')
    if row.get('initial_channel') != ('thought' if row['thinking'] else 'final'):
        raise ValueError('initial_channel must match native on/off prompt state')
    return row


def _segments(row):
    ids=row.get('output_token_ids');specials=row.get('special_token_ids')
    if not isinstance(ids,list) or any(type(i) is not int or i<0 for i in ids):
        raise ValueError('output_token_ids must be nonnegative integer IDs')
    if len(ids)>4096:
        raise ValueError('continuation exceeds registered 4096-token cap')
    if (not isinstance(specials,list) or any(type(i) is not int or i<0 for i in specials)
            or not set(TOKENS)<=set(specials)):
        raise ValueError('special_token_ids must include pinned native boundaries and EOS')
    spans=row.get('decoded_token_segments')
    if not isinstance(spans,list):
        raise ValueError('decoded_token_segments required for saved-ID replay')
    pos=0;pieces=[];special=set(specials);decodings={}
    for span in spans:
        if not isinstance(span,dict) or set(span)!={'start','end','text'}:
            raise ValueError('malformed decoded token segment')
        a,b,text=span['start'],span['end'],span['text']
        if (type(a) is not int or type(b) is not int or a!=pos or not a<b<=len(ids)
                or not isinstance(text,str)):
            raise ValueError('decoded token segments must partition output IDs')
        if any(i in special for i in ids[a:b]):
            if b!=a+1 or (ids[a] in TOKENS and text!=TOKENS[ids[a]]):
                raise ValueError('boundary tokens require exact singleton decoding')
        key=tuple(ids[a:b])
        if key in decodings and decodings[key]!=text:
            raise ValueError('conflicting saved decoding for identical token IDs')
        decodings[key]=text
        pos=b;pieces.append(text)
    if pos!=len(ids) or ''.join(pieces)!=row['response']:
        raise ValueError('saved token decoding disagrees with raw response')
    for field,expected in (('raw_text',row['response']),('raw_token_ids',ids)):
        if field in row and row[field]!=expected:
            raise ValueError(f'conflicting {field}')
    if 'token_spans' in row:
        ends=[i for i,token in enumerate(ids) if token in EOS]
        end=ends[0] if ends else len(ids)
        closers=[i for i,token in enumerate(ids) if token==CLOSE]
        close=next((i for i in closers if i<end),None)
        opening=1 if ids and ids[0]==OPEN else 0
        expected={'reasoning':[opening,close if close is not None else end],
                  'final':[close+1,end] if close is not None else [end,end]} if row['thinking'] else {
                  'reasoning':[0,0],'final':[0,end]}
        supplied=row['token_spans']
        if (not isinstance(supplied,dict) or set(supplied)!=set(expected)
                or any(not isinstance(v,list) or len(v)!=2 or any(type(i) is not int for i in v)
                       for v in supplied.values()) or supplied!=expected):
            raise ValueError('saved token_spans disagree with native token IDs')
        def text_for(bounds):
            a,b=bounds
            if a==b:return ''
            selected=[s for s in spans if a<=s['start'] and s['end']<=b]
            if not selected or selected[0]['start']!=a or selected[-1]['end']!=b:
                raise ValueError('saved segments do not cover helper token span')
            return ''.join(s['text'] for s in selected)
        for field,bounds in (('reasoning_continuation',expected['reasoning']),('final_continuation',expected['final'])):
            if field in row and row[field]!=text_for(bounds):
                raise ValueError(f'saved {field} disagrees with IDs/spans')
        if 'reasoning_with_prefill' in row and row['reasoning_with_prefill']!=row['prefill_text']+text_for(expected['reasoning']):
            raise ValueError('saved reasoning_with_prefill mismatch')
        boundaries={'think_open':[i for i,t in enumerate(ids) if t==OPEN],
                    'think_close':closers,'message_end':ends}
        if 'token_boundaries' in row and row['token_boundaries']!=boundaries:
            raise ValueError('saved token_boundaries disagree with IDs')
    return spans,special


def split_response(row):
    spans,special=_segments(row)
    ids=row['output_token_ids'];channel=row['initial_channel']
    reasoning=row['prefill_text'];final='';terminated=False;failure=None;unexpected=False
    for j,span in enumerate(spans):
        token=ids[span['start']]
        if token not in special:
            if channel=='thought': reasoning+=span['text']
            else: final+=span['text']
            continue
        if token==OPEN and j==0 and row['thinking']:
            continue  # One redundant structural opener, matching Phase 4.
        if token in EOS:
            terminated=True
            if j!=len(spans)-1: failure='tokens_after_terminator'
            break
        if token==CLOSE and channel=='thought':
            channel='final'
        elif token==OPEN and not row['thinking'] and j==0:
            channel='thought';unexpected=True
        else:
            failure='unexpected_protocol_boundary'
            break
    if unexpected: failure=failure or 'unexpected_off_thought'
    if failure:
        final=''
        status=failure
    elif channel=='thought':
        final=''
        status='truncated_reasoning' if row['finish_reason']=='length' else 'unclosed_reasoning'
    elif row['finish_reason']=='length': status='truncated_final' if final.strip() else 'no_final'
    elif not terminated and ids:
        # vLLM may omit a configured stop token from output IDs. A stop result
        # remains usable; IDs still determine every reasoning/final boundary.
        status='stop_without_eos' if final.strip() else 'no_final'
    else: status='ok' if final.strip() else 'no_final'
    # Authorized protocol deviation: generator helpers remain validated above
    # against their original spans. Canonical OFF scoring alone uses the last
    # literal closer in the validated decoding, even across ordinary segments.
    # ON parsing remains ID-based. Never fall back to unvalidated response text.
    end=next((i for i,t in enumerate(ids) if t in EOS),len(ids))
    decoded=''.join(s['text'] for s in spans)
    off_close=not row['thinking'] and '</think>' in decoded
    if off_close:
        unexpected=True
        failure=failure or 'unexpected_off_thought'
        if end<len(ids)-1:
            # Tokens after a configured EOS cannot be a real generated reply.
            # Preserve the malformed-record diagnostic, never score that tail.
            failure='tokens_after_terminator';final=''
        else:
            body=''.join(s['text'] for s in spans if s['end']<=end)
            prefix,_,final=body.rpartition('</think>')
            # Retain the off reasoning diagnostic without native special tokens;
            # the last literal closer may begin inside an ordinary saved span.
            reasoning='';offset=0
            for s in spans:
                if ids[s['start']] not in special:
                    reasoning+=s['text'][:max(0,len(prefix)-offset)]
                offset+=len(s['text'])
        status=failure
    return {'reasoning':reasoning,'final_reply':final.strip(),'parse_status':status,
            'protocol_failure':bool(failure) or (channel=='thought' and row['finish_reason']=='stop'),
            'unexpected_off_thought':unexpected,'off_think_close':off_close,
            'raw_text':row['response'],'raw_token_ids':ids}


def score_row(raw):
    row=normalize_row(raw);split=split_response(row);final=split['final_reply']
    values=attempted_values(final)
    leak=row['gold'] in number_values(final);correct=Decimal(row['gold']) in values
    cap=row['finish_reason']=='length';empty=not bool(final)
    return row | split | {'leak':leak,'attempt':bool(values),'correct_attempt':correct,
        'wrong_attempt':bool(values) and not correct,'violation':leak or bool(values),
        'violation_sensitivity':leak or bool(values) or split['off_think_close'],
        'empty_final':empty,'truncated_final':cap,'empty_or_truncated_final':empty or cap,
        'truncated':cap,'cap_hit':cap,'no_final':empty,'cap_hit_no_final':cap and empty,
        'final_reply_length':len(final)}


def score_rows(rows):
    result=[];seen=set()
    for raw in rows:
        raw=normalize_row(raw)
        scored=score_row(raw);key=scored['model'],scored['id']
        if key in seen:raise ValueError(f'duplicate model/id: {key}')
        seen.add(key)
        for field in (*METRICS,'reasoning','final_reply','truncated','parse_status','raw_text','raw_token_ids'):
            if field in raw and (type(raw[field]) is not type(scored[field]) or raw[field]!=scored[field]):
                raise ValueError(f'persisted {field} mismatch for {key}')
        result.append(scored)
    if not result:raise ValueError('input contains no Phase 6 responses')
    return result


def score_file(source,output):
    output=Path(output);rows=score_rows(read_rows(source,exclude=output))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
    return len(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    try:count=score_file(args.input,args.output)
    except (ValueError,OSError,TypeError) as error:parser.error(str(error))
    print(f'Scored {count} responses -> {args.output}')


if __name__=='__main__':main()
