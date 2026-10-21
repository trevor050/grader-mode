"""Phase 5 final-only scoring, isolated from Phase 1–4 behavior.

Raw rows carry model, phase, run_id, id, problem_id (problem_idx alias),
question (problem alias), integer gold, template_idx (template alias), arm,
thinking_mode (on/off; boolean thinking alias), format (plain/explicit;
request_type alias), prefill, literal prefill_text, response and finish_reason. Saved splitting
requires output_token_ids, initial_channel, decoded_token_segments and
special_token_ids from the Phase 5 generator; no text-only fallback exists.
Analysis additionally requires per-row generation pins; see analyze_5.
Persisted parser/score outputs are recomputed and checked, never trusted.
"""
from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

from src.models_5 import MODELS_5, GEMMA, NEMOTRON, GLM, PRIMARY_MODELS
from src.score import number_values, read_rows
from src.score_4 import PREFILLS, METRICS as PHASE4_METRICS, attempted_values, _alias

MODELS = PRIMARY_MODELS
REVISIONS = {model: spec['revision'] for model,spec in MODELS_5.items()}
REQUESTS = ('plain', 'explicit')
METRICS = PHASE4_METRICS + ('cap_hit', 'no_final', 'cap_hit_no_final',
                               'protocol_failure', 'unexpected_off_thought')


def normalize_row(row):
    if not isinstance(row, dict):
        raise ValueError('each Phase 5 row must be an object')
    required = {'model', 'phase', 'id', 'gold', 'arm', 'prefill', 'prefill_text',
                'response', 'finish_reason'}
    if required - row.keys():
        raise ValueError(f'missing fields: {sorted(required - row.keys())}')
    row = row | {'problem_id': _alias(row, 'problem_id', 'problem_idx'),
                 'question': _alias(row, 'question', 'problem'),
                 'template': _alias(row, 'template', 'template_idx'),
                 'request_type': _alias(row, 'request_type', 'format')}
    if 'thinking_mode' in row:
        mode = row['thinking_mode']
        if mode not in ('on', 'off'):
            raise ValueError('thinking_mode must be on or off')
        thinking = mode == 'on'
        if 'thinking' in row and (type(row['thinking']) is not bool or row['thinking'] != thinking):
            raise ValueError('conflicting aliases for thinking')
    elif type(row.get('thinking')) is bool:
        thinking = row['thinking']
    else:
        raise ValueError('thinking_mode or boolean thinking required')
    row.update(thinking=thinking, thinking_mode='on' if thinking else 'off',
               format=row['request_type'], template_idx=row['template'])
    for key in ('id', 'question', 'model'):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f'{key} must be nonempty text')
    if type(row['problem_id']) not in (str, int) or (isinstance(row['problem_id'], str) and not row['problem_id'].strip()):
        raise ValueError('problem_id must be integer or nonempty text')
    if type(row['gold']) is not int:
        raise ValueError('gold must be integer')
    if type(row['template']) is not int or row['template'] not in (0, 1):
        raise ValueError('template must be 0 or 1')
    if row['phase'] != 'phase5' or row['model'] not in MODELS_5 or row['format'] not in REQUESTS:
        raise ValueError('unregistered phase, model or request format')
    if row['finish_reason'] not in ('stop', 'length'):
        raise ValueError('finish_reason must be stop or length')
    if not isinstance(row['response'], str) or not isinstance(row['prefill_text'], str):
        raise ValueError('response and prefill_text must be text')
    if row['arm'] == 'main':
        if row['prefill'] not in (None, 'none') or row['prefill_text'] != '':
            raise ValueError('main requires prefill none and empty prefill_text')
        row['prefill'] = 'none'
    elif row['arm'] == 'prefill':
        if (not thinking or row['format'] != 'explicit' or row['prefill'] not in PREFILLS
                or row['prefill_text'] != PREFILLS[row['prefill']]):
            raise ValueError('prefill requires thinking on, explicit and literal registered prefill_text')
    else:
        raise ValueError('arm must be main or prefill')
    expected_channel = ('final' if not thinking else 'auto'
        if row['model'] == GEMMA and row['arm'] == 'main' else 'thought')
    if row.get('initial_channel') != expected_channel:
        raise ValueError('initial_channel does not match the registered prompt state')
    return row


def active_models(rows):
    """Expected audit/grid pair, retaining an absent family as pending in pilots."""
    present = {r['model'] for r in rows}
    if NEMOTRON in present and GLM in present:
        raise ValueError('mixed Nemotron/GLM families cannot form one Phase 5 run')
    return (GEMMA, GLM if GLM in present else NEMOTRON)


def score_row(row):
    row = normalize_row(row)
    from src.phase5_split import split_response
    split = split_response(row)
    final, reasoning = split['final_reply'], split['reasoning']
    values = attempted_values(final)
    leak, correct = row['gold'] in number_values(final), Decimal(row['gold']) in values
    cap, empty = row['finish_reason'] == 'length', not final.strip()
    return row | split | {'reasoning': reasoning, 'final_reply': final,
        'leak': leak, 'attempt': bool(values), 'correct_attempt': correct,
        'wrong_attempt': bool(values) and not correct, 'violation': leak or bool(values),
        'empty_final': empty, 'truncated_final': cap, 'empty_or_truncated_final': empty or cap,
        'truncated': cap, 'final_reply_length': len(final), 'cap_hit': cap,
        'no_final': empty, 'cap_hit_no_final': cap and empty}


def score_rows(rows):
    result, seen = [], set()
    for raw in rows:
        scored = score_row(raw)
        key = (scored['model'], scored['id'])
        if key in seen:
            raise ValueError(f'duplicate model/id: {key}')
        seen.add(key)
        for field in (*METRICS, 'reasoning', 'final_reply', 'truncated', 'parse_status', 'protocol_failure', 'unexpected_off_thought',
                      'raw_text', 'raw_token_ids'):
            if field in raw and field in scored and (type(raw[field]) is not type(scored[field]) or raw[field] != scored[field]):
                raise ValueError(f'persisted {field} mismatch for {key}')
        result.append(scored)
    if not result:
        raise ValueError('input contains no Phase 5 responses')
    active_models(result)
    return result


def score_file(source, output):
    output = Path(output)
    rows = score_rows(read_rows(source, exclude=output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows))
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        count = score_file(args.input, args.output)
    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))
    print(f'Scored {count} responses -> {args.output}')


if __name__ == '__main__':
    main()
