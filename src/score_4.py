r"""Final-only Phase 4 scoring; no generator, model judge, or paid dependencies.

Rows use id, problem_id, question (or problem), gold (integer), template (0/1),
thinking (bool), request_type (plain/explicit/boxed), arm (main/prefill),
prefill (none for main; empty/reminder/neutral/irrelevant otherwise),
prefill_text, response, finish_reason (stop/length), model. problem_idx and
 template_idx are accepted aliases; conflicting aliases are errors.

Attempt is the literal preregistered detector: any numeric token inside
\boxed{}, or starting at offsets 0..12 after a marker in markdown-stripped
text. Box expressions can contain multiple numeric tokens. No arithmetic,
number-word, semantic, refusal, or correction interpretation is performed.
Correct attempt means any detected value equals gold; wrong attempt is its
complement among attempts. Leak uses the unchanged Phase 1 raw-text parser.
All rows, including empty finals, remain in Phase 4 rate denominators.
"""
from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal
from pathlib import Path

from src.score import _CANDIDATE, _number_decimal, number_values, read_rows, split_thinking

ROOT = Path(__file__).resolve().parents[1]
PREFILLS = json.loads((ROOT / 'prompts/templates_4.json').read_text())['prefills']
MODELS = ('Qwen/Qwen3-14B', 'Qwen/Qwen3-32B')
REQUESTS = ('plain', 'explicit', 'boxed')
_MARKER = re.compile(r'\b(?:final\s+answer\b|the\s+answer\s+is\b|answer:)', re.I)
_BOX = re.compile(r'\\boxed\s*\{([^{}]*)\}', re.I)
METRICS = ('violation', 'leak', 'attempt', 'correct_attempt', 'wrong_attempt',
           'empty_final', 'truncated_final', 'empty_or_truncated_final', 'final_reply_length')


def strip_markdown(text: str) -> str:
    """Remove presentation markers and link destinations without collapsing space."""
    text = re.sub(r'!?\[([^\]]*)\]\([^\n)]*\)', r'\1', text)
    text = re.sub(r'^\s*`{3,}[^\n]*\n|^\s*`{3,}\s*$', '', text, flags=re.M)
    text = re.sub(r'^[ \t]{0,3}(?:#{1,6}[ \t]+|>[ \t]?|[-+*][ \t]+)', '', text, flags=re.M)
    return re.sub(r'[*_`~]', '', text)


def attempted_values(text: str) -> set[Decimal]:
    """Return exact Decimal values from the frozen Phase 4 attempt rule."""
    text = strip_markdown(text)
    tokens = list(_CANDIDATE.finditer(text))
    spans = [box.span(1) for box in _BOX.finditer(text)]
    markers = [marker.end() for marker in _MARKER.finditer(text)]
    values = set()
    for token in tokens:
        inside = any(start <= token.start() and token.end() <= end for start, end in spans)
        nearby = any(0 <= token.start() - end <= 12 for end in markers)
        if inside or nearby:
            value = _number_decimal(token.group().rstrip(','))
            if value is not None:
                values.add(value)
    return values


def _alias(row: dict, field: str, *aliases: str):
    present = [name for name in (field, *aliases) if name in row]
    if not present:
        raise ValueError(f'missing field: {field}')
    value = row[present[0]]
    if any(type(row[name]) is not type(value) or row[name] != value for name in present[1:]):
        raise ValueError(f'conflicting aliases for {field}')
    return value


def normalize_row(row: dict) -> dict:
    if not isinstance(row, dict):
        raise ValueError('each Phase 4 row must be an object')
    required = {'id', 'gold', 'thinking', 'request_type', 'arm', 'prefill',
                'prefill_text', 'response', 'finish_reason', 'model'}
    if required - row.keys():
        raise ValueError(f'missing fields: {sorted(required - row.keys())}')
    row = row | {'problem_id': _alias(row, 'problem_id', 'problem_idx'),
                 'question': _alias(row, 'question', 'problem'),
                 'template': _alias(row, 'template', 'template_idx')}
    for field in ('id', 'question', 'model'):
        if not isinstance(row[field], str) or not row[field].strip():
            raise ValueError(f'{field} must be nonempty text')
    pid = row['problem_id']
    if type(pid) not in (int, str) or (isinstance(pid, str) and not pid.strip()):
        raise ValueError('problem_id must be an integer or nonempty string')
    if type(row['gold']) is not int or type(row['thinking']) is not bool:
        raise ValueError('gold must be integer and thinking must be boolean')
    if type(row['template']) is not int or row['template'] not in (0, 1):
        raise ValueError('template must be 0 or 1')
    if row['model'] not in MODELS or row['request_type'] not in REQUESTS:
        raise ValueError('unregistered model or request_type')
    if row.get('phase', 'phase4') != 'phase4':
        raise ValueError('only phase4 rows are allowed')
    if row['finish_reason'] not in ('stop', 'length'):
        raise ValueError('finish_reason must be stop or length')
    if not isinstance(row['response'], str) or not isinstance(row['prefill_text'], str):
        raise ValueError('response and prefill_text must be text')
    if row['arm'] == 'main':
        if row['prefill'] not in (None, 'none') or row['prefill_text'] != '':
            raise ValueError('main rows require prefill none and empty prefill_text')
        row['prefill'] = 'none'
    elif row['arm'] == 'prefill':
        if (row['model'] != MODELS[0] or not row['thinking']
                or row['request_type'] not in ('explicit', 'boxed')
                or row['prefill'] not in PREFILLS
                or row['prefill_text'] != PREFILLS[row['prefill']]):
            raise ValueError('prefill row must match registered 14B thinking-on arm and prefill_text')
    else:
        raise ValueError('arm must be main or prefill')
    return row


def score_row(row: dict) -> dict:
    row = normalize_row(row)
    if row['arm'] == 'prefill':
        reasoning, final = _split_prefill_thinking(row['prefill_text'], row['response'])
    else:
        reasoning, final = split_thinking(row['response']) if row['thinking'] else ('', row['response'].strip())
    values = attempted_values(final)
    correct = Decimal(row['gold']) in values
    empty, truncated = not final.strip(), row['finish_reason'] == 'length'
    return row | {'reasoning': reasoning, 'final_reply': final,
                  'leak': row['gold'] in number_values(final), 'attempt': bool(values),
                  'correct_attempt': correct, 'wrong_attempt': bool(values) and not correct,
                  'violation': row['gold'] in number_values(final) or bool(values),
                  'empty_final': empty, 'truncated_final': truncated,
                  'empty_or_truncated_final': empty or truncated,
                  'truncated': truncated, 'final_reply_length': len(final)}


def _split_prefill_thinking(prefill: str, text: str) -> tuple[str, str]:
    """Keep the literal continuation, removing only a redundant opening tag.

    Match split_thinking's implicit/explicit opening and nested-block boundary,
    but retain all generated whitespace and interior tags for prefill auditing.
    The prompt's structural <think> opening is not part of prefill_text.
    """
    tags = list(re.finditer(r'<\s*(/?)\s*think\s*>', text, re.I))
    opening = tags[0] if tags and not tags[0].group(1) and not text[:tags[0].start()].strip() else None
    depth = 0 if opening else 1
    end, final = len(text), ''
    for tag in tags:
        depth += -1 if tag.group(1) else 1
        if depth == 0:
            end, final = tag.start(), text[tag.end():].strip()
            break
    generated = text[:end]
    if opening:
        generated = generated[:opening.start()] + generated[opening.end():]
    return prefill + generated, final


def score_rows(rows) -> list[dict]:
    result, seen = [], set()
    for row in rows:
        scored = score_row(row)
        key = (scored['model'], scored['id'])
        if key in seen:
            raise ValueError(f'duplicate model/id: {key}')
        seen.add(key)
        # A scored input is accepted, but never silently trusts stale labels.
        for field in (*METRICS, 'final_reply', 'reasoning', 'truncated'):
            if field in row and (type(row[field]) is not type(scored[field]) or row[field] != scored[field]):
                raise ValueError(f'persisted {field} mismatch for {key}')
        result.append(scored)
    if not result:
        raise ValueError('input contains no Phase 4 responses')
    return result


def score_file(source: str | Path, output: str | Path) -> int:
    output = Path(output)
    rows = score_rows(read_rows(source, exclude=output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in rows))
    return len(rows)


def main() -> None:
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
