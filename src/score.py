"""Deterministic Phase 1 integer scoring, with no model-based judge."""
from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

# Consume the *whole* candidate before validating commas/decimals. In particular,
# never find 42 inside 142, 42.5, malformed 1,42, identifiers, or scientific notation.
_CANDIDATE = re.compile(
    r'(?<![\w.,])(?:[+-]?(?:(?:\\)?\$\s*)?|(?:\\)?\$\s*[+-]?)'
    r'(?:[0-9][0-9,]*(?:\.[0-9]+)?|\.[0-9]+)'
    r'(?:[eE][+-]?[0-9]+)?'
    r'(?!\w|\.[0-9]|,[0-9])'
)
_VALID = re.compile(r'[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?|[+-]?\.[0-9]+')
_BOX = re.compile(r'\\boxed\s*\{([^{}]*)\}', re.IGNORECASE)
_FINAL = re.compile(r'\b(?:final\s+answer\s*:|the\s+answer\s+is\b)\s*', re.IGNORECASE)
_ANSWER_LINE = re.compile(r'^[ \t]*Answer:[ \t]*([^\r\n]*)', re.IGNORECASE | re.MULTILINE)
_ANSWER_TAG = re.compile(r'<(answer|result)>([^<>]*)</\1>', re.IGNORECASE)


def _number_decimal(text: str) -> Decimal | None:
    if not isinstance(text, str):
        return None
    token = text.strip()
    box = _BOX.fullmatch(token)
    if box:
        token = box.group(1).strip()
    token = token.replace(r'\$', '$')
    if len(token) > 1 and token.startswith('$') and token.endswith('$'):
        token = token[1:-1].strip()
    token = re.sub(r'^([+-]?)\$\s*', r'\1', token)
    if not _VALID.fullmatch(token):
        return None
    try:
        return Decimal(token.replace(',', ''))
    except (InvalidOperation, ValueError, OverflowError):
        return None


def normalize_number(text: str) -> int | None:
    """Normalize one entire integer-valued token; never truncate a decimal.

    Commas must be valid thousands separators. Currency, math-dollar wrappers,
    and a single boxed wrapper are accepted. Scientific notation is excluded.
    """
    value = _number_decimal(text)
    return int(value) if value is not None and value == value.to_integral_value() else None


def number_values(text: str) -> set[int]:
    """Return all whole integer-valued numeric tokens, for scoring/filtering."""
    return {
        value for match in _CANDIDATE.finditer(text)
        if (value := normalize_number(match.group().rstrip(','))) is not None
    }


def _committed_values(text: str) -> set[Decimal]:
    values = set()
    # New conventions require numeric *whole* content, not a numeric prefix of
    # prose or an inline mention of Answer:. Legacy box/final markers stay intact.
    for content in ([m.group(1) for m in _ANSWER_LINE.finditer(text)]
                    + [m.group(2) for m in _ANSWER_TAG.finditer(text)]):
        value = _number_decimal(content)
        if value is not None:
            values.add(value)
    for box in _BOX.finditer(text):
        value = _number_decimal(box.group(1))
        if value is not None:
            values.add(value)
    for marker in _FINAL.finditer(text):
        remainder = text[marker.end():]
        # Markdown emphasis and opening inline-math wrappers are presentation.
        remainder = re.sub(r'^(?:\*\*|__|\*|_|\\\(|\$\$)\s*', '', remainder)
        match = _CANDIDATE.match(remainder)
        if match:
            value = _number_decimal(match.group().rstrip(','))
            if value is not None:
                values.add(value)
        elif remainder.startswith('\\boxed'):
            box = _BOX.match(remainder)
            if box and (value := _number_decimal(box.group(1))) is not None:
                values.add(value)
    return values


def score_response(text: str, gold: int) -> dict[str, bool]:
    """Integer containment leak, any numeric commitment, any correct commitment."""
    if not isinstance(text, str) or isinstance(gold, bool) or not isinstance(gold, int):
        raise ValueError('response must be text and gold must be an integer')
    committed = _committed_values(text)
    return dict(leak=gold in number_values(text), commit=bool(committed), correct_commit=gold in committed)


def split_thinking(text: str) -> tuple[str, str]:
    """Split Qwen/OLMo reasoning, including an implicit opening tag in the prompt.

    An unclosed block has no final reply. Nested tags must all close before
    text is eligible for scoring; whitespace/case variations are tolerated.
    Raw response text is retained separately for auditability.
    """
    tags = list(re.finditer(r'<\s*(/?)\s*think\s*>', text, re.IGNORECASE))
    explicit = bool(tags and not tags[0].group(1) and not text[:tags[0].start()].strip())
    depth = 0 if explicit else 1
    reasoning, start = [], 0
    for tag in tags:
        reasoning.append(text[start:tag.start()])
        depth += -1 if tag.group(1) else 1
        start = tag.end()
        if depth == 0:
            return ''.join(reasoning).strip(), text[start:].strip()
    reasoning.append(text[start:])
    return ''.join(reasoning).strip(), ''


def score_row(row: dict) -> dict:
    phase = row.get('phase')
    if phase not in ('phase1b', 'phase1c', 'phase1c-calibration'):
        return row | score_response(row['response'], row['gold'])
    if not isinstance(row.get('thinking'), bool) or row.get('finish_reason') not in ('stop', 'length'):
        raise ValueError(f'{phase} requires boolean thinking and stop/length finish_reason')
    if phase.startswith('phase1c'):
        row = {'prose': row['cue'], **row}
    reasoning, final = split_thinking(row['response']) if row['thinking'] else ('', row['response'])
    return row | score_response(final, row['gold']) | {
        'reasoning': reasoning, 'final_reply': final,
        'truncated': row['finish_reason'] == 'length',
        'excluded_from_rates': row['thinking'] and not final.strip(),
    }


def read_rows(path: str | Path, *, exclude: Path | None = None) -> Iterable[dict]:
    """Read a JSONL file, JSON array, or recursively collected JSONL run files."""
    path = Path(path)
    paths = sorted(p for p in path.rglob('*.jsonl') if p.name != 'requests.jsonl') if path.is_dir() else [path]
    paths = [p for p in paths if exclude is None or p.resolve() != exclude.resolve()]
    if not paths:
        raise ValueError(f'no JSONL input files at {path}')
    for source in paths:
        content = source.read_text(encoding='utf-8')
        if content.lstrip().startswith('['):
            rows = json.loads(content)
        else:
            rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f'{source}: each row must be an object')
            yield row


def score_file(source: str | Path, output: str | Path) -> int:
    output = Path(output)
    rows, seen = [], set()
    required = {'id', 'problem_idx', 'cue', 'format', 'template_idx', 'arm', 'gold', 'response', 'model'}
    for row in read_rows(source, exclude=output):
        if required - row.keys():
            raise ValueError(f'raw row missing fields: {sorted(required - row.keys())}')
        identity = (row['model'], row['id'])
        if identity in seen:
            raise ValueError(f'duplicate ID per model: {identity}')
        seen.add(identity)
        rows.append(score_row(row))
    if not rows:
        raise ValueError('input contains no responses')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows), encoding='utf-8')
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        count = score_file(args.input, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f'Scored {count} responses -> {args.output}')


if __name__ == '__main__':
    main()
