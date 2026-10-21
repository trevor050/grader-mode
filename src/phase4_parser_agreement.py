"""Blind seed-0 stratified Phase 4 audit and agreement against lead labels.

The 60 replies are stratified by model/arm/thinking/request/prefill/attempt.
Round-robin allocation gives each nonempty stratum one row before another
round, redistributing exhausted strata. Rows and strata are sorted before
seed-0 shuffling; the final order is shuffled. IDs hash (model, source id),
hiding experimental factors and disambiguating identical IDs across models.

CLI: --input RAW_OR_SCORED --sample parser-audit-sample.jsonl
     --labels lead.jsonl|lead.csv --output parser-agreement.json
Labels contain id and attempt (JSON bool, integer 0/1, or case-insensitive
true/false, yes/no, 0/1 strings). Blank/null attempts remain pending. Optional
committed_number is one finite signed decimal, supplied as a JSON number or
numeric string (exponents allowed; no grouping, currency, or prose). Blank,
null, or absent numbers remain pending for lead-yes rows. A nonblank number
requires a lead-yes attempt; zero is a real label. Use strings or Decimal for
lossless in-memory labels; JSON files are loaded directly as Decimal.

Top-level agreement/status retain the attempt-only API, with confusion counts
and Cohen's kappa (null for constant marginals). The separate committed_number
report counts exact Decimal membership in the detected attempt values, not
equality to gold. One labeled value matching any detected value agrees, even
if other values were detected. No parser detection means numeric disagreement.
Numeric n_eligible counts lead-yes rows; n_pending counts missing numbers on
those rows plus unknown attempt labels. Numeric status is complete only with
all 60 attempt labels, all lead-yes numbers, and at least one numeric label;
agreement is null without numeric labels. Lead-no rows need no number.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from src.score import read_rows
from src.score_4 import attempted_values, score_rows

SAMPLE_SIZE = 60
SEED = 0


def _blind_id(row):
    identity = json.dumps([row['model'], row['id']], ensure_ascii=False, separators=(',', ':'))
    return 'p4-' + hashlib.sha256(identity.encode()).hexdigest()[:24]


def _audit_scored(rows):
    rows = score_rows(rows)
    if len(rows) < SAMPLE_SIZE:
        raise ValueError('parser audit requires at least 60 replies')
    strata = defaultdict(list)
    for row in sorted(rows, key=lambda r: (r['model'], r['id'])):
        key = (row['model'], row['arm'], row['thinking'], row['request_type'], row['prefill'], row['attempt'])
        strata[key].append(row)
    rng = random.Random(SEED)
    groups = [strata[key] for key in sorted(strata)]
    for group in groups:
        rng.shuffle(group)
    rng.shuffle(groups)
    selected = []
    while len(selected) < SAMPLE_SIZE:
        for group in groups:
            if group and len(selected) < SAMPLE_SIZE:
                selected.append(group.pop())
    rng.shuffle(selected)
    return selected


def audit_sample(rows) -> list[dict]:
    return [{'id':_blind_id(r), 'problem':r['question'], 'gold':r['gold'], 'final_reply':r['final_reply']}
            for r in _audit_scored(rows)]


def write_audit_sample(rows, path: str | Path) -> list[dict]:
    sample = audit_sample(rows)
    path = Path(path)
    if path.exists():
        existing = list(read_rows(path))
        if existing != sample:
            raise ValueError(f'existing audit sample differs at {path}; use a fresh output directory')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in sample))
    return sample


def _label(value):
    if value is None or value == '':
        return None
    if type(value) is bool:
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized in ('true', '1', 'yes', 'false', '0', 'no'):
            return normalized in ('true', '1', 'yes')
    raise ValueError('attempt labels must be boolean, 0/1, true/false, yes/no, or blank')


def _number_label(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if type(value) not in (str, int, float, Decimal):
        raise ValueError('committed_number must be one finite decimal or blank')
    token = str(value).strip()
    if not re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?', token):
        raise ValueError('committed_number must be one finite decimal or blank')
    try:
        number = Decimal(token)
    except InvalidOperation as error:
        raise ValueError('committed_number is outside the supported decimal range') from error
    if not number.is_finite():
        raise ValueError('committed_number must be finite')
    return number


def agreement(rows, labels=(), *, sample=None) -> dict:
    selected = _audit_scored(rows)
    expected = [{'id':_blind_id(r), 'problem':r['question'], 'gold':r['gold'], 'final_reply':r['final_reply']}
                for r in selected]
    if sample is not None and sample != expected:
        raise ValueError('audit sample differs from deterministic raw-row sample')
    predictions = {_blind_id(r):r['attempt'] for r in selected}
    numbers = {_blind_id(r):attempted_values(r['final_reply']) for r in selected}
    seen, pairs, numeric_pairs = set(), [], []
    for label in labels:
        if not isinstance(label, dict) or 'id' not in label or 'attempt' not in label:
            raise ValueError('lead labels require id and attempt')
        rid = label['id']
        if not isinstance(rid, str) or rid not in predictions:
            raise ValueError(f'unknown audit label id: {rid}')
        if rid in seen:
            raise ValueError(f'duplicate audit label id: {rid}')
        seen.add(rid)
        value = _label(label['attempt'])
        number = _number_label(label.get('committed_number'))
        if number is not None:
            if value is not True:
                raise ValueError(f'committed_number requires attempt yes/true for {rid}')
            numeric_pairs.append((rid, number in numbers[rid]))
        if value is not None:
            pairs.append((rid, predictions[rid], value))
    tp = sum(p and h for _,p,h in pairs)
    tn = sum(not p and not h for _,p,h in pairs)
    fp = sum(p and not h for _,p,h in pairs)
    fn = sum(not p and h for _,p,h in pairs)
    n = len(pairs)
    observed = (tp+tn)/n if n else None
    chance = ((tp+fp)*(tp+fn)+(tn+fn)*(tn+fp))/n**2 if n else None
    n_numeric = len(numeric_pairs)
    numeric_pending = SAMPLE_SIZE-n + tp+fn-n_numeric
    numeric_disagreements = [rid for rid, matches in numeric_pairs if not matches]
    return {'status':'complete' if n == SAMPLE_SIZE else 'pending',
            'n_sampled':SAMPLE_SIZE, 'n_labeled':n, 'n_pending':SAMPLE_SIZE-n,
            'agreement':observed,
            'cohen_kappa':(observed-chance)/(1-chance) if chance is not None and chance < 1 else None,
            'confusion':{'true_positive':tp, 'true_negative':tn, 'false_positive':fp, 'false_negative':fn},
            'n_disagreements':fp+fn,
            'disagreement_ids':[rid for rid,p,h in pairs if p != h],
            'committed_number':{
                'status':'complete' if n_numeric and numeric_pending == 0 else 'pending',
                'n_eligible':tp+fn, 'n_labeled':n_numeric, 'n_pending':numeric_pending,
                'agreement':(n_numeric-len(numeric_disagreements))/n_numeric if n_numeric else None,
                'n_disagreements':len(numeric_disagreements), 'disagreement_ids':numeric_disagreements},
            'sampling':{'seed':SEED, 'size':SAMPLE_SIZE,
                        'strata':['model','arm','thinking','request_type','prefill','attempt'],
                        'allocation':'round_robin_nonempty_strata', 'blind_id':'sha256(model,id)[:24]'}}


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate label field: {key}')
        result[key] = value
    return result


def load_labels(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        with path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            if len(fields) != len(set(fields)):
                raise ValueError('duplicate CSV label fields')
            return list(reader)
    # read_rows uses binary floats; label decimals must survive JSON decoding.
    paths = sorted(p for p in path.rglob('*.jsonl') if p.name != 'requests.jsonl') if path.is_dir() else [path]
    if not paths:
        raise ValueError(f'no JSONL label files at {path}')
    labels = []
    for source in paths:
        content = source.read_text(encoding='utf-8')
        if content.lstrip().startswith('['):
            labels.extend(json.loads(content, parse_float=_number_label, parse_constant=_number_label,
                                     object_pairs_hook=_unique_fields))
        else:
            labels.extend(json.loads(line, parse_float=_number_label, parse_constant=_number_label,
                                     object_pairs_hook=_unique_fields)
                          for line in content.splitlines() if line.strip())
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--sample', required=True, type=Path)
    parser.add_argument('--labels', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        report = agreement(list(read_rows(args.input)), load_labels(args.labels), sample=list(read_rows(args.sample)))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))
    print(f"Parser agreement: {report['status']}; {report['n_labeled']}/60 labeled -> {args.output}")


if __name__ == '__main__':
    main()
