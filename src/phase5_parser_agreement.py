"""Blind seed-0 splitter and attempt audits: 40 replies per Phase 5 model.

Small pilots sample every available reply and remain pending even if every
available reply is labeled. The legacy attempt export has only
id/problem/gold/final_reply. The separate splitter export shows the same blind
IDs with raw continuation, initial channel and saved token evidence; neither
export includes model/arm/prefill or automatic boolean metrics.

Lead labels use id, attempt, optional committed_number and human final_reply.
Numbers compare exact Decimals against parser-detected values, not gold.
Human finals compare exact text after trimming outer whitespace. Missing/null
final_reply stays pending; an explicit empty string is a real boundary label.
Top-level status retains the attempt-only API; splitter status is independent
and also required before inference. Completion records coverage, not accuracy;
no agreement cutoff was preregistered. Labels are never generated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from src.score import read_rows
from src.models_5 import MODELS_5
from src.score_5 import active_models, attempted_values, score_rows
from src.phase4_parser_agreement import _label, _number_label, load_labels

SAMPLE_SIZE = 40
SEED = 0


def _blind_id(row):
    key = json.dumps([row['model'], row['id']], ensure_ascii=False, separators=(',', ':'))
    return 'p5-' + hashlib.sha256(key.encode()).hexdigest()[:24]


def _selected(rows):
    selected = []
    for model in active_models(rows):
        strata = defaultdict(list)
        for r in sorted((r for r in rows if r['model'] == model), key=lambda r: r['id']):
            strata[(r['arm'], r['thinking'], r['format'], r['prefill'], r['attempt'])].append(r)
        rng = random.Random(SEED)
        groups = [strata[k] for k in sorted(strata)]
        for group in groups:
            rng.shuffle(group)
        rng.shuffle(groups)
        count = 0
        target = min(SAMPLE_SIZE, sum(map(len, groups)))
        while count < target:
            for group in groups:
                if group and count < target:
                    selected.append(group.pop())
                    count += 1
    random.Random(SEED).shuffle(selected)
    return selected


def _export(selected):
    return [{'id': _blind_id(r), 'problem': r['question'], 'gold': r['gold'],
             'final_reply': r['final_reply']} for r in selected]


def audit_sample(rows):
    return _export(_selected(score_rows(rows)))


def _export_splitter(selected):
    return [{'id':_blind_id(r), 'problem':r['question'], 'gold':r['gold'],
             'initial_channel':r['initial_channel'], 'raw_continuation':r['response'],
             'raw_token_ids':r['output_token_ids'],
             'special_token_ids':sorted(set(r['special_token_ids']) |
                 set(MODELS_5[r['model']]['special_tokens'].values())),
             'decoded_token_segments':r['decoded_token_segments'],
             'finish_reason':r['finish_reason']} for r in selected]


def splitter_sample(rows):
    """Export raw boundary evidence, paired by ID with the legacy attempt sample."""
    return _export_splitter(_selected(score_rows(rows)))


def _write_sample(sample, path):
    path = Path(path)
    if path.exists():
        if list(read_rows(path)) != sample:
            raise ValueError(f'existing audit sample differs at {path}; use a fresh output directory')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in sample))
    return sample


def write_audit_sample(rows, path):
    return _write_sample(audit_sample(rows), path)


def write_splitter_sample(rows, path):
    """Write once (or verify an identical export); never create human labels."""
    return _write_sample(splitter_sample(rows), path)


def _counts(pairs, numeric_pairs, *, sampled, target):
    tp = sum(p and h for _, p, h in pairs)
    tn = sum(not p and not h for _, p, h in pairs)
    fp = sum(p and not h for _, p, h in pairs)
    fn = sum(not p and h for _, p, h in pairs)
    n, nn = len(pairs), len(numeric_pairs)
    observed = (tp + tn) / n if n else None
    chance = ((tp+fp)*(tp+fn)+(tn+fn)*(tn+fp))/n**2 if n else None
    numeric_pending = target - n + tp + fn - nn
    disagreements = [rid for rid, matches in numeric_pairs if not matches]
    return {'status': 'complete' if n == target else 'pending',
        'n_required': target, 'n_sampled': sampled, 'n_labeled': n, 'n_pending': target-n,
        'agreement': observed,
        'cohen_kappa': (observed-chance)/(1-chance) if chance is not None and chance < 1 else None,
        'confusion': {'true_positive':tp, 'true_negative':tn, 'false_positive':fp, 'false_negative':fn},
        'n_disagreements': fp+fn, 'disagreement_ids':[rid for rid,p,h in pairs if p != h],
        'committed_number': {'status':'complete' if numeric_pending == 0 else 'pending',
            'n_eligible':tp+fn, 'n_labeled':nn, 'n_pending':numeric_pending,
            'agreement':(nn-len(disagreements))/nn if nn else None,
            'n_disagreements':len(disagreements), 'disagreement_ids':disagreements}}


def _splitter_counts(pairs, *, sampled, target):
    n = len(pairs)
    disagreements = [rid for rid, matches in pairs if not matches]
    return {'status':'complete' if n == target else 'pending',
        'n_required':target, 'n_sampled':sampled, 'n_labeled':n, 'n_pending':target-n,
        'agreement':(n-len(disagreements))/n if n else None,
        'n_agreements':n-len(disagreements), 'n_disagreements':len(disagreements),
        'disagreement_ids':disagreements, 'comparison':'exact_trimmed_final_reply'}


def agreement(rows, labels=(), *, sample=None, splitter_sample=None, model_scope=None):
    selected = _selected(score_rows(rows))
    if sample is not None and sample != _export(selected):
        raise ValueError('audit sample differs from deterministic raw-row sample')
    if splitter_sample is not None and splitter_sample != _export_splitter(selected):
        raise ValueError('splitter sample differs from deterministic raw-row sample')
    models = active_models(selected)
    if model_scope is not None:
        if model_scope not in MODELS_5 or {r['model'] for r in rows} != {model_scope}:
            raise ValueError('model_scope requires exactly one requested registered family')
        models = (model_scope,)
    predictions = {_blind_id(r):r for r in selected}
    pairs, numeric_pairs, splitter_pairs, seen = [], [], [], set()
    for label in labels:
        if (not isinstance(label, dict) or 'id' not in label
                or not {'attempt','final_reply'} & label.keys()):
            raise ValueError('lead labels require id and attempt or final_reply')
        rid = label['id']
        if not isinstance(rid, str) or rid not in predictions:
            raise ValueError(f'unknown audit label id: {rid}')
        if rid in seen:
            raise ValueError(f'duplicate audit label id: {rid}')
        seen.add(rid)
        final = label.get('final_reply')
        if final is not None:
            if not isinstance(final,str):
                raise ValueError(f'human final_reply must be text or null for {rid}')
            splitter_pairs.append((rid, final.strip() == predictions[rid]['final_reply'].strip()))
        attempt, number = _label(label.get('attempt')), _number_label(label.get('committed_number'))
        if number is not None:
            if attempt is not True:
                raise ValueError(f'committed_number requires attempt yes/true for {rid}')
            numeric_pairs.append((rid, number in attempted_values(predictions[rid]['final_reply'])))
        if attempt is not None:
            pairs.append((rid, predictions[rid]['attempt'], attempt))
    result = _counts(pairs, numeric_pairs, sampled=len(selected), target=SAMPLE_SIZE*len(models))
    result['splitter'] = _splitter_counts(splitter_pairs, sampled=len(selected), target=SAMPLE_SIZE*len(models))
    result['models'] = {}
    for model in models:
        ids = {rid for rid,r in predictions.items() if r['model'] == model}
        result['models'][model] = _counts([p for p in pairs if p[0] in ids],
            [p for p in numeric_pairs if p[0] in ids], sampled=len(ids), target=SAMPLE_SIZE)
        result['models'][model]['splitter'] = _splitter_counts(
            [p for p in splitter_pairs if p[0] in ids], sampled=len(ids), target=SAMPLE_SIZE)
    result['sampling'] = {'seed':SEED, 'size_per_model':SAMPLE_SIZE,
        'strata':['model','arm','thinking','format','prefill','attempt'],
        'allocation':'round_robin_nonempty_strata', 'blind_id':'sha256(model,id)[:24]'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--sample', required=True, type=Path)
    parser.add_argument('--splitter-sample', type=Path,
        help='Raw continuation export; defaults to splitter-audit-sample.jsonl beside --sample')
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        rows = list(read_rows(args.input))
        sample = write_audit_sample(rows, args.sample)
        splitter_path = args.splitter_sample or args.sample.with_name('splitter-audit-sample.jsonl')
        splitter = write_splitter_sample(rows, splitter_path)
        report = agreement(rows, load_labels(args.labels) if args.labels else (),
            sample=sample, splitter_sample=splitter)
        report['sample_path'] = str(args.sample)
        report['splitter']['sample_path'] = str(splitter_path)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))
    print(f"Parser agreement: {report['status']}; {report['n_labeled']}/80 attempt labels; "
          f"splitter: {report['splitter']['status']}; {report['splitter']['n_labeled']}/80 human finals -> {args.output}")


if __name__ == '__main__':
    main()
