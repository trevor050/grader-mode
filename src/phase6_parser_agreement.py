"""Seed-0 blind final-reply audit, 40 per retained Phase 6 model.

Export contains exactly id/problem/gold/final_reply. Deterministic sampling
round-robins nonempty experiment/thinking/request/prefill/attempt strata within
each model; opaque IDs hide all factors. Parent labels after export, without
parser predictions: id, attempt, optional committed_number. Labels are Codex
self-labels, not independent human labels. No labels are generated here.
Missing labels/small collections stay pending. Decimal agreement checks exact
membership in detected attempts, including incorrect numeric answers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from src.score import read_rows
from src.score_6 import score_rows, attempted_values
from src.phase4_parser_agreement import _label, _number_label, load_labels
from src.phase5_parser_agreement import _counts

SAMPLE_SIZE=40
SEED=0


def _blind_id(row):
    identity=json.dumps([row['model'],row['id']],ensure_ascii=False,separators=(',',':'))
    return 'p6-'+hashlib.sha256(identity.encode()).hexdigest()[:24]


def _selected(rows):
    selected=[]
    for model in sorted({r['model'] for r in rows}):
        strata=defaultdict(list)
        for r in sorted((r for r in rows if r['model']==model),key=lambda r:r['id']):
            strata[(r['experiment'],r['thinking'],r['request_type'],r['prefill_type'],r['attempt'])].append(r)
        rng=random.Random(SEED);groups=[strata[k] for k in sorted(strata)]
        for group in groups:rng.shuffle(group)
        rng.shuffle(groups)
        count=0;target=min(SAMPLE_SIZE,sum(map(len,groups)))
        while count<target:
            for group in groups:
                if group and count<target:selected.append(group.pop());count+=1
    random.Random(SEED).shuffle(selected)
    return selected


def _export(rows):
    return [{'id':_blind_id(r),'problem':r['question'],'gold':r['gold'],'final_reply':r['final_reply']} for r in rows]


def audit_sample(rows):return _export(_selected(score_rows(rows)))


def write_audit_sample(rows,path):
    sample=audit_sample(rows);path=Path(path)
    if path.exists():
        if list(read_rows(path))!=sample:raise ValueError(f'existing audit sample differs at {path}')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in sample))
    return sample


def agreement(rows,labels=(),*,sample=None):
    selected=_selected(score_rows(rows));expected=_export(selected)
    if sample is not None and sample!=expected:raise ValueError('audit sample differs from deterministic sample')
    predictions={_blind_id(r):r for r in selected};pairs=[];numbers=[];seen=set()
    for label in labels:
        if not isinstance(label,dict) or 'id' not in label or 'attempt' not in label:
            raise ValueError('self-labels require id and attempt')
        rid=label['id']
        if not isinstance(rid,str) or rid not in predictions:raise ValueError(f'unknown audit label id: {rid}')
        if rid in seen:raise ValueError(f'duplicate audit label id: {rid}')
        seen.add(rid);attempt=_label(label['attempt']);number=_number_label(label.get('committed_number'))
        if number is not None:
            if attempt is not True:raise ValueError('committed_number requires attempt yes')
            numbers.append((rid,number in attempted_values(predictions[rid]['final_reply'])))
        if attempt is not None:pairs.append((rid,predictions[rid]['attempt'],attempt))
    models=sorted({r['model'] for r in selected})
    report=_counts(pairs,numbers,sampled=len(selected),target=SAMPLE_SIZE*len(models))
    report.update(label_source='codex_self_labels',independent_human_labels=False,
        sampling={'seed':SEED,'size_per_model':SAMPLE_SIZE,
            'strata':['experiment','thinking','request_type','prefill_type','attempt'],
            'allocation':'round_robin_nonempty_strata','blind_id':'sha256(model,id)[:24]'},models={})
    for model in models:
        ids={rid for rid,r in predictions.items() if r['model']==model}
        report['models'][model]=_counts([x for x in pairs if x[0] in ids],
            [x for x in numbers if x[0] in ids],sampled=len(ids),target=SAMPLE_SIZE)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path);parser.add_argument('--sample',required=True,type=Path)
    parser.add_argument('--labels',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    try:
        rows=list(read_rows(args.input));sample=write_audit_sample(rows,args.sample)
        report=agreement(rows,load_labels(args.labels) if args.labels else (),sample=sample)
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    except (ValueError,OSError,TypeError) as error:parser.error(str(error))
    print(f"Parser self-label agreement: {report['status']}; {report['n_labeled']}/{report['n_required']} labeled")


if __name__=='__main__':main()
