"""Compact tables for the authorized, sequential Phase 5 family reports."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from src.models_5 import GEMMA


def interval(value):
    point=value.get('estimate')
    low,high=value.get('ci95',[None,None])
    if point is None:
        return 'unavailable'
    return f'{point:.4f} [{low:.4f}, {high:.4f}]' if low is not None else f'{point:.4f} [pending]'


def render(report):
    model=next(iter(report['models']))
    gemma=model==GEMMA
    interaction,reminder=('H18','H19') if gemma else ('H21','H20')
    audit=report['parser_agreement']
    lines=[f'# Phase 5: {model}', '',
           '| Collection | Value |','| --- | --- |',
           f"| Responses | {report['n_responses']} |",
           f"| Problems | {report['n_problems']} |",
           f"| Grid | {report['grid']['mode']} |",
           f"| Inference eligible | {report['inference']['eligible']} |",
           f"| Pending reasons | {', '.join(report['inference']['reasons']) or 'none'} |",
           '| Bootstrap | 2,000 paired problem clusters; seed 0; percentile 95% CI |',
           '| Denominator | All requests, including cap-hit and empty finals |',
           '| Audit author | Codex, self-labeled blind to arm at the user\'s explicit request; no independent human labels |',
           '', '| Audit | Labeled / required | Agreement | Disagreements |',
           '| --- | ---: | ---: | ---: |']
    for title,item in [('Attempt',audit),('Committed number',audit['committed_number']),('Final boundary',audit['splitter'])]:
        agreement=item.get('agreement')
        confusion=item.get('confusion',{})
        disagreements=(len(item['disagreement_ids']) if 'disagreement_ids' in item else
                       confusion.get('false_positive',0)+confusion.get('false_negative',0))
        agreement_text=(agreement if agreement is not None else
                        'not applicable (0 eligible)' if item.get('status')=='complete' and item.get('n_eligible')==0 else 'pending')
        lines.append(f"| {title} | {item.get('n_labeled',0)} / {item.get('n_required',item.get('n_eligible','NA'))} | {agreement_text} | {disagreements} |")
    lines+=['','| Dropped cell |','| --- |']
    for cell in report['grid']['dropped_cells']:
        lines.append(f"| {cell['arm']} / {'on' if cell['thinking'] else 'off'} / {cell['format']} / {cell['prefill']} |")
    if not report['grid']['dropped_cells']:
        lines.append('| none |')
    for title,data in [('All gold',report),('Gold >= 10',report['gold_ge_10'])]:
        item=data['models'].get(model)
        if not item:
            continue
        lines+=['',f'## {title}', '',
                f"| Problems | Replies |\n| ---: | ---: |\n| {data['n_problems']} | {data['n_responses']} |",'',
                '| Arm | Thinking | Request | Prefill | N | Component | Count | Estimate [95% CI] |',
                '| --- | --- | --- | --- | ---: | --- | ---: | --- |']
        for cell in item['cells']:
            for metric,value in cell['metrics'].items():
                lines.append(f"| {cell['arm']} | {cell['thinking_mode']} | {cell['request_type']} | {cell['prefill']} | {cell['n_responses']} | {metric} | {value.get('count','')} | {interval(value)} |")
        lines+=['','| Outcome | Contrast | Estimate [95% CI] | Tested | Supported |',
                '| --- | --- | --- | --- | --- |']
        for metric,contrasts in item['contrasts'].items():
            for name,value in contrasts.items():
                lines.append(f"| {metric} | {name} | {interval(value)} | {value['tested']} | {value['holds']} |")
    contrasts=report['models'][model]['contrasts']['violation']
    for h in (interaction,reminder):
        if h==interaction:
            value=contrasts[h]
            status='supported' if value['holds'] else 'not supported' if value['holds'] is False else 'unavailable/pending'
            detail=interval(value)
        else:
            key=f'{h}_reduced' if f'{h}_reduced' in contrasts else f'{h}_joint'
            value=contrasts[key]
            status='supported' if value['holds'] else 'not supported' if value['holds'] is False else 'unavailable/pending'
            comparators=['empty'] if key.endswith('reduced') else ['empty','neutral','irrelevant']
            detail='; '.join(f"{c} minus reminder {interval(contrasts[f'{h}_{c}_minus_reminder'])}" for c in comparators)
            if key.endswith('reduced'):
                detail+='; registered reduced contrast'
        lines+=['',f'{h}: {status}, {detail}.']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads(args.input.read_text())
    if len(report['models'])!=1:
        parser.error('a single-family report is required')
    args.output.write_text(render(report))


if __name__=='__main__':
    main()
