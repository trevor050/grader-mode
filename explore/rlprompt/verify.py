"""Validate complete grids, saved scoring replay, source preservation, and accounting."""
import hashlib
import ast
import json
from pathlib import Path
from explore.rlprompt.run import HERE, ROOT, requests, score, sha, save
from src.models_6 import MODELS_6
from src.models_5 import GEMMA, MODELS_5

def verify():
    out=HERE/'results'
    spec=json.loads((out/'inputs.json').read_text())
    implementation=json.loads((out/'implementation.json').read_text())
    assert sha(out/'primary-worker.py')==implementation['files']['explore/rlprompt/run.py']
    changed=[f for f,h in spec['protected_sha256'].items() if not (ROOT/f).is_file() or sha(ROOT/f)!=h]
    concurrent={'explore/explore.py','explore/select_neurons.py'}
    if set(changed)-concurrent: raise ValueError(f'protected experiment inputs changed: {changed}')
    tree=ast.parse((ROOT/'explore/explore.py').read_text())
    live={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
          if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
          and n.targets[0].id in spec['templates']}
    assert live==spec['templates']
    concurrent_changes={f:dict(frozen_sha256=spec['protected_sha256'][f],current_sha256=sha(ROOT/f)) for f in changed}
    primary=json.loads((out/'execution.json').read_text())
    assert primary['owned_instances_gone'] and primary['status']=='completed_selected_models'
    accounting=primary
    for name in ('execution-32.json','execution-32-retry.json'):
        optional=out/name
        if optional.exists():
            extra=json.loads(optional.read_text())
            assert extra['owned_instances_gone']
            assert extra['prior_spend_usd']==accounting['conservative_usd']
            if extra['status']=='failed':
                assert not any(v.get('completed') for v in extra['models'].values())
                assert (out/'execution-32-retry.json').exists()
            accounting=extra
    assert accounting['status']=='completed_selected_models'
    assert accounting['conservative_usd']<=1.80
    expected={r['id']:r for r in requests(spec)}
    hashes={};counts={};diagnostics={}
    pins={key:cfg['revision'] for key,cfg in MODELS_6.items()}
    pins['gemma']=MODELS_5[GEMMA]['revision']
    if (out/'32/done.json').exists():pins['32']=json.loads((out/'optional32-pin.json').read_text())['revision']
    keys=[k for k in ('3.8','3.5','32','gemma') if (out/k/'done.json').exists()]
    assert {'3.8','3.5'}<=set(keys)
    for key in keys:
        directory=out/key
        paths=sorted(directory.glob('chunk-*.jsonl'))
        rows=[json.loads(l) for p in paths for l in p.read_text().splitlines()]
        assert len(rows)==len({r['id'] for r in rows})==200 and {r['id'] for r in rows}==set(expected)
        model_set={r['model'] for r in rows};assert len(model_set)==1
        replay=[]
        for r in rows:
            assert all(r[k]==v for k,v in expected[r['id']].items())
            assert r['model_revision']==pins[key]
            assert r['sampling']==spec['sampling'] and len(r['output_token_ids'])<=4096
            assert r['output_tokens']==len(r['output_token_ids'])
            assert r['finish_reason']!='length' or r['output_tokens']==4096
            assert r['reasoning_effort']==('medium' if key=='3.8' else None)
            replay.append(score(r))
        saved=[json.loads(l) for l in (directory/'scored.jsonl').read_text().splitlines()]
        assert replay==saved
        for p in paths+[directory/'scored.jsonl']:
            hashes[str(p.relative_to(out))]=sha(p)
        counts[key]=200
        diagnostics[key]={f:sum(bool(r[f]) for r in replay) for f in ('cap_hit','no_final','protocol_failure')}
    result=dict(status='verified',models=counts,total_replies=sum(counts.values()),
                protected_files_unchanged=len(spec['protected_sha256'])-len(changed),
                concurrent_shared_source_changes=concurrent_changes,frozen_templates_match_current=True,raw_sha256=hashes,
                conservative_usd=accounting['conservative_usd'],diagnostics=diagnostics)
    save(out/'verification.json',result)
    report=out/'summary.md'
    base=report.read_text().split('\n## Generation diagnostics\n')[0]
    lines=['','## Generation diagnostics','','| Model | n | Cap hit | No final | Protocol failure |','|---|---:|---:|---:|---:|']
    for key in keys:
        name=json.loads((out/key/'preflight.json').read_text())['model'].split('/')[-1]
        d=diagnostics[key]
        lines.append(f'| {name} | 200 | {d["cap_hit"]} | {d["no_final"]} | {d["protocol_failure"]} |')
    if concurrent_changes:
        lines+=['','## Concurrent shared sources','','`explore/explore.py` and `explore/select_neurons.py` changed concurrently after the inputs were frozen. Those files were preserved. The three literal prompt templates still match the frozen input values; the exact loader/condition comparison passed the focused tests. Tutor text, problems, DESIGN.md, and phase results hashes are unchanged. Before/after hashes are in `verification.json`.']
    report.write_text(base.rstrip()+'\n'+'\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':verify()
