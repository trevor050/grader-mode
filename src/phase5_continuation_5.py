"""Strict provenance for the authorized two-run Nemotron controls continuation.

Raw rows retain both run IDs and source hashes. An explicit manifest proves the
six-cell prefix plus the two pending complete control cells; arbitrary mixed
runs still fail. This never rewrites raw generation or scoring fields.
"""
from collections import Counter
import hashlib
import json
from src.models_5 import NEMOTRON

PROVENANCE_FIELDS=('run_id','phase','stage','pilot','package_versions','render_date',
                   'source_sha256','implementation_sha256','registration_commit','engine')
PROMPTS=('prompts/templates.json','prompts/templates_4.json','prompts/system_tutor.txt')
WORKER=('src/models_5.py','src/vast_generate_5.py','src/phase5_split.py')


def provenance(row):
    return {k:row.get(k) for k in PROVENANCE_FIELDS}


def ids_sha256(rows):
    return hashlib.sha256(json.dumps(sorted(r['id'] for r in rows),separators=(',',':')).encode()).hexdigest()


def cell(row):
    return (f"prefill/{row['prefill']}" if row['arm']=='prefill' else
            f"main/{'on' if row['thinking'] else 'off'}/{row['request_type']}")


def validate_continuation(rows,proof):
    if (not isinstance(proof,dict) or proof.get('schema')!=1
            or proof.get('kind')!='phase5_nemotron_controls_continuation'
            or proof.get('deviation_commit')!='8219501'
            or not isinstance(proof.get('parts'),list) or len(proof['parts'])!=2):
        raise ValueError('explicit registered Nemotron continuation proof required')
    if len(rows)!=3200 or {r['model'] for r in rows}!={NEMOTRON}:
        raise ValueError('continuation requires complete 3200-row Nemotron grid')
    if len({r['id'] for r in rows})!=3200:
        raise ValueError('duplicate continuation request IDs')
    expected_cells=({'prefill/empty','prefill/reminder','main/off/plain','main/off/explicit','main/on/plain','main/on/explicit'},
                    {'prefill/neutral','prefill/irrelevant'})
    names=[part.get('run_id') for part in proof['parts']]
    if any(not isinstance(n,str) or not n for n in names) or len(set(names))!=2 or {r.get('run_id') for r in rows}!=set(names):
        raise ValueError('exactly the two named continuation runs required')
    for part,cells in zip(proof['parts'],expected_cells):
        selected=[r for r in rows if r['run_id']==part['run_id']]
        meta=part.get('provenance')
        if (len(selected)!=len(cells)*400 or part.get('n_rows')!=len(selected)
                or Counter(cell(r) for r in selected)!=Counter({c:400 for c in cells})
                or part.get('request_ids_sha256')!=ids_sha256(selected)
                or not isinstance(meta,dict) or set(meta)!=set(PROVENANCE_FIELDS)
                or any(provenance(r)!=meta for r in selected)):
            raise ValueError('shortened, mixed or unproven continuation part')
        if any(not isinstance(meta[name],dict) for name in ('source_sha256','implementation_sha256','engine')):
            raise ValueError('malformed continuation provenance')
        if (meta['phase']!='phase5' or meta['stage']!='main' or meta['pilot'] is not False
                or meta['engine'].get('batch_size')!=256 or meta['engine'].get('chunk_size')!=256
                or meta['engine'].get('enforce_eager') is not False):
            raise ValueError('continuation must retain full-batch registered main settings')
    a,b=[p['provenance'] for p in proof['parts']]
    for name in ('phase','stage','pilot','package_versions','render_date','registration_commit','engine'):
        if a[name]!=b[name]:raise ValueError('changed shared continuation '+name)
    for group,keys in (('source_sha256',PROMPTS),('implementation_sha256',WORKER)):
        for key in keys:
            left,right=a[group].get(key),b[group].get(key)
            if (not isinstance(left,str) or len(left)!=64 or any(c not in '0123456789abcdef' for c in left)
                    or left!=right):
                raise ValueError('changed registered continuation input: '+key)
    return True
