from copy import deepcopy
import hashlib
import json
import pytest
from tests.test_modal_second_family_5 import grid


def fixture():
    from src.phase5_continuation_5 import provenance, ids_sha256
    rows=[]
    for r in grid():
        controls=r['prefill'] in ('neutral','irrelevant')
        meta=dict(run_id='controls' if controls else 'prefix',phase='phase5',stage='main',pilot=False,
            package_versions={'vllm':'0.32.0'},render_date='2026-10-21',registration_commit=None,
            source_sha256={'DESIGN.md':('b' if controls else 'a')*64,'prompts/templates.json':'c'*64,'prompts/templates_4.json':'d'*64,'prompts/system_tutor.txt':'e'*64},
            implementation_sha256={k:'f'*64 for k in ('src/models_5.py','src/phase5_split.py','src/vast_generate_5.py')},
            engine={'batch_size':256,'chunk_size':256,'enforce_eager':False})
        rows.append(r|meta)
    parts=[]
    for run in ('prefix','controls'):
        selected=[r for r in rows if r['run_id']==run]
        parts.append(dict(run_id=run,n_rows=len(selected),request_ids_sha256=ids_sha256(selected),provenance=provenance(selected[0])))
    return rows,dict(schema=1,kind='phase5_nemotron_controls_continuation',deviation_commit='8219501',parts=parts)


def test_strict_continuation_preserves_two_complete_runs():
    from src.phase5_continuation_5 import validate_continuation
    rows,proof=fixture();original=deepcopy(rows)
    validate_continuation(rows,proof)
    assert rows==original


@pytest.mark.parametrize('kind',['short','mixed','prompt','worker','engine','ids'])
def test_continuation_rejects_shortened_changed_or_unproven_input(kind):
    from src.phase5_continuation_5 import validate_continuation
    rows,p=fixture()
    if kind=='short':rows.pop()
    if kind=='mixed':rows[-1]['run_id']='rogue'
    if kind=='prompt':p['parts'][1]['provenance']['source_sha256']['prompts/system_tutor.txt']='9'*64
    if kind=='worker':p['parts'][1]['provenance']['implementation_sha256']['src/vast_generate_5.py']='9'*64
    if kind=='engine':p['parts'][1]['provenance']['engine']['enforce_eager']=True
    if kind=='ids':p['parts'][1]['request_ids_sha256']='9'*64
    with pytest.raises(ValueError):validate_continuation(rows,p)


def test_analyzer_retains_default_mixed_run_rejection_and_accepts_explicit_proof():
    from tests.test_phase5_analysis import grid as analysis_grid
    from src.score_5 import score_rows
    from src.analyze_5 import _validate_pins
    from src.models_5 import NEMOTRON
    from src.phase5_continuation_5 import provenance,ids_sha256
    rows=[r for r in analysis_grid(200) if r['model']==NEMOTRON]
    _,p=fixture()
    for r in rows:
        part=p['parts'][1 if r['prefill'] in ('neutral','irrelevant') else 0]
        for k in ('run_id','source_sha256','implementation_sha256','engine','stage','pilot'):
            r[k]=deepcopy(part['provenance'][k])
    scored=score_rows(rows)
    for part in p['parts']:
        selected=[r for r in scored if r['run_id']==part['run_id']]
        part['provenance']=provenance(selected[0]);part['request_ids_sha256']=ids_sha256(selected)
    with pytest.raises(ValueError,match='mixed'):_validate_pins(scored,pilot=False)
    _validate_pins(scored,pilot=False,continuation=p)


@pytest.mark.parametrize('field',['source_sha256','implementation_sha256','engine'])
def test_malformed_provenance_is_rejected(field):
    from src.phase5_continuation_5 import validate_continuation
    rows,p=fixture()
    p['parts'][0]['provenance'][field]=None
    for r in rows:
        if r['run_id']=='prefix':r[field]=None
    with pytest.raises(ValueError):validate_continuation(rows,p)
