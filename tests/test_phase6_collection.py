import hashlib,json
import pytest
from src.phase6_collect import collect_stage


def fixture(tmp_path):
    expected=[dict(id='one',question='Original?',model='Qwen/Qwen3.8-27B-FP8')]
    row=expected[0]|dict(stage='main',requests_sha256='fixed',excluded_from_main=False,
                         output_tokens=2,output_token_ids=[7,248046])
    path=tmp_path/'main/chunk-00000.jsonl';path.parent.mkdir()
    path.write_text(json.dumps(row)+'\n')
    item=dict(path='main/chunk-00000.jsonl',count=1,output_tokens=2,
              sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    done=dict(status='completed',requests=1,output_tokens=2,requests_sha256='fixed',chunks=[item])
    (tmp_path/'main.done.json').write_text(json.dumps(done))
    return expected,row,path,done


def test_complete_immutable_chunk_is_verified(tmp_path):
    expected,row,_,_=fixture(tmp_path)
    rows,receipt=collect_stage(tmp_path,'main',expected)
    assert rows==[row] and receipt['requests']==1


@pytest.mark.parametrize('bad',['transfer','pilot','request','completion'])
def test_rejects_corrupt_pilot_mutated_or_incomplete_stage(tmp_path,bad):
    expected,row,path,done=fixture(tmp_path)
    if bad=='transfer':path.write_text('{partial')
    elif bad=='pilot':
        row['excluded_from_main']=True;path.write_text(json.dumps(row)+'\n')
        done['chunks'][0]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    elif bad=='request':expected[0]['question']='Changed?'
    else:done['status']='failed'
    (tmp_path/'main.done.json').write_text(json.dumps(done))
    with pytest.raises(ValueError):collect_stage(tmp_path,'main',expected)
