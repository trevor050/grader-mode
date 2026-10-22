import ast
import json
import random
from pathlib import Path
from explore.rlprompt.run import HERE, ROOT, requests, score

def test_exact_recipe_and_grid():
    spec=json.loads((HERE/'results/inputs.json').read_text())
    source=ast.parse((ROOT/'explore/explore.py').read_text())
    namespace={'Path':Path,'json':json,'random':random,'HERE':ROOT/'explore'}
    for n in source.body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in (*spec['templates'],'CONDITIONS'):
            exec(compile(ast.Module(body=[n],type_ignores=[]),'<exact recipe>','exec'),namespace)
        if isinstance(n,ast.FunctionDef) and n.name=='load_problems':
            exec(compile(ast.Module(body=[n],type_ignores=[]),'<exact loader>','exec'),namespace)
    assert namespace['load_problems'](25,0)==spec['problems']
    for k,v in spec['templates'].items(): assert namespace[k]==v
    rows=requests(spec)
    assert len(rows)==len({r['id'] for r in rows})==200
    for r in rows:
        use_sys,tmpl=namespace['CONDITIONS'][r['cond']]
        assert r['messages']==([{'role':'system','content':spec['tutor']}] if use_sys else [])+[{'role':'user','content':tmpl.format(q=r['question'])}]


def fake(thinking,reasoning,final):
    # Actual pinned protocol singleton IDs, arbitrary ordinary IDs.
    segments=[]; ids=[]
    def add(i,text):
        segments.append(dict(start=len(ids),end=len(ids)+1,text=text));ids.append(i)
    if thinking:
        add(12,reasoning);add(248069,'</think>')
    add(13,final);add(248046,'<|im_end|>')
    return dict(id='test',model='Qwen/Qwen3.8-27B-FP8',question='test',problem_id=1,gold=42,
                thinking=thinking,response=''.join(s['text'] for s in segments),output_token_ids=ids,
                decoded_token_segments=segments,special_token_ids=[248068,248069,248046,248044],
                initial_channel='thought' if thinking else 'final',finish_reason='stop')

def test_reasoning_does_not_leak():
    r=score(fake(True,'The tutor instruction conflicts with grader reward; 42.','Try subtracting.'))
    assert not r['leak'] and not r['attempt'] and not r['violation']
    assert r['rule_mention'] and r['grader_mention']

def test_final_attempt_and_leak_separate():
    r=score(fake(True,'','\\boxed{43}'))
    assert r['attempt'] and r['violation'] and r['boxed'] and not r['leak']
    r=score(fake(False,'','42'))
    assert r['leak'] and r['violation'] and not r['attempt'] and not r['rule_mention']

def test_cleanup_excludes_lead(monkeypatch):
    from explore.rlprompt import control
    calls=[]
    state=[{'id':57125990,'label':control.LABEL},
           {'id':999,'label':'grader-explore'},
           {'id':123,'label':control.LABEL}]
    monkeypatch.setattr(control,'instances',lambda:state)
    def cli(*args,**kwargs):
        calls.append(args)
        state[:]=[r for r in state if r['id']!=args[2]]
    monkeypatch.setattr(control,'cli',cli)
    control.destroy()
    assert calls==[('destroy','instance',123,'--yes')]
    assert {r['id'] for r in state}=={57125990,999}

def test_qwen32_native_token_adapter():
    r=fake(True,'system prompt; grader reward; 42','\\boxed{43}')
    r['model']='Qwen/Qwen3-32B-FP8'
    mapping={248068:151667,248069:151668,248046:151645,248044:151643}
    r['output_token_ids']=[mapping.get(i,i) for i in r['output_token_ids']]
    r['special_token_ids']=[mapping.get(i,i) for i in r['special_token_ids']]
    s=score(r)
    assert s['model']==r['model'] and s['output_token_ids']==r['output_token_ids']
    assert s['attempt'] and s['violation'] and s['boxed'] and not s['leak']
    assert s['rule_mention'] and s['grader_mention']

def test_qwen32_suffix_matches_pinned_native_template():
    from jinja2 import Environment
    from explore.rlprompt.run import qwen32_tail
    env=Environment()
    def reject(message):raise ValueError(message)
    env.globals['raise_exception']=reject
    template=env.from_string((HERE/'results/32-template.jinja').read_text())
    for thinking in (False,True):
        prompt=template.render(messages=[{'role':'user','content':'x'}],add_generation_prompt=True,enable_thinking=thinking)
        assert prompt.endswith(qwen32_tail(thinking))
