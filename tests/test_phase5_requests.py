"""Phase 5 construction and CPU-only generation contract tests."""
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.models_5 import GEMMA, NEMOTRON, GLM, model_spec


def problems(n=200):
    return [{'problem_idx': i * 3, 'question': f'Question {i}?', 'gold': i + 20} for i in range(n)]


def test_registry_pins_and_stop_policy():
    from src.models_5 import (MODELS_5, SAMPLING, VLLM_VERSION, PRIMARY_MODELS,
                              ACTIVE_MODELS, FALLBACK_MODEL, PACKAGE_VERSIONS, resolve_model)
    assert set(MODELS_5) == {GEMMA, NEMOTRON, GLM}
    assert PRIMARY_MODELS == ACTIVE_MODELS == (GEMMA, NEMOTRON) and FALLBACK_MODEL == GLM
    assert MODELS_5[GEMMA]['revision'] == '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'
    assert MODELS_5[NEMOTRON]['revision'] == 'bee7596271d1495f6992ae224aefde4410e816b8'
    assert MODELS_5[GLM]['revision'] == '88e3d3d913c0d97c8f505cdc03433c48226bedc3'
    for key, model in (('gemma', GEMMA), ('nemotron', NEMOTRON), ('glm', GLM)):
        assert resolve_model(key) == resolve_model(model) == model
        spec = model_spec(key)
        assert spec['dtype'] == 'bfloat16' and spec['native_off'] is True
        assert spec['revision'] == spec['tokenizer_revision']
        assert spec['reasoning_effort'] is None
        spec['file_sha256'].clear()
        assert model_spec(model)['file_sha256']  # Detached metadata API.
    with pytest.raises(ValueError): resolve_model('gpt-oss')
    assert MODELS_5[NEMOTRON]['quantization'] is None
    assert MODELS_5[NEMOTRON]['quantization_policy'] == 'checkpoint_config'
    assert MODELS_5[NEMOTRON]['checkpoint_quantization'] == 'modelopt'
    assert MODELS_5[GLM]['quantization'] == 'awq'
    assert MODELS_5[GEMMA]['stop_token_ids'] == [1, 106, 50]
    assert MODELS_5[NEMOTRON]['stop_token_ids'] == [2, 11]
    assert MODELS_5[GLM]['stop_token_ids'] == [154820, 154827, 154829]
    assert SAMPLING == {'temperature': .7, 'top_p': .95, 'top_k': -1, 'max_tokens': 4096}
    assert VLLM_VERSION == '0.32.0'
    assert PACKAGE_VERSIONS == {'vllm':'0.32.0', 'transformers':'5.19.0',
                                'tokenizers':'0.23.3', 'jinja2':'3.1.6'}


@pytest.mark.parametrize('model,count', [(GEMMA,3200),(NEMOTRON,3200),(GLM,3200)])
def test_registered_grid_and_pairing(model, count):
    from src.vast_generate_5 import expand_requests
    rows = expand_requests(problems(), model)
    assert len(rows) == count == len({r['id'] for r in rows})
    assert {r['problem_id'] for r in rows} == {p['problem_idx'] for p in problems()}
    literal = json.loads(Path('prompts/templates_4.json').read_text())
    original = json.loads(Path('prompts/templates.json').read_text())
    for row in rows:
        assert row['phase'] == 'phase5' and row['model'] == model
        assert row['request_type'] in ('plain','explicit')
        assert row['template_idx'] in (0,1)
        expected = original['cues']['none'][row['template_idx']].replace('{problem}',row['question'])
        assert row['messages'][1]['content'] == expected + literal['requests'][row['request_type']]
        if row['arm'] == 'prefill':
            assert row['thinking'] is True and row['request_type'] == 'explicit'
            assert row['prefill_text'] == literal['prefills'][row['prefill']]
        else:
            assert row['prefill'] == 'none' and row['prefill_text'] == ''
    other = [r for m in (GEMMA, NEMOTRON, GLM) if m != model
             for r in expand_requests(problems(), m)]
    for pid in (0,3):
        for template in (0,1):
            assert len({r['seed'] for r in rows+other if r['problem_id']==pid and r['template_idx']==template}) == 1


@pytest.mark.parametrize('model', [GEMMA,NEMOTRON,GLM])
def test_pilot_exactly_16_all_cells_balanced_templates(model):
    from src.vast_generate_5 import expand_requests, pilot_requests
    rows = expand_requests(problems(),model)
    pilot = pilot_requests(rows)
    assert len(pilot) == len({r['id'] for r in pilot}) == 16
    cell = lambda r: (r['arm'],r['thinking'],r['request_type'],r['prefill'])
    assert {cell(r) for r in pilot} == {cell(r) for r in rows}
    assert Counter(r['template_idx'] for r in pilot) == {0:8,1:8}
    for c in {cell(r) for r in pilot}:
        group = [r for r in pilot if cell(r)==c]
        assert Counter(r['template_idx'] for r in group)[0] == Counter(r['template_idx'] for r in group)[1]


def test_build_keeps_full_sample_before_prefix(monkeypatch):
    import datasets
    import src.vast_generate_5 as app
    monkeypatch.setattr(datasets,'load_dataset',lambda *a,**kw: [])
    monkeypatch.setattr(app,'sample_problems',lambda rows: problems())
    rows, manifest = app.build(GEMMA, limit_problems=2)
    assert len(rows) == 32
    assert manifest['sample'] == problems()
    assert manifest['selected_sample'] == problems(2)
    assert manifest['source_sha256']['prompts/templates_4.json']
    assert manifest['amendment'] == 1
    assert manifest['registration_commit'] == 'c3e705068f0c3622c3b5d4cb9575530004ad68ae'
    assert manifest['original_registration_commit'] == 'f15dcc41904f6fad858bdab6c8b7daac09238520'
    for limit in (True,0,201,1.5):
        with pytest.raises(ValueError): app.build(GEMMA,limit_problems=limit)


class RenderTokenizer:
    def __init__(self, model=GEMMA): self.calls=[]; self.model=model
    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages,kwargs))
        if self.model != GEMMA:
            spec = model_spec(self.model)
            return spec['native_on_tail'] if kwargs['enable_thinking'] else spec['native_off_tail']
        return '<|turn>model\n' + ('' if kwargs['enable_thinking'] else '<|channel>thought\n<channel|>')
    def encode(self,text,**kwargs): return list(text.encode())


def test_render_actual_system_native_modes_and_literal_prefills():
    from src.vast_generate_5 import expand_requests, render_prompt, initial_channel
    for model in (GEMMA, NEMOTRON, GLM):
        for row in expand_requests(problems(1), model):
            tok=RenderTokenizer(model); prompt=render_prompt(tok,row)
            messages,kw=tok.calls[0]
            assert kw == {'tokenize':False, 'add_generation_prompt':True,
                          'enable_thinking':row['thinking']}
            assert messages == row['messages'] and messages[0]['role'] == 'system'
            if model != GEMMA:
                tail = model_spec(model)['native_on_tail' if row['thinking'] else 'native_off_tail']
                assert prompt == tail + row['prefill_text']
                assert initial_channel(row) == ('thought' if row['thinking'] else 'final')
                if row['arm'] == 'prefill':
                    opener = '<think>\n' if model == NEMOTRON else '<think>'
                    assert prompt.endswith(opener + row['prefill_text'])
                    assert prompt.count('<think>') == 1
            else:
                if row['arm']=='prefill': assert prompt.endswith('<|channel>thought\n'+row['prefill_text'])
                elif row['thinking']: assert prompt=='<|turn>model\n'
                else: assert prompt.endswith('<|channel>thought\n<channel|>')


@pytest.mark.parametrize('model', [GEMMA, NEMOTRON, GLM])
def test_generate_cpu_fake_retains_raw_ids_and_counts(monkeypatch,tmp_path,model):
    import sys
    import src.vast_generate_5 as app
    from src.models_5 import MODELS_5
    from test_phase5_formats import Tokenizer
    calls=[]
    tok=Tokenizer(model)
    tok.apply_chat_template=RenderTokenizer(model).apply_chat_template
    terminator={GEMMA:'<turn|>', NEMOTRON:'<|im_end|>', GLM:'<|user|>'}[model]
    response='Try a step.'+terminator
    stop=tok.convert_tokens_to_ids(terminator)
    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls,model,**kw):
            assert kw['revision']==MODELS_5[model]['revision']; return tok
    class LLM:
        def __init__(self,**kw): calls.append(('engine',kw))
        def generate(self,prompts,params,**kw):
            calls.append(('batch',params))
            ids=tok.encode(response)
            return [SimpleNamespace(prompt_token_ids=p['prompt_token_ids'],outputs=[SimpleNamespace(
                token_ids=ids,text='wrong stripped text',finish_reason='stop',stop_reason=stop)]) for p in prompts]
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=AutoTokenizer))
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(__version__='0.32.0',LLM=LLM,
        SamplingParams=lambda **kw: SimpleNamespace(**kw)))
    monkeypatch.setattr(app,'runtime_versions',lambda: dict(app.PACKAGE_VERSIONS))
    monkeypatch.setattr(app,'verify_tokenizer',lambda *a:None)
    rows=[r for r in app.expand_requests(problems(1),model) if not r['thinking']]
    metrics=app.generate(model,rows,'test',{'stage':'pilot'}, {'output_dir':str(tmp_path),'batch_size':2,'chunk_size':2})
    raw_path=tmp_path/(model.split('/')[-1]+'.jsonl')
    raw=[json.loads(line) for line in raw_path.read_text().splitlines()]
    assert len(raw)==4 and metrics['requests']==4
    assert metrics['output_tokens']==sum(r['output_tokens'] for r in raw)
    assert metrics['elapsed_seconds']>=metrics['generation_seconds']>0
    assert all(r['response']==response and r['output_token_ids']==tok.encode(r['response']) for r in raw)
    assert all(r['initial_channel']=='final' and r['final_reply']=='Try a step.' for r in raw)
    assert all(r['stage']=='pilot' and r['pilot'] is True and r['thinking'] is False for r in raw)
    assert all(r['rendered_prompt'] and r['prompt_token_ids'] and r['model_revision'] for r in raw)
    engine=calls[0][1]
    assert engine['revision']==engine['tokenizer_revision']==MODELS_5[model]['revision']
    assert engine['generation_config']=='vllm' and engine['max_model_len']==8192
    assert engine['quantization']==MODELS_5[model]['quantization']
    assert engine['dtype']=='bfloat16' and engine['enforce_eager'] is True
    for kind,params in calls[1:]:
        for p in params:
            assert p.max_tokens==4096 and p.skip_special_tokens is False
            assert p.top_k==-1 and p.stop_token_ids==MODELS_5[model]['stop_token_ids']
    with pytest.raises(FileExistsError): app.generate(model,rows,'test',{}, {'output_dir':str(tmp_path)})


def test_cli_pilot_uses_exact_existing_output_dir(monkeypatch,tmp_path):
    import sys
    import src.vast_generate_5 as app
    (tmp_path/'execution.log').write_text('controller log')
    rows=app.expand_requests(problems(2),NEMOTRON)
    monkeypatch.setattr(app,'build',lambda *a: (rows,{'stage':'main'}))
    calls=[]
    monkeypatch.setattr(app,'generate',lambda *a: calls.append(a))
    monkeypatch.setattr(sys,'argv',['worker','--model','nemotron','--pilot','--run-id','ID','--output-dir',str(tmp_path)])
    app.main()
    model,selected,run_id,manifest,config=calls[0]
    assert model==NEMOTRON and len(selected)==16 and run_id=='ID'
    assert manifest['stage']=='pilot' and config['output_dir']==str(tmp_path)
    assert (tmp_path/'execution.log').read_text()=='controller log'


@pytest.mark.parametrize('model', ['gemma', 'nemotron', 'glm'])
def test_cli_full_run_refuses_before_build_or_runtime(monkeypatch, model):
    import sys
    import src.vast_generate_5 as app
    def forbidden(*args, **kwargs):
        raise AssertionError('No build or runtime action is allowed without an explicit mode')
    monkeypatch.setattr(app, 'build', forbidden)
    monkeypatch.setattr(app, 'generate', forbidden)
    monkeypatch.setattr(sys, 'argv', ['worker', '--model', model])
    with pytest.raises(SystemExit) as exc:
        app.main()
    assert exc.value.code == 2
